"""Incremental HDF5 episode recorder.

Each episode is one ``episode_XXXXXX.hdf5`` file. Datasets are created lazily from
the first :meth:`add_step` call and grown one row per step (resizable along axis
0), so memory stays flat regardless of episode length. Images are stored per-frame
chunked + compressed.

Layout (keys are whatever you pass to ``add_step``; the controller uses)::

    /observation/follower_qpos     (T, 7)   float32   measured joint angles (rad)
    /observation/follower_ee_pose  (T, 7)   float32   [x,y,z, qw,qx,qy,qz]
    /observation/leader_qpos       (T, 6)   float32   5 arm joints + gripper
    /observation/leader_ee_pose    (T, 7)   float32
    /observation/images/<cam>      (T,H,W,3) uint8    RGB
    /observation/depth/<cam>       (T,H,W)  uint16    (optional)
    /action/follower_qpos          (T, 7)   float32   commanded joint target (the action)
    /action/follower_ee_pose       (T, 7)   float32   retargeted EE target
    /action/gripper                (T, 1)   float32
    /timestamp                     (T,)     float64

This mirrors the common ALOHA-style HDF5 convention, which converts cleanly to
LeRobot datasets (``observation.state`` / ``action`` / ``observation.images.*``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np


def _is_image(elem_shape: tuple[int, ...], dtype: np.dtype) -> bool:
    # Treat any >=2D per-step element as image-like (color HxWx3, depth HxW).
    return len(elem_shape) >= 2


class HDF5Recorder:
    def __init__(
        self,
        output_dir: str | os.PathLike,
        *,
        fps: float = 30.0,
        image_compression: str | None = "gzip",
        compression_opts: int | None = 4,
        float_dtype: str = "float32",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.image_compression = image_compression
        self.compression_opts = compression_opts
        self.float_dtype = np.dtype(float_dtype)

        self._file: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._n_steps = 0
        self._path: Path | None = None
        self._metadata: dict = {}

    # -- properties ---------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return self._n_steps

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def is_recording(self) -> bool:
        return self._file is not None

    # -- lifecycle ----------------------------------------------------------
    def _next_episode_index(self) -> int:
        existing = sorted(self.output_dir.glob("episode_*.hdf5"))
        indices = []
        for p in existing:
            try:
                indices.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return (max(indices) + 1) if indices else 0

    def start_episode(self, episode_index: int | None = None, metadata: dict | None = None) -> Path:
        if self._file is not None:
            raise RuntimeError("An episode is already open; call end_episode() first.")
        idx = self._next_episode_index() if episode_index is None else int(episode_index)
        self._path = self.output_dir / f"episode_{idx:06d}.hdf5"
        self._file = h5py.File(self._path, "w")
        self._datasets = {}
        self._n_steps = 0
        self._metadata = {"episode_index": idx, "fps": self.fps, **(metadata or {})}
        return self._path

    def _create_dataset(self, key: str, value: np.ndarray) -> h5py.Dataset:
        elem_shape = value.shape
        dtype = value.dtype
        maxshape = (None, *elem_shape)
        kwargs: dict = {"maxshape": maxshape}
        if _is_image(elem_shape, dtype):
            kwargs["chunks"] = (1, *elem_shape)
            if self.image_compression:
                kwargs["compression"] = self.image_compression
                if self.compression_opts is not None and self.image_compression == "gzip":
                    kwargs["compression_opts"] = self.compression_opts
        else:
            kwargs["chunks"] = True
        assert self._file is not None
        return self._file.create_dataset(key, shape=(0, *elem_shape), dtype=dtype, **kwargs)

    def _coerce(self, value) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype.kind == "f":
            arr = arr.astype(self.float_dtype, copy=False)
        return arr

    def add_step(self, data: dict[str, object]) -> None:
        """Append one timestep. ``data`` maps dataset path -> array/scalar.

        Keys must be identical across steps within an episode.
        """
        if self._file is None:
            raise RuntimeError("start_episode() must be called before add_step().")

        coerced = {k: self._coerce(v) for k, v in data.items()}

        if self._n_steps == 0:
            for key, arr in coerced.items():
                self._datasets[key] = self._create_dataset(key, arr)
        else:
            missing = set(self._datasets) ^ set(coerced)
            if missing:
                raise ValueError(f"add_step keys changed mid-episode: {sorted(missing)}")

        n = self._n_steps
        for key, arr in coerced.items():
            ds = self._datasets[key]
            ds.resize(n + 1, axis=0)
            ds[n] = arr
        self._n_steps += 1

    def end_episode(self) -> Path:
        if self._file is None:
            raise RuntimeError("No episode is open.")
        self._metadata["num_steps"] = self._n_steps
        self._metadata["datasets"] = sorted(self._datasets.keys())
        for k, v in self._metadata.items():
            self._file.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        path = self._path
        self._file.close()
        self._file = None
        self._datasets = {}
        assert path is not None
        return path

    def abort(self) -> None:
        """Close and delete the current (partial) episode file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._datasets = {}
            if self._path is not None and self._path.exists():
                self._path.unlink()

    # context manager for one episode --------------------------------------
    def episode(self, episode_index: int | None = None, metadata: dict | None = None):
        recorder = self

        class _Ctx:
            def __enter__(self_inner):
                recorder.start_episode(episode_index, metadata)
                return recorder

            def __exit__(self_inner, exc_type, *_):
                if exc_type is not None:
                    recorder.abort()
                else:
                    recorder.end_episode()

        return _Ctx()
