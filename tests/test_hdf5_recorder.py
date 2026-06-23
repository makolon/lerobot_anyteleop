from __future__ import annotations

import numpy as np
import h5py

from lerobot_anyteleop.recording import HDF5Recorder


def _make_step(i, h=8, w=8):
    return {
        "observation/follower_qpos": np.full(7, float(i)),
        "observation/follower_ee_pose": np.arange(7, dtype=np.float64) + i,
        "observation/images/top": np.full((h, w, 3), i % 256, dtype=np.uint8),
        "action/follower_qpos": np.full(7, -float(i)),
        "action/gripper": np.array([i / 10.0]),
        "timestamp": np.float64(i * 0.033),
    }


def test_record_and_readback(tmp_path):
    rec = HDF5Recorder(tmp_path, fps=30.0)
    path = rec.start_episode(metadata={"task": "unit"})
    n = 5
    for i in range(n):
        rec.add_step(_make_step(i))
    assert rec.num_steps == n
    rec.end_episode()

    with h5py.File(path, "r") as f:
        assert f["observation/follower_qpos"].shape == (n, 7)
        assert f["observation/images/top"].shape == (n, 8, 8, 3)
        assert f["observation/images/top"].dtype == np.uint8
        # images compressed, vectors not
        assert f["observation/images/top"].compression == "gzip"
        # float dtype downcast to float32
        assert f["observation/follower_qpos"].dtype == np.float32
        # values preserved
        assert np.allclose(f["observation/follower_qpos"][3], 3.0)
        assert int(f["observation/images/top"][4].flat[0]) == 4
        assert f.attrs["num_steps"] == n
        assert f.attrs["task"] == "unit"
        assert f.attrs["fps"] == 30.0


def test_index_autoincrements(tmp_path):
    rec = HDF5Recorder(tmp_path)
    p0 = rec.start_episode()
    rec.add_step(_make_step(0))
    rec.end_episode()
    p1 = rec.start_episode()
    rec.add_step(_make_step(0))
    rec.end_episode()
    assert p0.name == "episode_000000.hdf5"
    assert p1.name == "episode_000001.hdf5"


def test_key_mismatch_raises(tmp_path):
    rec = HDF5Recorder(tmp_path)
    rec.start_episode()
    rec.add_step(_make_step(0))
    import pytest

    with pytest.raises(ValueError):
        rec.add_step({"only": np.zeros(3)})
    rec.abort()


def test_context_manager_and_abort(tmp_path):
    rec = HDF5Recorder(tmp_path)
    with rec.episode(metadata={"task": "ctx"}) as r:
        for i in range(3):
            r.add_step(_make_step(i))
    assert (tmp_path / "episode_000000.hdf5").exists()

    # aborting an episode removes the file
    p = rec.start_episode()
    rec.add_step(_make_step(0))
    rec.abort()
    assert not p.exists()
