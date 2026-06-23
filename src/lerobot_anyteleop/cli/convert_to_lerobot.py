"""``anyteleop-convert`` — convert recorded HDF5 episodes to a LeRobot dataset.

This is a **best-effort scaffold**. The LeRobot dataset API has changed across
releases (v2.0 -> v2.1 -> v3.0); the ``--dry-run`` path (which only needs h5py +
numpy) prints the exact feature spec and per-frame mapping so you can adapt the
write path to whichever ``lerobot`` version you pin.

Default feature mapping (ALOHA-style):

    observation.state            <- /observation/follower_qpos   (measured joints)
    action                       <- /action/follower_qpos        (commanded joints)
    observation.images.<cam>     <- /observation/images/<cam>    (video)

Run ``--include-ee`` to also export EE poses as extra (non-standard) features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

STATE_KEY = "observation/follower_qpos"
ACTION_KEY = "action/follower_qpos"
IMAGES_GROUP = "observation/images"


def _image_camera_names(f: h5py.File) -> list[str]:
    if IMAGES_GROUP not in f:
        return []
    return sorted(f[IMAGES_GROUP].keys())


def _attr(f: h5py.File, key: str, default):
    v = f.attrs.get(key, default)
    if isinstance(v, (bytes, str)):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def build_features(episode_path: Path, fps: float, include_ee: bool) -> dict:
    with h5py.File(episode_path, "r") as f:
        state_dim = f[STATE_KEY].shape[1]
        action_dim = f[ACTION_KEY].shape[1]
        follower_joint_names = _attr(f, "follower_joint_names", [f"j{i}" for i in range(state_dim)])
        features: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": list(follower_joint_names),
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": list(follower_joint_names),
            },
        }
        for cam in _image_camera_names(f):
            h, w, c = f[f"{IMAGES_GROUP}/{cam}"].shape[1:]
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [int(h), int(w), int(c)],
                "names": ["height", "width", "channels"],
            }
        if include_ee:
            features["observation.ee_pose"] = {"dtype": "float32", "shape": [7],
                                               "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]}
            features["action.ee_pose"] = {"dtype": "float32", "shape": [7],
                                          "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]}
    return features


def _iter_frames(f: h5py.File, cams: list[str], include_ee: bool):
    n = f[STATE_KEY].shape[0]
    for i in range(n):
        frame = {
            "observation.state": np.asarray(f[STATE_KEY][i], dtype=np.float32),
            "action": np.asarray(f[ACTION_KEY][i], dtype=np.float32),
        }
        for cam in cams:
            frame[f"observation.images.{cam}"] = np.asarray(f[f"{IMAGES_GROUP}/{cam}"][i])
        if include_ee:
            frame["observation.ee_pose"] = np.asarray(f["observation/follower_ee_pose"][i], np.float32)
            frame["action.ee_pose"] = np.asarray(f["action/follower_ee_pose"][i], np.float32)
        yield frame


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="Directory of episode_*.hdf5 files.")
    p.add_argument("--repo-id", default="local/anyteleop", help="LeRobot dataset repo id.")
    p.add_argument("--root", default=None, help="Output root dir for the LeRobot dataset.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--robot-type", default="xarm7")
    p.add_argument("--include-ee", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print feature spec only; do not write.")
    args = p.parse_args(argv)

    episodes = sorted(Path(args.input_dir).glob("episode_*.hdf5"))
    if not episodes:
        print(f"No episode_*.hdf5 found in {args.input_dir}")
        return 1

    features = build_features(episodes[0], args.fps, args.include_ee)
    print(f"Found {len(episodes)} episode(s).")
    print("Feature spec:")
    print(json.dumps(features, indent=2))

    if args.dry_run:
        print("\n[dry-run] not writing a LeRobot dataset.")
        return 0

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:
        print(f"\nCannot import LeRobotDataset ({e}).")
        print("Install with `pip install lerobot` (pin a version) and adapt the write path "
              "to that release's add_frame/save_episode API. Use --dry-run to inspect mapping.")
        return 1

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.fps),
        root=args.root,
        robot_type=args.robot_type,
        features=features,
        use_videos=True,
    )

    for ep_path in episodes:
        with h5py.File(ep_path, "r") as f:
            cams = _image_camera_names(f)
            task = _attr(f, "task", "teleop")
            for frame in _iter_frames(f, cams, args.include_ee):
                # Newer lerobot wants task as a kwarg; older expects it inside the frame.
                try:
                    dataset.add_frame(frame, task=task)
                except TypeError:
                    dataset.add_frame({**frame, "task": task})
            dataset.save_episode()
        print(f"  converted {ep_path.name}")

    print(f"\nDone. Dataset at: {dataset.root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
