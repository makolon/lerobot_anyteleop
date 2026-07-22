"""HDF5 -> LeRobot feature/frame mapping (dry-run path; only h5py + numpy)."""

from __future__ import annotations

import h5py
import numpy as np

from lerobot_anyteleop.cli.convert_to_lerobot import build_features, _iter_frames


def _write_episode(path, *, n=4, arm=7, obs_gripper=True, action_gripper=True):
    with h5py.File(path, "w") as f:
        f["observation/follower_qpos"] = np.tile(np.arange(arm, dtype=np.float32), (n, 1))
        f["action/follower_qpos"] = np.tile(-np.arange(arm, dtype=np.float32), (n, 1))
        f["observation/follower_ee_pose"] = np.zeros((n, 7), np.float32)
        f["action/follower_ee_pose"] = np.zeros((n, 7), np.float32)
        f["observation/images/top"] = np.zeros((n, 4, 4, 3), np.uint8)
        f["timestamp"] = np.arange(n, dtype=np.float64)
        if action_gripper:
            f["action/gripper"] = np.full((n, 1), 0.2, np.float32)   # commanded
        if obs_gripper:
            f["observation/gripper"] = np.full((n, 1), 0.9, np.float32)  # measured
        f.attrs["follower_joint_names"] = '["j0","j1","j2","j3","j4","j5","j6"]'
        f.attrs["task"] = "grab"


def test_gripper_appended_to_state_and_action(tmp_path):
    ep = tmp_path / "episode_000000.hdf5"
    _write_episode(ep)

    features = build_features(ep, fps=30.0, include_ee=False, want_gripper=True)
    assert features["observation.state"]["shape"] == [8]
    assert features["action"]["shape"] == [8]
    assert features["observation.state"]["names"][-1] == "gripper"
    assert features["action"]["names"][-1] == "gripper"

    with h5py.File(ep, "r") as f:
        frames = list(_iter_frames(f, ["top"], include_ee=False, want_gripper=True))
    assert len(frames) == 4
    fr = frames[0]
    assert fr["observation.state"].shape == (8,)
    assert fr["action"].shape == (8,)
    # state gripper comes from measured /observation/gripper (0.9);
    # action gripper from commanded /action/gripper (0.2)
    assert fr["observation.state"][-1] == np.float32(0.9)
    assert fr["action"][-1] == np.float32(0.2)


def test_state_gripper_falls_back_to_commanded(tmp_path):
    # older recording: no /observation/gripper, but /action/gripper exists
    ep = tmp_path / "episode_000000.hdf5"
    _write_episode(ep, obs_gripper=False)

    features = build_features(ep, fps=30.0, include_ee=False, want_gripper=True)
    assert features["observation.state"]["shape"] == [8]
    assert features["action"]["shape"] == [8]

    with h5py.File(ep, "r") as f:
        fr = next(_iter_frames(f, ["top"], include_ee=False, want_gripper=True))
    # both fall back to the commanded gripper (0.2)
    assert fr["observation.state"][-1] == np.float32(0.2)
    assert fr["action"][-1] == np.float32(0.2)


def test_no_gripper_flag_keeps_arm_only(tmp_path):
    ep = tmp_path / "episode_000000.hdf5"
    _write_episode(ep)
    features = build_features(ep, fps=30.0, include_ee=False, want_gripper=False)
    assert features["observation.state"]["shape"] == [7]
    assert features["action"]["shape"] == [7]
    with h5py.File(ep, "r") as f:
        fr = next(_iter_frames(f, ["top"], include_ee=False, want_gripper=False))
    assert fr["observation.state"].shape == (7,)
    assert fr["action"].shape == (7,)


def test_absent_gripper_datasets_arm_only(tmp_path):
    # neither gripper dataset present -> arm-only, no crash
    ep = tmp_path / "episode_000000.hdf5"
    _write_episode(ep, obs_gripper=False, action_gripper=False)
    features = build_features(ep, fps=30.0, include_ee=False, want_gripper=True)
    assert features["action"]["shape"] == [7]
    with h5py.File(ep, "r") as f:
        fr = next(_iter_frames(f, ["top"], include_ee=False, want_gripper=True))
    assert fr["action"].shape == (7,)
