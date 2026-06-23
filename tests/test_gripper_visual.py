"""Gripper visualization resolution for viser."""

from __future__ import annotations

import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

import numpy as np  # noqa: E402

from lerobot_anyteleop.robots import get_robot_spec  # noqa: E402
from lerobot_anyteleop.viz.gripper_visual import (  # noqa: E402
    finger_targets,
    resolve_follower_visual,
)


def test_finger_targets_mapping():
    fj = {"finger_joint": (0.0, 0.8)}  # open=0.0, closed=0.8
    assert finger_targets(fj, 1.0)["finger_joint"] == 0.0   # fully open
    assert finger_targets(fj, 0.0)["finger_joint"] == 0.8   # fully closed
    assert abs(finger_targets(fj, 0.5)["finger_joint"] - 0.4) < 1e-9
    # clamps out-of-range
    assert finger_targets(fj, 2.0)["finger_joint"] == 0.0
    assert finger_targets(fj, -1.0)["finger_joint"] == 0.8


def test_none_model_is_arm_only():
    fv = resolve_follower_visual(get_robot_spec("xarm7"), "none")
    assert fv.combined and fv.gripper_urdf is None and fv.finger_joints == {}


def _try(fn):
    try:
        return fn()
    except Exception as e:  # network / cache unavailable
        pytest.skip(f"gripper URDF unavailable: {e}")


def test_ur5e_mounts_robotiq():
    fv = _try(lambda: resolve_follower_visual(get_robot_spec("ur5e"), "robotiq_2f85"))
    assert not fv.combined
    assert fv.gripper_urdf is not None
    assert "finger_joint" in fv.finger_joints
    assert len(fv.gripper_urdf.scene.geometry) > 0  # meshes loaded


def test_xarm_native_is_combined():
    fv = _try(lambda: resolve_follower_visual(get_robot_spec("xarm7"), "xarm"))
    assert fv.combined and fv.gripper_urdf is None
    assert "drive_joint" in fv.finger_joints
    assert len(fv.arm_urdf.scene.geometry) > 10  # arm + gripper meshes
    assert "drive_joint" in fv.arm_urdf.actuated_joint_names


def test_mount_default_per_arm():
    # UR tool0 needs no correction; xArm flange gets a yaw correction.
    ur = _try(lambda: resolve_follower_visual(get_robot_spec("ur5e"), "robotiq_2f85"))
    xa = _try(lambda: resolve_follower_visual(get_robot_spec("xarm7"), "robotiq_2f85"))
    assert np.allclose(ur.mount_offset.wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert not np.allclose(xa.mount_offset.wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-3)


def test_panda_robotiq_strips_franka_hand():
    fv = _try(lambda: resolve_follower_visual(get_robot_spec("panda"), "robotiq_2f85"))
    assert not fv.combined and fv.gripper_urdf is not None
    # the built-in Franka Hand is removed so it doesn't double up with the Robotiq
    assert "panda_hand" not in fv.arm_urdf.link_map
    assert "panda_leftfinger" not in fv.arm_urdf.link_map
    assert "finger_joint" in fv.finger_joints  # Robotiq's joint
