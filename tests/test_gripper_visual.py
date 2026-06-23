"""Gripper visualization resolution for viser."""

from __future__ import annotations

import pytest

pytest.importorskip("yourdfpy")
pytest.importorskip("robot_descriptions")

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
