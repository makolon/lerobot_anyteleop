"""Pluggable grippers, decoupled from the arm.

A gripper is an *attachment*: the same arm can carry its native gripper, a Robotiq
2F-85, a Franka Hand, etc. The teleop loop always speaks one language — a
normalized command in ``[0, 1]`` (1 = open, 0 = closed) derived from the SO-101
leader's gripper joint — and each driver maps it to its hardware protocol.

Register a new gripper = add a backend here + a driver module.
"""

from __future__ import annotations

from .base import GripperInterface

#: backend name -> "module:ClassName" (imported lazily on use)
GRIPPER_BACKENDS: dict[str, str] = {
    "none": "lerobot_anyteleop.devices.gripper.none:NoGripper",
    "xarm": "lerobot_anyteleop.devices.gripper.xarm:XArmGripper",
    "robotiq": "lerobot_anyteleop.devices.gripper.robotiq:RobotiqGripper",
    "franka": "lerobot_anyteleop.devices.gripper.franka:FrankaHand",
}


def get_gripper_class(backend: str):
    if backend not in GRIPPER_BACKENDS:
        raise KeyError(f"Unknown gripper backend {backend!r}. Known: {sorted(GRIPPER_BACKENDS)}")
    import importlib

    module_path, class_name = GRIPPER_BACKENDS[backend].split(":")
    return getattr(importlib.import_module(module_path), class_name)


__all__ = ["GripperInterface", "GRIPPER_BACKENDS", "get_gripper_class"]
