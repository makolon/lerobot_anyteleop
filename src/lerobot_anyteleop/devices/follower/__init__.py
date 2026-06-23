"""Follower drivers, selectable by backend name.

Each driver lazily imports its SDK, so importing this package never requires any
robot SDK. Register a new robot by adding a backend here + a driver module.
"""

from __future__ import annotations

from .base import FollowerInterface

#: backend name -> "module:ClassName" (imported lazily on use)
FOLLOWER_BACKENDS: dict[str, str] = {
    "xarm7": "lerobot_anyteleop.devices.follower.xarm7:XArm7Follower",
    "ur": "lerobot_anyteleop.devices.follower.ur:URFollower",
    "ur5e": "lerobot_anyteleop.devices.follower.ur:URFollower",
    "ur10e": "lerobot_anyteleop.devices.follower.ur:URFollower",
    "franka": "lerobot_anyteleop.devices.follower.franka:FrankaFollower",
    "panda": "lerobot_anyteleop.devices.follower.franka:FrankaFollower",
}


def get_follower_class(backend: str):
    if backend not in FOLLOWER_BACKENDS:
        raise KeyError(
            f"Unknown follower backend {backend!r}. Known: {sorted(FOLLOWER_BACKENDS)}"
        )
    import importlib

    module_path, class_name = FOLLOWER_BACKENDS[backend].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


__all__ = ["FollowerInterface", "FOLLOWER_BACKENDS", "get_follower_class"]
