"""Hardware (and mock) device interfaces: leader, follower, cameras."""

from __future__ import annotations

from .camera.base import CameraFrame, CameraInterface
from .camera.manager import MultiCameraManager
from .follower.base import FollowerInterface
from .leader.base import LeaderInterface, LeaderState

__all__ = [
    "LeaderInterface",
    "LeaderState",
    "FollowerInterface",
    "CameraInterface",
    "CameraFrame",
    "MultiCameraManager",
]
