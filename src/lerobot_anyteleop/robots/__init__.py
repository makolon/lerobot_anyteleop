"""Robot kinematic specifications (URDF source, EE link, joints, home)."""

from __future__ import annotations

from .registry import ROBOTS, RobotSpec, get_robot_spec

__all__ = ["ROBOTS", "RobotSpec", "get_robot_spec"]
