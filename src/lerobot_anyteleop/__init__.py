"""lerobot_anyteleop.

Kinematics-based teleoperation interface that drives a UFACTORY xArm7 follower
from a LeRobot SO-101 leader arm, recording multi-camera RealSense streams and
robot state to HDF5.

Pipeline (per control tick)::

    leader joints --FK--> leader EE pose --retarget(scale)--> follower EE target
                                                       |
                                                       v
    follower joints <--IK-- follower EE target  --> command follower --> record
"""

from __future__ import annotations

__version__ = "0.1.0"

from .transforms import Pose  # noqa: E402

__all__ = ["Pose", "__version__"]
