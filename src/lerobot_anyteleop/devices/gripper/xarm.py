"""UFACTORY xArm native gripper.

Shares the arm's ``XArmAPI`` connection (you can't open a second one to the same
controller), so it takes the :class:`XArm7Follower` and uses its handle.

Range (xArm Gripper): position 0..850 pulses, **0 = closed, 850 = open**;
speed in r/min (~1..5000).
"""

from __future__ import annotations

from .base import GripperInterface

XARM_GRIPPER_MAX = 850.0


class XArmGripper(GripperInterface):
    deadband = 0.0  # xArm gripper accepts fast streamed targets

    def __init__(self, follower, *, speed: int = 2000) -> None:
        self._follower = follower  # XArm7Follower (provides .api after connect)
        self.speed = int(speed)
        self._arm = None

    def connect(self) -> None:
        self._arm = self._follower.api
        if self._arm is None:
            raise RuntimeError("XArmGripper: follower must be connected first.")
        self._arm.set_gripper_enable(True)
        self._arm.set_gripper_mode(0)
        self._arm.set_gripper_speed(self.speed)

    def disconnect(self) -> None:
        self._arm = None

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    def set_normalized(self, value: float) -> None:
        pos = self._clamp01(value) * XARM_GRIPPER_MAX  # 1 -> 850 (open), 0 -> 0 (closed)
        self._arm.set_gripper_position(round(pos), wait=False)
