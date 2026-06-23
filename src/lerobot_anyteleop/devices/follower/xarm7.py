"""Real UFACTORY xArm7 follower via ``xArm-Python-SDK``.

Control model (verified against the SDK):
* Homing uses **mode 0** + ``set_servo_angle(..., wait=True)``.
* Teleoperation streams joint targets with ``set_servo_angle_j`` in **mode 1**
  (servo motion mode), state 0, at 100–250 Hz. Each target must be a *small*
  increment from the current pose (satisfied by IK solving at the control rate);
  ``speed``/``mvacc`` are ignored in servo mode.

All angles are exchanged in **radians** (``XArmAPI(..., is_radian=True)``).
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import FollowerInterface


class XArm7Follower(FollowerInterface):
    def __init__(
        self,
        ip: str,
        joint_names,
        *,
        use_gripper: bool = False,
        gripper_max: float = 850.0,
        home_speed: float = 0.35,  # rad/s for the homing move
    ) -> None:
        self.joint_names = list(joint_names)
        self.ip = ip
        self.use_gripper = use_gripper
        self.gripper_max = float(gripper_max)
        self.home_speed = float(home_speed)
        self._arm = None

    def connect(self) -> None:
        from xarm.wrapper import XArmAPI  # type: ignore

        self._arm = XArmAPI(self.ip, is_radian=True)
        self._arm.motion_enable(enable=True)
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.set_mode(0)
        self._arm.set_state(0)
        if self.use_gripper:
            self._arm.set_gripper_enable(True)
            self._arm.set_gripper_mode(0)
            self._arm.set_gripper_speed(2000)

    def disconnect(self) -> None:
        if self._arm is not None:
            try:
                self._arm.set_state(4)  # stop
            finally:
                self._arm.disconnect()
            self._arm = None

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and bool(getattr(self._arm, "connected", True))

    def get_joint_positions(self) -> np.ndarray:
        code, angles = self._arm.get_servo_angle(is_radian=True)
        self._check(code, "get_servo_angle")
        return np.asarray(angles[: len(self.joint_names)], dtype=np.float64)

    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        q = np.asarray(q, dtype=np.float64)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        code = self._arm.set_servo_angle(
            angle=q.tolist(), is_radian=True, speed=self.home_speed, wait=blocking
        )
        self._check(code, "set_servo_angle(home)")

    def enter_servo_mode(self) -> None:
        self._arm.set_mode(1)
        self._arm.set_state(0)

    def send_joint_positions(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        code = self._arm.set_servo_angle_j(q.tolist(), is_radian=True)
        self._check(code, "set_servo_angle_j")

    def set_gripper(self, value: float) -> None:
        if not self.use_gripper:
            return
        pos = float(np.clip(value, 0.0, 1.0)) * self.gripper_max
        self._arm.set_gripper_position(pos, wait=False)

    @staticmethod
    def _check(code, what: str) -> None:
        if code not in (0, None):
            warnings.warn(f"xArm {what} returned non-zero code {code}", stacklevel=2)
