"""Franka Emika Panda / FR3 follower via ``panda-python`` (``panda_py``).

Verified against ``panda_py``:
* Read joints (7 rad): ``panda.q``.
* Home (blocking): ``panda.move_to_joint_position(q, speed_factor)``.
* Streaming joint control: ``controllers.JointPosition`` started with
  ``panda.start_controller(ctrl)``; push targets each tick via
  ``ctrl.set_control(q)``. Our outer loop provides the timing.

7-DOF. Hardware requirements (inherent to libfranka/FCI): a PREEMPT_RT realtime
kernel, the FCI feature enabled, and a wired connection to the robot subnet.
``pip install panda-python`` (import name ``panda_py``).
"""

from __future__ import annotations

import numpy as np

from .base import FollowerInterface


class FrankaFollower(FollowerInterface):
    def __init__(
        self,
        ip: str,
        joint_names,
        *,
        home_speed_factor: float = 0.2,
    ) -> None:
        self.joint_names = list(joint_names)
        self.ip = ip
        self.home_speed_factor = float(home_speed_factor)
        self._panda = None
        self._ctrl = None

    def connect(self) -> None:
        import panda_py  # type: ignore

        self._panda = panda_py.Panda(self.ip)

    def disconnect(self) -> None:
        if self._panda is not None:
            try:
                if self._ctrl is not None:
                    self._panda.stop_controller()
            finally:
                self._ctrl = None
                self._panda = None

    @property
    def is_connected(self) -> bool:
        return self._panda is not None

    def get_joint_positions(self) -> np.ndarray:
        q = getattr(self._panda, "q", None)
        if q is None:
            q = self._panda.get_state().q
        return np.asarray(q, dtype=np.float64)

    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        q = np.asarray(q, dtype=np.float64).tolist()
        self._panda.move_to_joint_position(q, speed_factor=self.home_speed_factor)

    def enter_servo_mode(self) -> None:
        from panda_py import controllers  # type: ignore

        self._ctrl = controllers.JointPosition()
        self._panda.start_controller(self._ctrl)

    def send_joint_positions(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        self._ctrl.set_control(q)
