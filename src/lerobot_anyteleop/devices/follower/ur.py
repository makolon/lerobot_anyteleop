"""Universal Robots follower (UR5e / UR10e / ...) via ``ur_rtde``.

Verified against ``ur_rtde``:
* Read joints (rad): ``RTDEReceiveInterface.getActualQ()``.
* Home (blocking): ``RTDEControlInterface.moveJ(q, speed, accel)``.
* Stream servo: ``servoJ(q, speed, accel, dt, lookahead_time, gain)`` — ``speed``
  and ``accel`` are ignored but positionally required; ``dt`` is the control
  period, ``lookahead_time`` ~0.1 s, ``gain`` ~100–2000. Stop with ``servoStop()``.

6-DOF, joint order: base, shoulder, elbow, wrist1, wrist2, wrist3.

Hardware note: ``pip install ur_rtde`` (UR e-Series streams at 500 Hz). No gripper
control here (UR grippers are vendor-specific add-ons).
"""

from __future__ import annotations

import numpy as np

from .base import FollowerInterface


class URFollower(FollowerInterface):
    def __init__(
        self,
        ip: str,
        joint_names,
        *,
        servo_dt: float = 1.0 / 125.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        home_speed: float = 1.05,
        home_accel: float = 1.4,
    ) -> None:
        self.joint_names = list(joint_names)
        self.ip = ip
        self.servo_dt = float(servo_dt)
        self.lookahead_time = float(lookahead_time)
        self.gain = float(gain)
        self.home_speed = float(home_speed)
        self.home_accel = float(home_accel)
        self._rtde_c = None
        self._rtde_r = None

    def connect(self) -> None:
        import rtde_control  # type: ignore
        import rtde_receive  # type: ignore

        self._rtde_c = rtde_control.RTDEControlInterface(self.ip)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)

    def disconnect(self) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.servoStop()
                self._rtde_c.stopScript()
            finally:
                self._rtde_c.disconnect()
            self._rtde_c = None
        if self._rtde_r is not None:
            self._rtde_r.disconnect()
            self._rtde_r = None

    @property
    def is_connected(self) -> bool:
        return self._rtde_c is not None and bool(self._rtde_c.isConnected())

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)

    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        q = np.asarray(q, dtype=np.float64).tolist()
        # asynchronous = not blocking
        self._rtde_c.moveJ(q, self.home_speed, self.home_accel, not blocking)

    def enter_servo_mode(self) -> None:
        # servoJ is stateless; nothing to switch. (kept for interface parity)
        pass

    def send_joint_positions(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).tolist()
        # speed/accel are ignored by servoJ but are required positional args.
        self._rtde_c.servoJ(q, 0.5, 0.5, self.servo_dt, self.lookahead_time, self.gain)
