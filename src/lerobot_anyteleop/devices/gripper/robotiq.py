"""Robotiq 2F-85 / 2F-140 gripper.

Register convention: **0 = open, 255 = closed**, so normalized maps as
``raw = round((1 - value) * 255)``.

Two control paths (``backend``):

* ``"serial"`` (default) — Modbus RTU over USB via ``pyRobotiqGripper`` (gripper
  plugged into the PC; ``pip install pyRobotiqGripper``).
* ``"ur"`` — through a Universal Robots controller's URCap socket on port 63352,
  using UR's standalone ``robotiq_gripper.py`` (vendor that file so
  ``import robotiq_gripper`` works; it is not a pip package).
"""

from __future__ import annotations

from .base import GripperInterface

ROBOTIQ_MAX = 255


class RobotiqGripper(GripperInterface):
    deadband = 0.03

    def __init__(
        self,
        *,
        backend: str = "serial",
        com_port: str = "auto",        # serial backend
        host: str | None = None,       # ur backend (robot IP)
        port: int = 63352,             # ur backend
        speed: int = 255,
        force: int = 255,
        device_id: int = 9,            # serial Modbus slave address
        activate_on_connect: bool = True,
    ) -> None:
        self.backend = backend
        self.com_port = com_port
        self.host = host
        self.port = int(port)
        self.speed = int(speed)
        self.force = int(force)
        self.device_id = int(device_id)
        self.activate_on_connect = bool(activate_on_connect)
        if not 0 <= self.speed <= ROBOTIQ_MAX:
            raise ValueError("Robotiq speed must be in [0, 255].")
        if not 0 <= self.force <= ROBOTIQ_MAX:
            raise ValueError("Robotiq force must be in [0, 255].")
        if not 1 <= self.device_id <= 247:
            raise ValueError("Robotiq Modbus device_id must be in [1, 247].")
        self._g = None

    def connect(self) -> None:
        if self.backend == "serial":
            import pyrobotiqgripper  # type: ignore

            gripper = pyrobotiqgripper.RobotiqGripper(
                com_port=self.com_port, device_id=self.device_id
            )
            try:
                # pyRobotiqGripper v3 separates construction from opening the
                # serial transport. activate() before connect() cannot work.
                gripper.connect()
                if self.activate_on_connect:
                    gripper.activate()
            except BaseException:
                try:
                    gripper.disconnect()
                except Exception:
                    pass
                raise
            self._g = gripper
        elif self.backend == "ur":
            import robotiq_gripper  # type: ignore  # vendor UR's robotiq_gripper.py

            if not self.host:
                raise ValueError("RobotiqGripper(backend='ur') requires `host` (robot IP).")
            self._g = robotiq_gripper.RobotiqGripper()
            self._g.connect(self.host, self.port)
            self._g.activate()
        else:
            raise ValueError(f"Unknown Robotiq backend {self.backend!r} (use 'serial' or 'ur').")

    def disconnect(self) -> None:
        gripper = self._g
        if gripper is None:
            return
        stop_error: BaseException | None = None
        # A previous move is asynchronous. pyRobotiqGripper v3 stop() clears
        # go-to so closing cannot continue merely because transport closes.
        if self.backend == "serial" and hasattr(gripper, "stop"):
            try:
                gripper.stop()
            except BaseException as exc:
                stop_error = exc
        disconnect_error: BaseException | None = None
        if hasattr(gripper, "disconnect"):
            try:
                gripper.disconnect()
            except BaseException as exc:
                disconnect_error = exc
        self._g = None
        if stop_error is not None:
            if disconnect_error is not None:
                stop_error.add_note(f"Robotiq disconnect also failed: {disconnect_error}")
            raise stop_error
        if disconnect_error is not None:
            raise disconnect_error

    @property
    def is_connected(self) -> bool:
        return self._g is not None

    def set_normalized(self, value: float) -> None:
        if self._g is None:
            raise RuntimeError("Robotiq gripper is not connected")
        pos = round((1.0 - self._clamp01(value)) * ROBOTIQ_MAX)  # 1 -> 0 (open), 0 -> 255 (closed)
        if self.backend == "serial":
            self._g.move(pos, speed=self.speed, force=self.force, wait=False)
        else:  # ur socket
            self._g.move(pos, self.speed, self.force)
