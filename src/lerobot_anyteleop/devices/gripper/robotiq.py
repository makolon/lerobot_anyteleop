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
    ) -> None:
        self.backend = backend
        self.com_port = com_port
        self.host = host
        self.port = int(port)
        self.speed = int(speed)
        self.force = int(force)
        self._g = None

    def connect(self) -> None:
        if self.backend == "serial":
            import pyrobotiqgripper  # type: ignore

            self._g = pyrobotiqgripper.RobotiqGripper(com_port=self.com_port)
            self._g.activate()
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
        if self._g is not None and hasattr(self._g, "disconnect"):
            try:
                self._g.disconnect()
            except Exception:
                pass
        self._g = None

    @property
    def is_connected(self) -> bool:
        return self._g is not None

    def set_normalized(self, value: float) -> None:
        pos = round((1.0 - self._clamp01(value)) * ROBOTIQ_MAX)  # 1 -> 0 (open), 0 -> 255 (closed)
        if self.backend == "serial":
            self._g.move(pos, speed=self.speed, force=self.force, wait=False)
        else:  # ur socket
            self._g.move(pos, self.speed, self.force)
