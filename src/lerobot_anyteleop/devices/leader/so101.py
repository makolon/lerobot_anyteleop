"""Real SO-101 leader arm via the ``lerobot`` package.

``lerobot`` is imported lazily inside :meth:`connect` so the mock pipeline and the
unit tests do not require it (it pulls in torch, motor drivers, etc.).

Notes on units (verified against current ``lerobot`` source):
* ``get_action()`` returns ``{"<joint>.pos": value}`` with the 5 arm joints in
  **degrees** when the teleop is configured with ``use_degrees=True``, and the
  ``gripper`` always in the range ``0..100``.
* The arm-joint degrees are calibration-centered, so the FK zero depends on the
  leader's calibration mid-point. Per-joint sign/offset corrections may be needed
  to align with the URDF zero — verify on hardware.
"""

from __future__ import annotations

import math

import numpy as np

from .base import LeaderInterface, LeaderState

# SO-101 arm joints (gripper handled separately), matching the URDF joint names.
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class SO101Leader(LeaderInterface):
    def __init__(
        self,
        port: str,
        *,
        arm_id: str = "so101_leader",
        calibrate: bool = False,
        calibration_dir: str | None = None,
        joint_sign: dict[str, float] | None = None,
        joint_offset: dict[str, float] | None = None,
    ) -> None:
        self.joint_names = list(SO101_ARM_JOINTS)
        self.port = port
        self.arm_id = arm_id
        self.calibrate = calibrate
        self.calibration_dir = calibration_dir
        self._sign = joint_sign or {}
        self._offset = joint_offset or {}  # radians, applied after sign
        self._dev = None

    def connect(self) -> None:
        from lerobot.teleoperators.so_leader import (  # type: ignore
            SO101Leader as _LeRobotSO101Leader,
            SO101LeaderConfig,
        )

        kwargs = dict(port=self.port, id=self.arm_id, use_degrees=True)
        if self.calibration_dir is not None:
            kwargs["calibration_dir"] = self.calibration_dir
        cfg = SO101LeaderConfig(**kwargs)
        self._dev = _LeRobotSO101Leader(cfg)
        self._dev.connect(calibrate=self.calibrate)

    def disconnect(self) -> None:
        if self._dev is not None:
            self._dev.disconnect()
            self._dev = None

    @property
    def is_connected(self) -> bool:
        return self._dev is not None and getattr(self._dev, "is_connected", True)

    def get_state(self) -> LeaderState:
        if self._dev is None:
            raise RuntimeError("SO101Leader.connect() must be called first.")
        action = self._dev.get_action()  # {"<joint>.pos": value}
        joints: dict[str, float] = {}
        for name in self.joint_names:
            deg = float(action[f"{name}.pos"])
            rad = math.radians(deg)
            rad = self._sign.get(name, 1.0) * rad + self._offset.get(name, 0.0)
            joints[name] = rad
        gripper = float(np.clip(action["gripper.pos"] / 100.0, 0.0, 1.0))
        return LeaderState(joint_positions=joints, gripper=gripper)
