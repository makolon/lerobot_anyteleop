"""Typed configuration for the teleoperation system, loadable from YAML.

Robots are referenced by name (see :mod:`lerobot_anyteleop.robots`); URDF, EE
link, joint names and home come from the registry. Anything can be overridden
per-config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .transforms import rpy_to_quat


def _filter(cls, d: dict) -> dict:
    return {k: d[k] for k in (d or {}) if k in cls.__dataclass_fields__}


@dataclass
class LeaderConfig:
    robot: str = "so101"
    backend: str = "so101"        # device driver (currently only "so101")
    urdf: str | None = None       # override; else from robot registry
    ee_link: str | None = None
    # real SO-101
    port: str | None = None
    arm_id: str = "so101_leader"
    calibrate: bool = False
    calibration_dir: str | None = None
    joint_sign: dict | None = None
    joint_offset: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LeaderConfig":
        return cls(**_filter(cls, d))


@dataclass
class GripperConfig:
    type: str = "none"            # none | xarm | robotiq | franka
    options: dict = field(default_factory=dict)  # driver kwargs (com_port, host, speed, ...)
    deadband: float | None = None  # override the driver's default deadband

    @classmethod
    def from_dict(cls, d: dict) -> "GripperConfig":
        return cls(**_filter(cls, d))


@dataclass
class FollowerConfig:
    robot: str = "xarm7"          # xarm7 | panda | ur5e | ...
    backend: str | None = None    # device driver; else robot's default backend
    urdf: str | None = None
    ee_link: str | None = None
    home: list[float] | None = None  # arm-DOF home (rad); else from registry
    ip: str | None = None
    options: dict = field(default_factory=dict)  # extra driver kwargs
    gripper: GripperConfig = field(default_factory=GripperConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "FollowerConfig":
        d = dict(d or {})
        gripper = GripperConfig.from_dict(d.pop("gripper", {}))
        obj = cls(**_filter(cls, d))
        obj.gripper = gripper
        return obj


@dataclass
class RetargetConfig:
    position_scale: float = 1.0
    orientation_scale: float = 1.0
    align_rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # rad
    ik_pos_weight: float = 50.0
    ik_ori_weight: float = 10.0
    ik_rest_weight: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "RetargetConfig":
        return cls(**_filter(cls, d))

    def align_wxyz(self) -> np.ndarray:
        return rpy_to_quat(*self.align_rpy)


@dataclass
class CameraConfig:
    name: str
    backend: str = "realsense"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        return cls(**_filter(cls, d))


@dataclass
class RecordConfig:
    output_dir: str = "data/recordings"
    fps: float = 30.0
    image_compression: str | None = "gzip"
    compression_opts: int | None = 4
    record_depth: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "RecordConfig":
        return cls(**_filter(cls, d))


@dataclass
class LoopConfig:
    rate_hz: float = 30.0
    max_steps: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        return cls(**_filter(cls, d))


@dataclass
class TeleopConfig:
    task: str = "teleop"
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    record: RecordConfig = field(default_factory=RecordConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "TeleopConfig":
        d = dict(d or {})
        return cls(
            task=d.get("task", "teleop"),
            leader=LeaderConfig.from_dict(d.get("leader", {})),
            follower=FollowerConfig.from_dict(d.get("follower", {})),
            retarget=RetargetConfig.from_dict(d.get("retarget", {})),
            cameras=[CameraConfig.from_dict(c) for c in d.get("cameras", [])],
            record=RecordConfig.from_dict(d.get("record", {})),
            loop=LoopConfig.from_dict(d.get("loop", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeleopConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
