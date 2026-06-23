"""Build concrete teleop components from a :class:`TeleopConfig`.

Robot specs (URDF, EE link, joints, home) come from :mod:`lerobot_anyteleop.robots`;
device drivers are selected by backend and imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    CameraConfig,
    FollowerConfig,
    GripperConfig,
    LeaderConfig,
    RecordConfig,
    RetargetConfig,
    TeleopConfig,
)
from .devices import FollowerInterface, LeaderInterface, MultiCameraManager
from .devices.camera.base import CameraInterface
from .devices.gripper.base import GripperInterface
from .kinematics import KinematicsModel
from .recording import HDF5Recorder
from .retargeting import PoseRetargeter
from .robots import RobotSpec, get_robot_spec
from .teleop.pipeline import KinematicRetargetPipeline


@dataclass
class TeleopSystem:
    leader: LeaderInterface
    follower: FollowerInterface
    gripper: GripperInterface
    leader_kin: KinematicsModel
    follower_kin: KinematicsModel
    retargeter: PoseRetargeter
    pipeline: KinematicRetargetPipeline
    cameras: MultiCameraManager
    recorder: HDF5Recorder
    follower_home: np.ndarray
    config: TeleopConfig


# --------------------------------------------------------------------------- #
# Kinematics + retargeting (no hardware) — shared by controller and viser app.
# --------------------------------------------------------------------------- #
def build_kinematics(urdf: str, ee_link: str, retarget: RetargetConfig) -> KinematicsModel:
    from .kinematics.pyroki_model import PyrokiKinematics

    return PyrokiKinematics(
        urdf,
        ee_link,
        pos_weight=retarget.ik_pos_weight,
        ori_weight=retarget.ik_ori_weight,
        rest_weight=retarget.ik_rest_weight,
    )


def _resolve(robot: str, urdf: str | None, ee_link: str | None) -> tuple[RobotSpec, str, str]:
    spec = get_robot_spec(robot)
    return spec, (urdf or spec.urdf), (ee_link or spec.ee_link)


def build_pipeline(
    leader: LeaderConfig, follower: FollowerConfig, retarget: RetargetConfig
) -> tuple[KinematicRetargetPipeline, KinematicsModel, KinematicsModel, RobotSpec, RobotSpec]:
    leader_spec, leader_urdf, leader_ee = _resolve(leader.robot, leader.urdf, leader.ee_link)
    follower_spec, follower_urdf, follower_ee = _resolve(
        follower.robot, follower.urdf, follower.ee_link
    )
    leader_kin = build_kinematics(leader_urdf, leader_ee, retarget)
    follower_kin = build_kinematics(follower_urdf, follower_ee, retarget)
    retargeter = PoseRetargeter(
        position_scale=retarget.position_scale,
        orientation_scale=retarget.orientation_scale,
        align_wxyz=retarget.align_wxyz(),
    )
    pipeline = KinematicRetargetPipeline(
        leader_kin, follower_kin, retargeter, follower_spec.arm_joint_names
    )
    return pipeline, leader_kin, follower_kin, leader_spec, follower_spec


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #
def build_leader_device(cfg: LeaderConfig) -> LeaderInterface:
    if cfg.backend != "so101":
        raise ValueError(f"Unknown leader backend {cfg.backend!r} (only 'so101' is supported).")
    from .devices.leader.so101 import SO101Leader

    if not cfg.port:
        raise ValueError("leader.port is required for the SO-101 leader.")
    return SO101Leader(
        port=cfg.port,
        arm_id=cfg.arm_id,
        calibrate=cfg.calibrate,
        calibration_dir=cfg.calibration_dir,
        joint_sign=cfg.joint_sign,
        joint_offset=cfg.joint_offset,
    )


def build_follower_device(cfg: FollowerConfig, spec: RobotSpec) -> FollowerInterface:
    from .devices.follower import get_follower_class

    backend = cfg.backend or spec.follower_backend
    if not backend:
        raise ValueError(f"No follower backend for robot {spec.name!r}; set follower.backend.")
    if not cfg.ip:
        raise ValueError(f"follower.ip is required for backend {backend!r}.")
    cls = get_follower_class(backend)
    return cls(ip=cfg.ip, joint_names=spec.arm_joint_names, **cfg.options)


def build_gripper(
    cfg: GripperConfig, follower: FollowerInterface, follower_ip: str | None
) -> GripperInterface:
    from .devices.gripper import get_gripper_class

    cls = get_gripper_class(cfg.type)
    opts = dict(cfg.options)
    if cfg.type == "none":
        gripper = cls()
    elif cfg.type == "xarm":
        gripper = cls(follower, **opts)            # shares the arm's connection
    elif cfg.type == "franka":
        ip = opts.pop("ip", follower_ip)
        gripper = cls(ip, **opts)
    elif cfg.type == "robotiq":
        if opts.get("backend") == "ur" and not opts.get("host"):
            opts["host"] = follower_ip             # Robotiq on a UR controller
        gripper = cls(**opts)
    else:  # pragma: no cover - registry guards this
        gripper = cls(**opts)
    if cfg.deadband is not None:
        gripper.deadband = cfg.deadband
    return gripper


def build_camera(cfg: CameraConfig, seed: int = 0) -> CameraInterface:
    if cfg.backend != "realsense":
        raise ValueError(f"Unknown camera backend {cfg.backend!r} (only 'realsense' is supported).")
    from .devices.camera.realsense import RealSenseCamera

    return RealSenseCamera(
        cfg.name,
        serial=cfg.serial,
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        enable_depth=cfg.enable_depth,
    )


def build_cameras(cfgs: list[CameraConfig]) -> MultiCameraManager:
    return MultiCameraManager([build_camera(c, seed=i) for i, c in enumerate(cfgs)])


def build_recorder(cfg: RecordConfig) -> HDF5Recorder:
    return HDF5Recorder(
        cfg.output_dir,
        fps=cfg.fps,
        image_compression=cfg.image_compression,
        compression_opts=cfg.compression_opts,
    )


def build_system(cfg: TeleopConfig) -> TeleopSystem:
    pipeline, leader_kin, follower_kin, _leader_spec, follower_spec = build_pipeline(
        cfg.leader, cfg.follower, cfg.retarget
    )
    home = np.asarray(
        cfg.follower.home if cfg.follower.home is not None else follower_spec.home,
        dtype=np.float64,
    )
    follower = build_follower_device(cfg.follower, follower_spec)
    gripper = build_gripper(cfg.follower.gripper, follower, cfg.follower.ip)
    return TeleopSystem(
        leader=build_leader_device(cfg.leader),
        follower=follower,
        gripper=gripper,
        leader_kin=leader_kin,
        follower_kin=follower_kin,
        retargeter=pipeline.retargeter,
        pipeline=pipeline,
        cameras=build_cameras(cfg.cameras),
        recorder=build_recorder(cfg.record),
        follower_home=home,
        config=cfg,
    )
