"""Resolve the follower's gripper *visualization* for viser.

Some arm URDFs already include a gripper (Franka Hand in ``panda_description``,
the xArm gripper when rendered with ``add_gripper:=true``); others (UR5e, bare
xArm) have none, so a standalone gripper URDF is mounted at the flange.

Either way the leader's normalized gripper value (1 = open) animates the gripper's
actuated joint(s) via a per-model ``(open_value, closed_value)`` mapping.

Two strategies:
* **combined** — the follower visual URDF already contains the gripper; its finger
  joint is set alongside the arm joints (no separate mount).
* **mounted** — a separate gripper URDF is rendered as a child of the flange frame
  and its finger joint is animated; the mount frame tracks the follower EE pose.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from ..kinematics.urdf import load_urdf
from ..robots import RobotSpec
from ..transforms import Pose, rpy_to_quat


@dataclass
class FollowerVisual:
    arm_urdf: object                          # yourdfpy.URDF rendered as the follower
    gripper_urdf: object | None               # separate mounted gripper, or None
    finger_joints: dict                        # name -> (open_value, closed_value)
    combined: bool                             # True: finger joints live in arm_urdf
    mount_offset: Pose = field(default_factory=Pose)  # flange -> gripper base (mounted)


def finger_targets(finger_joints: dict, gripper_value: float) -> dict:
    """Map normalized gripper value (1=open) to each finger joint's angle."""
    g = 0.0 if gripper_value < 0 else 1.0 if gripper_value > 1 else float(gripper_value)
    return {j: o + (1.0 - g) * (c - o) for j, (o, c) in finger_joints.items()}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_robotiq_2f85():
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    return load_robot_description("robotiq_2f85_description", load_meshes=True)


def _load_xarm_with_gripper():
    """Render the xArm device URDF with the native gripper (arm + gripper combined)."""
    import xacrodoc
    import yourdfpy
    from robot_descriptions import xarm7_description

    pkg = xarm7_description.PACKAGE_PATH
    xacrodoc.packages.update_package_cache({"xarm_description": pkg})
    doc = xacrodoc.XacroDoc.from_file(
        os.path.join(pkg, "urdf/xarm_device.urdf.xacro"),
        subargs={"dof": "7", "robot_type": "xarm", "add_gripper": "true"},
    )
    urdf_path = os.path.join(tempfile.mkdtemp(), "xarm7_gripper.urdf")
    with open(urdf_path, "w") as f:
        f.write(doc.to_urdf_string())

    def handler(fname: str) -> str:
        if fname.startswith("file://"):
            return fname[len("file://"):]
        if fname.startswith("package://"):
            return os.path.join(pkg, fname[len("package://"):].split("/", 1)[1])
        return fname

    return yourdfpy.URDF.load(urdf_path, filename_handler=handler, load_meshes=True)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _default_model(robot: str) -> str:
    return {"xarm7": "xarm", "ur5e": "robotiq_2f85", "panda": "franka"}.get(robot, "none")


def resolve_follower_visual(
    spec: RobotSpec,
    model: str | None = None,
    *,
    mount_xyz=(0.0, 0.0, 0.0),
    mount_rpy=(0.0, 0.0, 0.0),
) -> FollowerVisual:
    """Build the follower visual (arm [+ gripper]) for the chosen gripper model.

    ``model``: ``none`` | ``xarm`` | ``robotiq_2f85`` | ``franka`` | a URDF path /
    ``robot_descriptions`` name (mounted). ``None`` -> sensible default per robot.
    """
    model = model or _default_model(spec.name)
    mount = Pose(position=list(mount_xyz), wxyz=rpy_to_quat(*mount_rpy))

    if model == "none":
        return FollowerVisual(load_urdf(spec.urdf, load_meshes=True), None, {}, combined=True)

    if model == "xarm":
        # native xArm gripper -> combined arm+gripper URDF (drive_joint open=0, closed=0.85)
        return FollowerVisual(
            _load_xarm_with_gripper(), None, {"drive_joint": (0.0, 0.85)}, combined=True
        )

    if model == "franka":
        # Franka Hand is already in panda_description; animate its finger (width m).
        return FollowerVisual(
            load_urdf(spec.urdf, load_meshes=True),
            None,
            {"panda_finger_joint1": (0.04, 0.0)},
            combined=True,
        )

    # --- mounted grippers -------------------------------------------------
    if model == "robotiq_2f85":
        gripper = _load_robotiq_2f85()
        fingers = {"finger_joint": (0.0, 0.8)}
    else:
        # treat `model` as a URDF path or robot_descriptions name; finger joints
        # are taken from its actuated joints with a default 0..0.8 mapping.
        gripper = load_urdf(model, load_meshes=True)
        fingers = {n: (0.0, 0.8) for n in list(gripper.actuated_joint_names)[:1]}

    return FollowerVisual(
        load_urdf(spec.urdf, load_meshes=True), gripper, fingers, combined=False, mount_offset=mount
    )
