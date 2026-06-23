"""Resolve the follower's gripper *visualization* for viser.

Gripper URDFs are **vendored** under ``assets/urdf/grippers/`` (editable) rather
than pulled at runtime, so mount frames / geometry can be adjusted. Two strategies:

* **combined** — the follower visual URDF already contains the gripper (xArm
  rendered with ``add_gripper``; Franka Hand in ``panda_description``); its finger
  joint is set alongside the arm joints.
* **mounted** — a separate (vendored) gripper URDF is rendered as a child of the
  flange frame and animated; the mount frame tracks the follower EE pose.

Two tunables make this correct across arms:

* :data:`GRIPPER_MOUNTS` — per-arm flange->gripper transform. UR's ``tool0`` needs
  no correction, but xArm ``link_eef`` and the Panda flange are rotated 90 deg
  about yaw relative to it, so mounted grippers get a yaw correction there.
* :data:`STRIP_ON_MOUNT` — arm links to hide when a separate gripper is mounted
  (e.g. drop the built-in Franka Hand so a Robotiq doesn't double up).

Both are plain editable tables; ``--gripper-mount`` overrides the mount per run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..kinematics.urdf import load_urdf
from ..robots import RobotSpec
from ..transforms import Pose, rpy_to_quat


# repo_root/assets (editable install: src/lerobot_anyteleop/viz/this_file)
_ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _asset(*parts: str) -> str:
    p = _ASSETS.joinpath(*parts)
    return str(p) if p.exists() else str(Path("assets").joinpath(*parts))


#: vendored standalone gripper URDFs (editable)
VENDORED_GRIPPERS = {
    "robotiq_2f85": _asset("urdf", "grippers", "robotiq_2f85", "robotiq_2f85.urdf"),
}

#: per-arm flange -> gripper mount for MOUNTED grippers ((xyz), (rpy)). Editable.
#: UR tool0 needs no correction; xArm link_eef and the Panda flange are rotated
#: 90 deg about yaw vs it (flip the sign if your gripper points the other way).
GRIPPER_MOUNTS = {
    "ur5e": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "xarm7": ((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2)),
    "panda": ((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2)),
}

#: arm links to hide when a separate gripper is mounted (drop the built-in hand).
STRIP_ON_MOUNT = {
    "panda": ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_hand_tcp"),
}


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
# Helpers
# --------------------------------------------------------------------------- #
def _strip_links(urdf, names):
    """Return a copy of ``urdf`` with the named links (and their joints) removed."""
    import yourdfpy

    names = set(names)
    robot = urdf.robot
    robot.links = [l for l in robot.links if l.name not in names]
    robot.joints = [j for j in robot.joints if j.parent not in names and j.child not in names]
    return yourdfpy.URDF(robot=robot, load_meshes=True, filename_handler=urdf._filename_handler)


def _load_xarm_with_gripper():
    """Render the xArm device URDF with the native gripper (arm + gripper combined)."""
    import os
    import tempfile

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
    mount_xyz=None,
    mount_rpy=None,
) -> FollowerVisual:
    """Build the follower visual (arm [+ gripper]) for the chosen gripper model.

    ``model``: ``none`` | ``xarm`` | ``robotiq_2f85`` | ``franka`` | a URDF path
    (mounted). ``None`` -> sensible default per robot.

    Mount defaults come from :data:`GRIPPER_MOUNTS` per arm; pass ``mount_xyz`` /
    ``mount_rpy`` to override.
    """
    model = model or _default_model(spec.name)

    d_xyz, d_rpy = GRIPPER_MOUNTS.get(spec.name, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    mount_xyz = d_xyz if mount_xyz is None else mount_xyz
    mount_rpy = d_rpy if mount_rpy is None else mount_rpy
    mount = Pose(position=list(mount_xyz), wxyz=rpy_to_quat(*mount_rpy))

    if model == "none":
        return FollowerVisual(load_urdf(spec.urdf, load_meshes=True), None, {}, combined=True)

    if model == "xarm":
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
        gripper = load_urdf(VENDORED_GRIPPERS["robotiq_2f85"], load_meshes=True)
        fingers = {"finger_joint": (0.0, 0.8)}
    else:
        gripper = load_urdf(model, load_meshes=True)  # arbitrary URDF / description name
        fingers = {n: (0.0, 0.8) for n in list(gripper.actuated_joint_names)[:1]}

    arm = load_urdf(spec.urdf, load_meshes=True)
    strip = STRIP_ON_MOUNT.get(spec.name)
    if strip:
        arm = _strip_links(arm, strip)  # e.g. drop the Franka Hand so it doesn't double up

    return FollowerVisual(arm, gripper, fingers, combined=False, mount_offset=mount)
