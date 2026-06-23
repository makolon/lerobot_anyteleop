"""Registry of supported robots.

A :class:`RobotSpec` is the single source of truth for a robot's kinematics: the
URDF source (a ``robot_descriptions`` module name or a path), the end-effector
link to target, the **arm** joints the hardware actually commands (in hardware
order), and a safe home configuration.

The kinematics layer (pyroki) may expose *more* actuated joints than the arm has
(e.g. ``panda_description`` includes a finger joint, and pyroki may reorder
joints). ``arm_joint_names`` + name-based mapping (see ``joint_utils``) bridge the
full FK/IK joint vector and the hardware joint vector.

Adding a robot = add one entry here + (for real control) a follower driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SO101_URDF = "assets/urdf/so101/so101_new_calib.urdf"


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf: str                       # robot_descriptions name or path to .urdf
    ee_link: str
    arm_joint_names: tuple[str, ...]
    home: tuple[float, ...]         # arm-DOF home configuration (radians)
    base_link: str = ""
    follower_backend: str = ""      # default real driver backend for this robot

    @property
    def dof(self) -> int:
        return len(self.arm_joint_names)


ROBOTS: dict[str, RobotSpec] = {
    # ---- leader -----------------------------------------------------------
    "so101": RobotSpec(
        name="so101",
        urdf=SO101_URDF,
        ee_link="gripper_frame_link",
        arm_joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        home=(0.0, 0.0, 0.0, 0.0, 0.0),
        base_link="base_link",
        follower_backend="",
    ),
    # ---- followers --------------------------------------------------------
    "xarm7": RobotSpec(
        name="xarm7",
        urdf="xarm7_description",
        ee_link="link_eef",
        arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        home=(0.0, -0.35, 0.0, 0.6, 0.0, 0.95, 0.0),
        base_link="link_base",
        follower_backend="xarm7",
    ),
    "panda": RobotSpec(
        name="panda",
        urdf="panda_description",
        ee_link="panda_hand",
        arm_joint_names=(
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ),
        home=(0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
        base_link="panda_link0",
        follower_backend="franka",
    ),
    "ur5e": RobotSpec(
        name="ur5e",
        urdf="ur5e_description",
        ee_link="tool0",
        arm_joint_names=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        home=(0.0, -1.57, 1.57, -1.57, -1.57, 0.0),
        base_link="base_link",
        follower_backend="ur",
    ),
}


def get_robot_spec(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise KeyError(f"Unknown robot {name!r}. Known: {sorted(ROBOTS)}")
    return ROBOTS[name]
