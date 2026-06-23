"""Hardware-independent kinematic retargeting core.

This is the math that turns a leader joint reading into a follower joint command:
``leader FK -> retarget(scale) -> follower IK``. It has no device or I/O
dependencies, so it is shared by the real-hardware :class:`TeleopController` and
the :mod:`viser` visualization, and is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..joint_utils import JointMap
from ..kinematics.base import KinematicsModel
from ..retargeting import PoseRetargeter
from ..transforms import Pose


@dataclass
class RetargetOutput:
    leader_pose: Pose
    follower_target_pose: Pose
    follower_q_full: np.ndarray  # kinematics (full actuated) order
    follower_q_arm: np.ndarray   # hardware (arm) order


class KinematicRetargetPipeline:
    def __init__(
        self,
        leader_kin: KinematicsModel,
        follower_kin: KinematicsModel,
        retargeter: PoseRetargeter,
        follower_arm_names,
    ) -> None:
        self.leader_kin = leader_kin
        self.follower_kin = follower_kin
        self.retargeter = retargeter
        self.jmap = JointMap(follower_kin.actuated_names, follower_arm_names)
        self._rest_full = follower_kin.rest_pose()

    # -- poses --------------------------------------------------------------
    def leader_pose(self, leader_joint_positions: dict[str, float]) -> Pose:
        q = self.leader_kin.order_from_dict(leader_joint_positions)
        return self.leader_kin.fk(q)

    def follower_pose_from_arm(self, q_arm: np.ndarray) -> Pose:
        return self.follower_kin.fk(self.jmap.to_full(q_arm, base=self._rest_full))

    # -- engage / step ------------------------------------------------------
    def engage(self, leader_joint_positions: dict[str, float], follower_q_arm: np.ndarray) -> None:
        self.retargeter.engage(
            self.leader_pose(leader_joint_positions),
            self.follower_pose_from_arm(follower_q_arm),
        )

    def step(self, leader_joint_positions: dict[str, float], q_init_arm: np.ndarray) -> RetargetOutput:
        leader_pose = self.leader_pose(leader_joint_positions)
        target = self.retargeter.compute_target(leader_pose)
        q_init_full = self.jmap.to_full(q_init_arm, base=self._rest_full)
        q_full = self.follower_kin.ik(target, q_init=q_init_full)
        q_full = self.follower_kin.clip_to_limits(q_full)
        q_arm = self.jmap.to_arm(q_full)
        return RetargetOutput(
            leader_pose=leader_pose,
            follower_target_pose=target,
            follower_q_full=q_full,
            follower_q_arm=q_arm,
        )
