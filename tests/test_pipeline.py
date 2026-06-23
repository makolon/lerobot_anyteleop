"""Kinematic retargeting pipeline tests on the real URDFs.

Skipped unless the kinematics stack (jax + pyroki) is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyroki")

from lerobot_anyteleop.config import FollowerConfig, LeaderConfig, RetargetConfig  # noqa: E402
from lerobot_anyteleop.factory import build_pipeline  # noqa: E402
from lerobot_anyteleop.robots import get_robot_spec  # noqa: E402

FOLLOWERS = ["xarm7", "panda", "ur5e"]


@pytest.fixture(scope="module", params=FOLLOWERS)
def built(request):
    follower = request.param
    pipeline, leader_kin, follower_kin, leader_spec, follower_spec = build_pipeline(
        LeaderConfig(robot="so101"),
        FollowerConfig(robot=follower),
        RetargetConfig(position_scale=1.0, orientation_scale=1.0),
    )
    return follower, pipeline, follower_kin, follower_spec


def test_arm_dof_matches_spec(built):
    follower, pipeline, follower_kin, spec = built
    assert pipeline.jmap.num_arm == spec.dof
    assert pipeline.jmap.num_full == follower_kin.num_actuated


def test_fk_ik_roundtrip(built):
    follower, pipeline, follower_kin, spec = built
    q = np.asarray(spec.home, dtype=np.float64)
    q_full = pipeline.jmap.to_full(q, base=follower_kin.rest_pose())
    pose = follower_kin.fk(q_full)
    q_ik = follower_kin.ik(pose, q_init=q_full)
    err = np.linalg.norm(follower_kin.fk(q_ik).position - pose.position)
    assert err < 2e-3


def test_engage_then_no_leader_motion_keeps_follower(built):
    follower, pipeline, follower_kin, spec = built
    home = np.asarray(spec.home, dtype=np.float64)
    leader_zero = {n: 0.0 for n in get_robot_spec("so101").arm_joint_names}
    pipeline.engage(leader_zero, home)
    out = pipeline.step(leader_zero, home)  # leader unchanged
    # follower EE target should equal the follower's home EE pose
    home_pose = pipeline.follower_pose_from_arm(home)
    assert np.allclose(out.follower_target_pose.position, home_pose.position, atol=1e-3)


def test_leader_motion_moves_follower_target(built):
    follower, pipeline, follower_kin, spec = built
    home = np.asarray(spec.home, dtype=np.float64)
    leader0 = {n: 0.0 for n in get_robot_spec("so101").arm_joint_names}
    pipeline.engage(leader0, home)
    home_pose = pipeline.follower_pose_from_arm(home)
    moved = dict(leader0)
    moved["shoulder_lift"] = 0.4  # move a proximal leader joint
    out = pipeline.step(moved, home)
    shift = np.linalg.norm(out.follower_target_pose.position - home_pose.position)
    assert shift > 1e-3  # the follower target actually moved
