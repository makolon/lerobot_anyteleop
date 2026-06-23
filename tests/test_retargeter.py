from __future__ import annotations

import numpy as np
import pytest

from lerobot_anyteleop.retargeting import PoseRetargeter
from lerobot_anyteleop.transforms import (
    Pose,
    quat_to_matrix,
    quat_to_rotvec,
    rotvec_to_quat,
)


def make_pose(pos, rotvec=(0, 0, 0)):
    return Pose(position=np.asarray(pos, float), wxyz=rotvec_to_quat(np.asarray(rotvec, float)))


def test_requires_engage():
    r = PoseRetargeter()
    with pytest.raises(RuntimeError):
        r.compute_target(make_pose([0, 0, 0]))


def test_no_motion_returns_follower_anchor():
    r = PoseRetargeter(position_scale=2.0, orientation_scale=0.5)
    leader0 = make_pose([0.1, 0.2, 0.3], [0.2, 0, 0])
    follower0 = make_pose([0.5, 0.0, 0.4], [0, 0.1, 0])
    r.engage(leader0, follower0)
    out = r.compute_target(leader0)  # leader unchanged
    assert np.allclose(out.position, follower0.position, atol=1e-9)
    assert np.allclose(quat_to_matrix(out.wxyz), quat_to_matrix(follower0.wxyz), atol=1e-9)


def test_position_scaling():
    r = PoseRetargeter(position_scale=2.0)
    leader0 = make_pose([0, 0, 0])
    follower0 = make_pose([1, 1, 1])
    r.engage(leader0, follower0)
    leader = make_pose([0.1, -0.2, 0.05])
    out = r.compute_target(leader)
    expected = follower0.position + 2.0 * np.array([0.1, -0.2, 0.05])
    assert np.allclose(out.position, expected, atol=1e-9)


def test_orientation_scaling():
    r = PoseRetargeter(orientation_scale=0.5)
    leader0 = make_pose([0, 0, 0])
    follower0 = make_pose([0, 0, 0])  # identity orientation anchor
    r.engage(leader0, follower0)
    # leader rotates 1.0 rad about z
    leader = make_pose([0, 0, 0], [0, 0, 1.0])
    out = r.compute_target(leader)
    # follower should rotate 0.5 rad about z relative to (identity) anchor
    assert np.allclose(quat_to_rotvec(out.wxyz), [0, 0, 0.5], atol=1e-7)


def test_align_rotation_maps_axes():
    # align = +90 deg about z maps leader +x delta to follower +y.
    align = rotvec_to_quat(np.array([0, 0, np.pi / 2]))
    r = PoseRetargeter(position_scale=1.0, align_wxyz=align)
    leader0 = make_pose([0, 0, 0])
    follower0 = make_pose([0, 0, 0])
    r.engage(leader0, follower0)
    out = r.compute_target(make_pose([0.3, 0, 0]))
    assert np.allclose(out.position, [0, 0.3, 0], atol=1e-7)


def test_clutch_reengage_recenters():
    r = PoseRetargeter(position_scale=1.0)
    r.engage(make_pose([0, 0, 0]), make_pose([1, 0, 0]))
    # move leader, then re-engage at the new leader pose without moving follower target
    r.compute_target(make_pose([0.5, 0, 0]))
    r.engage(make_pose([0.5, 0, 0]), make_pose([1.0, 0, 0]))
    out = r.compute_target(make_pose([0.6, 0, 0]))
    assert np.allclose(out.position, [1.1, 0, 0], atol=1e-9)
