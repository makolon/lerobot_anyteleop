from __future__ import annotations

import numpy as np
import pytest

from lerobot_anyteleop.transforms import (
    Pose,
    matrix_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    quat_to_rotvec,
    rotvec_to_quat,
    rpy_to_quat,
    quat_to_rpy,
    scale_rotation,
)

rng = np.random.default_rng(0)


def random_quat():
    q = rng.standard_normal(4)
    return quat_normalize(q)


def test_quat_matrix_roundtrip():
    for _ in range(50):
        q = random_quat()
        R = quat_to_matrix(q)
        # R is a proper rotation.
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        q2 = matrix_to_quat(R)
        # q and -q are the same rotation; compare matrices instead.
        assert np.allclose(quat_to_matrix(q2), R, atol=1e-9)


def test_quat_multiply_matches_matrix():
    for _ in range(50):
        q1, q2 = random_quat(), random_quat()
        R = quat_to_matrix(quat_multiply(q1, q2))
        assert np.allclose(R, quat_to_matrix(q1) @ quat_to_matrix(q2), atol=1e-9)


def test_rotvec_roundtrip():
    for _ in range(50):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi * 0.99)  # < pi so the log is unique
        v = axis * angle
        q = rotvec_to_quat(v)
        v2 = quat_to_rotvec(q)
        assert np.allclose(v, v2, atol=1e-7)


def test_scale_rotation_endpoints_and_half():
    q = rotvec_to_quat(np.array([0.0, 0.0, 1.2]))
    assert np.allclose(scale_rotation(q, 0.0), [1, 0, 0, 0], atol=1e-9)
    assert np.allclose(quat_to_matrix(scale_rotation(q, 1.0)), quat_to_matrix(q), atol=1e-9)
    half = scale_rotation(q, 0.5)
    assert np.allclose(quat_to_rotvec(half), [0, 0, 0.6], atol=1e-7)


def test_rpy_roundtrip():
    for _ in range(50):
        rpy = rng.uniform(-1.0, 1.0, size=3)
        q = rpy_to_quat(*rpy)
        rpy2 = quat_to_rpy(q)
        # compare via rotation matrices to dodge wraparound.
        assert np.allclose(quat_to_matrix(rpy_to_quat(*rpy2)), quat_to_matrix(q), atol=1e-9)


def test_pose_inverse_and_compose():
    for _ in range(50):
        p = Pose(position=rng.standard_normal(3), wxyz=random_quat())
        ident = p @ p.inverse()
        assert np.allclose(ident.position, 0.0, atol=1e-9)
        assert np.allclose(quat_to_matrix(ident.wxyz), np.eye(3), atol=1e-9)

        a = Pose(position=rng.standard_normal(3), wxyz=random_quat())
        b = Pose(position=rng.standard_normal(3), wxyz=random_quat())
        assert np.allclose((a @ b).matrix(), a.matrix() @ b.matrix(), atol=1e-9)


def test_pose_pos_quat_roundtrip():
    p = Pose(position=[1, 2, 3], wxyz=random_quat())
    p2 = Pose.from_position_wxyz(p.as_pos_quat())
    assert np.allclose(p.position, p2.position)
    assert np.allclose(p.matrix(), p2.matrix(), atol=1e-12)


def test_pose_from_matrix():
    p = Pose(position=rng.standard_normal(3), wxyz=random_quat())
    assert np.allclose(Pose.from_matrix(p.matrix()).matrix(), p.matrix(), atol=1e-9)
