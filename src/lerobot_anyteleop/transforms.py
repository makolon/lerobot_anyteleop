"""Minimal, dependency-light SE(3) / SO(3) math (NumPy only).

Quaternions are stored **scalar-first** (``wxyz``) to match the convention used by
``jaxlie`` / ``pyroki`` forward kinematics, so poses round-trip through the
kinematics layer without re-ordering.

Keeping this module free of JAX means the retargeting logic and unit tests run
without the (heavy) kinematics stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Quaternion helpers (wxyz, scalar-first)
# --------------------------------------------------------------------------- #
def quat_normalize(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    # Canonicalize sign so that the real part is non-negative (q and -q are equal
    # rotations); this keeps logs/deltas continuous.
    if q[0] < 0:
        q = -q
    return q


def quat_multiply(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    """Hamilton product ``q1 * q2`` (both wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: ArrayLike) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_inverse(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return quat_conjugate(q) / float(np.dot(q, q))


def quat_rotate(q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Rotate 3-vector ``v`` by unit quaternion ``q``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = np.array([0.0, *v])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_to_matrix(q: ArrayLike) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: ArrayLike) -> np.ndarray:
    """Rotation matrix -> wxyz quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


# --------------------------------------------------------------------------- #
# SO(3) exp / log  (axis-angle "rotation vector" <-> quaternion)
# --------------------------------------------------------------------------- #
def rotvec_to_quat(rotvec: ArrayLike) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec)
    if theta < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_to_rotvec(q: ArrayLike) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    v = np.array([x, y, z])
    sin_half = np.linalg.norm(v)
    if sin_half < _EPS:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(sin_half, w)
    return (v / sin_half) * theta


def scale_rotation(q: ArrayLike, scale: float) -> np.ndarray:
    """Scale a rotation by ``scale`` along its geodesic (axis-angle scaling).

    ``scale_rotation(q, 0) == identity`` and ``scale_rotation(q, 1) == q``.
    """
    return rotvec_to_quat(scale * quat_to_rotvec(q))


# --------------------------------------------------------------------------- #
# RPY (xArm-style extrinsic XYZ:  R = Rz(yaw) @ Ry(pitch) @ Rx(roll))
# --------------------------------------------------------------------------- #
def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_rpy(q: ArrayLike) -> np.ndarray:
    """Inverse of :func:`rpy_to_quat`; returns ``[roll, pitch, yaw]`` (radians)."""
    w, x, y, z = quat_normalize(q)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# --------------------------------------------------------------------------- #
# Pose: rigid transform represented as (position, quaternion-wxyz)
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """An SE(3) transform ``T`` mapping the local frame into a reference frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.wxyz = quat_normalize(np.asarray(self.wxyz, dtype=np.float64).reshape(4))

    # -- constructors -------------------------------------------------------
    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    @classmethod
    def from_matrix(cls, T: ArrayLike) -> "Pose":
        T = np.asarray(T, dtype=np.float64)
        return cls(position=T[:3, 3], wxyz=matrix_to_quat(T[:3, :3]))

    @classmethod
    def from_position_wxyz(cls, arr: ArrayLike) -> "Pose":
        """From a 7-vector ``[x, y, z, qw, qx, qy, qz]``."""
        arr = np.asarray(arr, dtype=np.float64).reshape(7)
        return cls(position=arr[:3], wxyz=arr[3:])

    # -- conversions --------------------------------------------------------
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(self.wxyz)
        T[:3, 3] = self.position
        return T

    def as_pos_quat(self) -> np.ndarray:
        """Return a 7-vector ``[x, y, z, qw, qx, qy, qz]`` (storage/logging format)."""
        return np.concatenate([self.position, self.wxyz])

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.wxyz)

    # -- group operations ---------------------------------------------------
    def inverse(self) -> "Pose":
        q_inv = quat_conjugate(self.wxyz)
        return Pose(position=-quat_rotate(q_inv, self.position), wxyz=q_inv)

    def multiply(self, other: "Pose") -> "Pose":
        """Compose: ``self @ other`` (apply ``other`` then ``self``)."""
        return Pose(
            position=self.position + quat_rotate(self.wxyz, other.position),
            wxyz=quat_multiply(self.wxyz, other.wxyz),
        )

    def __matmul__(self, other: "Pose") -> "Pose":
        return self.multiply(other)

    def copy(self) -> "Pose":
        return Pose(position=self.position.copy(), wxyz=self.wxyz.copy())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = np.array2string(self.position, precision=4, suppress_small=True)
        q = np.array2string(self.wxyz, precision=4, suppress_small=True)
        return f"Pose(position={p}, wxyz={q})"
