"""Name-based joint vector mapping.

Two coordinate systems exist for follower joints:

* **full** — every actuated joint the kinematics model (pyroki) exposes, in its
  order. May include extra joints (e.g. a Panda finger) and may be reordered
  relative to the URDF declaration.
* **arm** — the joints the hardware driver actually commands, in hardware order
  (``RobotSpec.arm_joint_names``).

:class:`JointMap` converts between them by joint name.
"""

from __future__ import annotations

import numpy as np


def reorder(values: dict[str, float], target_names, default: float = 0.0) -> np.ndarray:
    """Build a vector ordered by ``target_names`` from a name->value mapping."""
    return np.array([float(values.get(n, default)) for n in target_names], dtype=np.float64)


class JointMap:
    def __init__(self, full_names, arm_names) -> None:
        self.full_names = list(full_names)
        self.arm_names = list(arm_names)
        missing = [n for n in self.arm_names if n not in self.full_names]
        if missing:
            raise ValueError(
                f"Arm joints {missing} not found in kinematics actuated joints {self.full_names}."
            )
        self._arm_idx = np.array([self.full_names.index(n) for n in self.arm_names])

    @property
    def num_full(self) -> int:
        return len(self.full_names)

    @property
    def num_arm(self) -> int:
        return len(self.arm_names)

    def to_arm(self, q_full: np.ndarray) -> np.ndarray:
        """Select the arm joints (hardware order) from a full joint vector."""
        return np.asarray(q_full, dtype=np.float64)[self._arm_idx]

    def to_full(self, q_arm: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Scatter arm joints into a full vector; non-arm joints come from ``base``."""
        out = np.asarray(base, dtype=np.float64).copy()
        q_arm = np.asarray(q_arm, dtype=np.float64)
        out[self._arm_idx] = q_arm
        return out

    def full_dict(self, q_full: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.full_names, np.asarray(q_full))}
