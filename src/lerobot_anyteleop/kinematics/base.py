"""Abstract kinematics interface shared by leader and follower robots."""

from __future__ import annotations

import abc

import numpy as np

from ..transforms import Pose


class KinematicsModel(abc.ABC):
    """FK/IK for a single serial manipulator about one end-effector link.

    Joint vectors are always ordered to match :attr:`actuated_names`.
    """

    @property
    @abc.abstractmethod
    def actuated_names(self) -> list[str]:
        """Names of the actuated joints, in the order ``fk``/``ik`` expect."""

    @property
    @abc.abstractmethod
    def num_actuated(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def lower_limits(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def upper_limits(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def ee_link_name(self) -> str:
        ...

    @abc.abstractmethod
    def fk(self, q: np.ndarray) -> Pose:
        """Forward kinematics: joint vector -> end-effector :class:`Pose`."""

    @abc.abstractmethod
    def ik(self, target: Pose, q_init: np.ndarray | None = None) -> np.ndarray:
        """Inverse kinematics: target EE :class:`Pose` -> joint vector.

        ``q_init`` (when given) biases the solution toward the current
        configuration for temporal continuity during teleoperation.
        """

    # -- convenience --------------------------------------------------------
    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower_limits, self.upper_limits)

    def rest_pose(self) -> np.ndarray:
        """Mid-range configuration (a sane default / IK seed)."""
        return 0.5 * (self.lower_limits + self.upper_limits)

    def order_from_dict(self, values: dict[str, float], default: float = 0.0) -> np.ndarray:
        """Build a joint vector (in :attr:`actuated_names` order) from a name->value map."""
        return np.array([float(values.get(n, default)) for n in self.actuated_names])
