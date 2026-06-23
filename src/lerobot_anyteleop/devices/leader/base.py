"""Leader (teleoperation source) interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class LeaderState:
    """A single reading from the leader arm.

    ``joint_positions`` maps arm joint name -> angle in **radians**.
    ``gripper`` is normalized to ``[0, 1]`` (0 = closed, 1 = open).
    """

    joint_positions: dict[str, float]
    gripper: float


class LeaderInterface(abc.ABC):
    #: arm joint names (excludes the gripper), in a stable order.
    joint_names: list[str]

    @abc.abstractmethod
    def connect(self) -> None:
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        ...

    @abc.abstractmethod
    def get_state(self) -> LeaderState:
        """Read joint angles (radians) and gripper opening in a single round-trip."""

    # context-manager sugar -------------------------------------------------
    def __enter__(self) -> "LeaderInterface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
