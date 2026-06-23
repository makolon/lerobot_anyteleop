"""Follower (controlled robot) interface."""

from __future__ import annotations

import abc

import numpy as np


class FollowerInterface(abc.ABC):
    #: actuated joint names, ordered to match the follower kinematics model.
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
    def get_joint_positions(self) -> np.ndarray:
        """Measured joint angles in **radians**, ordered like :attr:`joint_names`."""

    @abc.abstractmethod
    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        """Planned move to ``q`` (used for homing before servoing). Radians."""

    @abc.abstractmethod
    def enter_servo_mode(self) -> None:
        """Switch to high-rate streaming control mode for teleoperation."""

    @abc.abstractmethod
    def send_joint_positions(self, q: np.ndarray) -> None:
        """Stream a servo joint target (must be in servo mode). Radians."""

    # Gripper control is handled by a separate, pluggable GripperInterface
    # (see ``devices.gripper``), since grippers are interchangeable attachments.

    # context-manager sugar -------------------------------------------------
    def __enter__(self) -> "FollowerInterface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
