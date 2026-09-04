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
    def send_joint_positions(self, q: np.ndarray) -> np.ndarray | None:
        """Stream a servo joint target (must be in servo mode). Radians.

        A follower that modifies the requested target (for example, to apply a
        safety speed limit) returns the target it actually sent. ``None`` means
        that ``q`` was accepted unchanged. This keeps existing follower
        implementations backward compatible while letting recordings contain
        the real commanded joint action.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop streaming and hold/stop motion. Must be safe to call repeatedly.

        This is a controlled software stop, not an emergency stop. Physical
        safeguarding and the robot's emergency-stop circuit remain mandatory.
        """

    # Gripper control is handled by a separate, pluggable GripperInterface
    # (see ``devices.gripper``), since grippers are interchangeable attachments.

    # context-manager sugar -------------------------------------------------
    def __enter__(self) -> "FollowerInterface":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        stop_error: BaseException | None = None
        try:
            self.stop()
        except BaseException as error:
            stop_error = error

        disconnect_error: BaseException | None = None
        try:
            self.disconnect()
        except BaseException as error:
            disconnect_error = error

        # Cleanup must not replace an exception raised by the controlled body.
        if exc_value is not None:
            if stop_error is not None:
                exc_value.add_note(f"follower stop also failed: {stop_error}")
            if disconnect_error is not None:
                exc_value.add_note(f"follower disconnect also failed: {disconnect_error}")
            return None
        if stop_error is not None:
            if disconnect_error is not None:
                stop_error.add_note(f"follower disconnect also failed: {disconnect_error}")
            raise stop_error
        if disconnect_error is not None:
            raise disconnect_error
