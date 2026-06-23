"""Gripper interface."""

from __future__ import annotations

import abc


class GripperInterface(abc.ABC):
    """Drive a gripper from a normalized command.

    ``set_normalized(value)``: ``value`` in ``[0, 1]``, **1 = fully open,
    0 = fully closed**. Each driver maps this to its hardware units.

    ``deadband``: minimum change in the normalized command before re-issuing a
    command — lets slow grippers (Franka, Robotiq) ignore tiny jitter while the
    control loop runs fast.
    """

    deadband: float = 0.0

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
    def set_normalized(self, value: float) -> None:
        ...

    @staticmethod
    def _clamp01(value: float) -> float:
        return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)
