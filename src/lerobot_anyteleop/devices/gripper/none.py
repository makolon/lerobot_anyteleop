"""No-op gripper (the default when nothing is attached)."""

from __future__ import annotations

from .base import GripperInterface


class NoGripper(GripperInterface):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool:
        return True

    def set_normalized(self, value: float) -> None: ...
