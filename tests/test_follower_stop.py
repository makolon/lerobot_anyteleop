"""Hardware-free regression tests for follower stop semantics."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_anyteleop.devices.follower.base import FollowerInterface
from lerobot_anyteleop.devices.follower.ur import URFollower
from lerobot_anyteleop.devices.follower.xarm7 import XArm7Follower


class _XArm:
    def __init__(self) -> None:
        self.stop_result = 7
        self.stop_calls = 0

    def set_state(self, state: int):
        assert state == 4
        self.stop_calls += 1
        return self.stop_result


class _URControl:
    def __init__(self) -> None:
        self.stop_result = False
        self.stop_calls = 0

    def servoStop(self):
        self.stop_calls += 1
        return self.stop_result


def test_xarm_stop_failure_is_not_latched_as_stopped() -> None:
    follower = XArm7Follower("192.0.2.1", [f"joint{i}" for i in range(7)])
    arm = _XArm()
    follower._arm = arm
    follower._stopped = False

    with pytest.raises(RuntimeError, match="non-zero code 7"):
        follower.stop()
    assert not follower._stopped

    arm.stop_result = 0
    follower.stop()
    follower.stop()
    assert follower._stopped
    assert arm.stop_calls == 2


def test_ur_stop_false_is_not_latched_as_stopped() -> None:
    follower = URFollower("192.0.2.2", [f"joint{i}" for i in range(6)])
    control = _URControl()
    follower._rtde_c = control
    follower._stopped = False

    with pytest.raises(RuntimeError, match="servoStop failed"):
        follower.stop()
    assert not follower._stopped

    control.stop_result = True
    follower.stop()
    follower.stop()
    assert follower._stopped
    assert control.stop_calls == 2


class _ContextFollower(FollowerInterface):
    joint_names = ["joint"]

    def __init__(self) -> None:
        self.events: list[str] = []

    def connect(self) -> None:
        self.events.append("connect")

    def disconnect(self) -> None:
        self.events.append("disconnect")
        raise OSError("disconnect failed")

    @property
    def is_connected(self) -> bool:
        return True

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(1)

    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        pass

    def enter_servo_mode(self) -> None:
        pass

    def send_joint_positions(self, q: np.ndarray) -> None:
        pass

    def stop(self) -> None:
        self.events.append("stop")
        raise RuntimeError("stop failed")


def test_follower_context_preserves_body_error_and_runs_both_cleanups() -> None:
    follower = _ContextFollower()
    with pytest.raises(ValueError, match="body failed") as caught:
        with follower:
            raise ValueError("body failed")

    assert follower.events == ["connect", "stop", "disconnect"]
    notes = getattr(caught.value, "__notes__", [])
    assert any("stop failed" in note for note in notes)
    assert any("disconnect failed" in note for note in notes)
