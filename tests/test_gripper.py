"""Gripper normalized->hardware mapping (no SDK / hardware required)."""

from __future__ import annotations

from lerobot_anyteleop.devices.gripper.none import NoGripper
from lerobot_anyteleop.devices.gripper.robotiq import RobotiqGripper
from lerobot_anyteleop.devices.gripper.xarm import XArmGripper


class _FakeArm:
    def __init__(self):
        self.positions = []

    def set_gripper_enable(self, *a, **k): ...
    def set_gripper_mode(self, *a, **k): ...
    def set_gripper_speed(self, *a, **k): ...

    def set_gripper_position(self, pos, wait=False):
        self.positions.append(pos)


class _FakeFollower:
    def __init__(self):
        self.api = _FakeArm()


class _FakeRobotiq:
    def __init__(self):
        self.positions = []

    def move(self, pos, speed=255, force=255, wait=False):
        self.positions.append(pos)


def test_no_gripper_is_safe():
    g = NoGripper()
    g.connect()
    g.set_normalized(0.7)  # no error
    assert g.is_connected


def test_xarm_gripper_mapping():
    # 1 -> 850 (open), 0 -> 0 (closed), 0.5 -> 425
    f = _FakeFollower()
    g = XArmGripper(f)
    g.connect()
    for v in (1.0, 0.0, 0.5):
        g.set_normalized(v)
    assert f.api.positions == [850, 0, 425]


def test_robotiq_mapping_inverts():
    # Robotiq: 0 = open, 255 = closed -> raw = round((1-v)*255)
    g = RobotiqGripper(backend="serial")
    g._g = _FakeRobotiq()  # bypass connect()/SDK
    for v in (1.0, 0.0, 0.5):
        g.set_normalized(v)
    assert g._g.positions == [0, 255, 128]


def test_clamping():
    g = RobotiqGripper(backend="serial")
    g._g = _FakeRobotiq()
    g.set_normalized(2.0)   # -> clamp 1.0 -> open -> 0
    g.set_normalized(-1.0)  # -> clamp 0.0 -> closed -> 255
    assert g._g.positions == [0, 255]


def test_default_deadbands():
    assert XArmGripper(_FakeFollower()).deadband == 0.0
    assert RobotiqGripper().deadband == 0.03
