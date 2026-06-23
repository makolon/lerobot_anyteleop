"""Controller + recorder integration, using in-test device fakes (no hardware).

Skipped unless the kinematics stack (jax + pyroki) is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyroki")
import h5py  # noqa: E402

from lerobot_anyteleop.config import TeleopConfig  # noqa: E402
from lerobot_anyteleop.devices import MultiCameraManager  # noqa: E402
from lerobot_anyteleop.devices.follower.base import FollowerInterface  # noqa: E402
from lerobot_anyteleop.devices.gripper.base import GripperInterface  # noqa: E402
from lerobot_anyteleop.devices.leader.base import LeaderInterface, LeaderState  # noqa: E402
from lerobot_anyteleop.factory import TeleopSystem, build_pipeline  # noqa: E402
from lerobot_anyteleop.recording import HDF5Recorder  # noqa: E402
from lerobot_anyteleop.robots import get_robot_spec  # noqa: E402
from lerobot_anyteleop.teleop import TeleopController  # noqa: E402


class FakeLeader(LeaderInterface):
    def __init__(self, arm_names):
        self.joint_names = list(arm_names)
        self._t = 0

    def connect(self): ...
    def disconnect(self): ...

    @property
    def is_connected(self):
        return True

    def get_state(self):
        # gentle scripted motion so the follower actually moves
        a = 0.3 * np.sin(self._t * 0.2)
        self._t += 1
        joints = {n: (a if i == 1 else 0.05 * a) for i, n in enumerate(self.joint_names)}
        return LeaderState(joint_positions=joints, gripper=0.5)


class FakeFollower(FollowerInterface):
    def __init__(self, arm_names, home):
        self.joint_names = list(arm_names)
        self._home = np.asarray(home, float)
        self._q = self._home.copy()

    def connect(self): ...
    def disconnect(self): ...

    @property
    def is_connected(self):
        return True

    def get_joint_positions(self):
        return self._q.copy()

    def move_to_joint_positions(self, q, blocking=True):
        self._q = np.asarray(q, float).copy()

    def enter_servo_mode(self): ...

    def send_joint_positions(self, q):
        self._q = np.asarray(q, float).copy()  # perfect tracking


class FakeGripper(GripperInterface):
    def __init__(self):
        self.commands = []

    def connect(self): ...
    def disconnect(self): ...

    @property
    def is_connected(self):
        return True

    def set_normalized(self, value):
        self.commands.append(float(value))


def _system(tmp_path) -> tuple[TeleopSystem, TeleopConfig]:
    cfg = TeleopConfig.from_dict(
        {
            "leader": {"robot": "so101"},
            "follower": {"robot": "xarm7"},
            "record": {"output_dir": str(tmp_path)},
            "loop": {"rate_hz": 0, "max_steps": 6},
        }
    )
    pipeline, leader_kin, follower_kin, leader_spec, follower_spec = build_pipeline(
        cfg.leader, cfg.follower, cfg.retarget
    )
    home = np.asarray(follower_spec.home, float)
    sys = TeleopSystem(
        leader=FakeLeader(get_robot_spec("so101").arm_joint_names),
        follower=FakeFollower(follower_spec.arm_joint_names, home),
        gripper=FakeGripper(),
        leader_kin=leader_kin,
        follower_kin=follower_kin,
        retargeter=pipeline.retargeter,
        pipeline=pipeline,
        cameras=MultiCameraManager([]),
        recorder=HDF5Recorder(tmp_path),
        follower_home=home,
        config=cfg,
    )
    return sys, cfg


def test_controller_runs_and_records(tmp_path):
    sys, cfg = _system(tmp_path)
    controller = TeleopController(sys, cfg)
    n = controller.run(record=True)
    assert n == 6

    path = sys.recorder.path
    assert path is not None and path.exists()
    with h5py.File(path, "r") as f:
        assert f["observation/follower_qpos"].shape == (6, 7)
        assert f["action/follower_qpos"].shape == (6, 7)
        assert f["observation/leader_qpos"].shape == (6, 6)  # 5 arm + gripper
        assert f["observation/follower_ee_pose"].shape == (6, 7)
        assert f.attrs["num_steps"] == 6
        # follower moved in response to leader motion
        assert np.std(f["action/follower_qpos"][:]) > 1e-4

    # leader gripper (0.5) was forwarded to the gripper device
    assert sys.gripper.commands and sys.gripper.commands[0] == 0.5
