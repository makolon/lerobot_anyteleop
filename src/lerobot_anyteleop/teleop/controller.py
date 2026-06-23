"""The teleoperation control loop (real hardware).

Per tick::

    leader.get_state()  ->  KinematicRetargetPipeline.step (FK -> retarget -> IK)
       ->  stream joint servo  ->  read state  ->  read cameras
       ->  (optionally) record one HDF5 step
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import TeleopConfig
from ..transforms import Pose

if TYPE_CHECKING:
    from ..factory import TeleopSystem


class Rate:
    """Fixed-rate loop limiter using ``time.perf_counter``."""

    def __init__(self, hz: float) -> None:
        self.period = 1.0 / float(hz) if hz > 0 else 0.0
        self._next = time.perf_counter()

    def reset(self) -> None:
        self._next = time.perf_counter()

    def sleep(self) -> None:
        if self.period <= 0:
            return
        self._next += self.period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.perf_counter()  # fell behind; resync


@dataclass
class StepResult:
    leader_pose: Pose
    follower_target_pose: Pose
    follower_measured_pose: Pose
    follower_q_cmd: np.ndarray
    follower_q_meas: np.ndarray
    gripper: float


class TeleopController:
    def __init__(self, system: TeleopSystem, config: TeleopConfig | None = None) -> None:
        self.sys = system
        self.cfg = config or system.config
        self._last_q_arm: np.ndarray | None = None
        self._last_grip: float | None = None
        self._t0: float = 0.0

    # -- lifecycle ----------------------------------------------------------
    def setup(self) -> None:
        s = self.sys
        s.leader.connect()
        s.follower.connect()
        s.gripper.connect()  # after the follower (xArm gripper shares its connection)
        s.cameras.start()

        s.follower.move_to_joint_positions(s.follower_home, blocking=True)
        self._last_q_arm = s.follower.get_joint_positions()

        state = s.leader.get_state()
        s.pipeline.engage(state.joint_positions, self._last_q_arm)

        s.follower.enter_servo_mode()
        self._t0 = time.perf_counter()

    def shutdown(self) -> None:
        s = self.sys
        s.cameras.stop()
        s.gripper.disconnect()
        s.follower.disconnect()
        s.leader.disconnect()

    # -- one control tick ---------------------------------------------------
    def step(self, record: bool = False) -> StepResult:
        s = self.sys
        state = s.leader.get_state()
        out = s.pipeline.step(state.joint_positions, self._last_q_arm)

        s.follower.send_joint_positions(out.follower_q_arm)

        # Map the leader gripper (normalized [0,1]) to the attached gripper.
        # Deadband avoids spamming slow grippers (Franka/Robotiq) at loop rate.
        g = state.gripper
        if self._last_grip is None or abs(g - self._last_grip) >= s.gripper.deadband:
            s.gripper.set_normalized(g)
            self._last_grip = g

        q_meas = s.follower.get_joint_positions()
        self._last_q_arm = q_meas
        measured_pose = s.pipeline.follower_pose_from_arm(q_meas)

        frames = s.cameras.read()
        if record:
            self._record_step(state, out, measured_pose, q_meas, frames)

        return StepResult(
            leader_pose=out.leader_pose,
            follower_target_pose=out.follower_target_pose,
            follower_measured_pose=measured_pose,
            follower_q_cmd=out.follower_q_arm,
            follower_q_meas=q_meas,
            gripper=state.gripper,
        )

    def _record_step(self, state, out, measured_pose, q_meas, frames) -> None:
        s = self.sys
        leader_qpos = np.array(
            [state.joint_positions[n] for n in s.leader.joint_names] + [state.gripper]
        )
        data: dict[str, object] = {
            "observation/leader_qpos": leader_qpos,
            "observation/leader_ee_pose": out.leader_pose.as_pos_quat(),
            "observation/follower_qpos": q_meas,
            "observation/follower_ee_pose": measured_pose.as_pos_quat(),
            "action/follower_qpos": out.follower_q_arm,
            "action/follower_ee_pose": out.follower_target_pose.as_pos_quat(),
            "action/gripper": np.array([state.gripper]),
            "timestamp": np.float64(time.perf_counter() - self._t0),
        }
        for name, frame in frames.items():
            data[f"observation/images/{name}"] = frame.color
            if self.cfg.record.record_depth and frame.depth is not None:
                data[f"observation/depth/{name}"] = frame.depth
        s.recorder.add_step(data)

    # -- run ----------------------------------------------------------------
    def run(self, record: bool = False) -> int:
        """Run the loop until ``loop.max_steps`` (or Ctrl-C). Returns step count."""
        self.setup()
        s = self.sys
        if record:
            meta = {
                "task": self.cfg.task,
                "leader": self.cfg.leader.robot,
                "follower": self.cfg.follower.robot,
                "camera_names": s.cameras.names,
                "follower_joint_names": list(s.follower.joint_names),
                "leader_joint_names": list(s.leader.joint_names),
            }
            s.recorder.start_episode(metadata=meta)

        rate = Rate(self.cfg.loop.rate_hz)
        rate.reset()
        n = 0
        try:
            while self.cfg.loop.max_steps is None or n < self.cfg.loop.max_steps:
                self.step(record=record)
                n += 1
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if record and s.recorder.is_recording:
                s.recorder.end_episode()
            self.shutdown()
        return n
