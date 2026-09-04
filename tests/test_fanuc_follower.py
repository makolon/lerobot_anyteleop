"""ROS-independent tests for the FANUC adapter's safety boundary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lerobot_anyteleop.devices.follower.fanuc import (
    CRX10IA_L_HOME,
    CRX10IA_L_JOINT_NAMES,
    CRX10IA_L_UPPER_LIMITS,
    FanucCommandSafety,
    FanucError,
    FanucROS2Follower,
    FanucSafetyError,
    JointStateBuffer,
    JointStateUnavailable,
    min_jerk_profile,
)
from lerobot_anyteleop.devices.gripper.robotiq import RobotiqGripper
from lerobot_anyteleop.config import TeleopConfig
from lerobot_anyteleop.factory import build_follower_device, build_gripper
from lerobot_anyteleop.robots import get_robot_spec


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class _Duration:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _Point:
    def __init__(self) -> None:
        self.positions = []
        self.time_from_start = _Duration()


class _Trajectory:
    def __init__(self) -> None:
        self.joint_names = []
        self.points = []


class _Publisher:
    def __init__(self, subscribers: int = 1) -> None:
        self.messages = []
        self.subscribers = subscribers

    def publish(self, message) -> None:
        self.messages.append(message)

    def get_subscription_count(self) -> int:
        return self.subscribers


class _FailingPublisher(_Publisher):
    def publish(self, message) -> None:
        raise OSError("DDS publish failed")


class _FailingOncePublisher(_Publisher):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def publish(self, message) -> None:
        self.calls += 1
        if self.calls == 1:
            raise OSError("transient DDS failure")
        super().publish(message)


class _ParameterValue:
    def __init__(self, *, string="", double=0.0, boolean=False) -> None:
        self.string_value = string
        self.double_value = double
        self.bool_value = boolean


class _ParameterFuture:
    def __init__(self, values) -> None:
        self._result = type("Response", (), {"values": values})()

    def add_done_callback(self, callback) -> None:
        callback(self)

    def result(self):
        return self._result


class _ParameterClient:
    def __init__(self, values) -> None:
        self.values = values

    def wait_for_services(self, timeout_sec) -> bool:
        return True

    def get_parameters(self, names):
        assert names == [
            "robot_model",
            "publish_rate_hz",
            "join_enabled",
            "setpoint_source",
            "require_supervisor",
        ]
        return _ParameterFuture(self.values)


def _ready_follower(clock: _Clock, **kwargs) -> tuple[FanucROS2Follower, _Publisher]:
    follower = FanucROS2Follower(clock=clock, **kwargs)
    publisher = _Publisher()
    follower._connected = True
    follower._publisher = publisher
    follower._trajectory_type = _Trajectory
    follower._point_type = _Point
    follower._parameter_verified = True
    follower.ingest_joint_state(CRX10IA_L_JOINT_NAMES, CRX10IA_L_HOME)
    # Two distinct samples prove that tick_seq, rather than only the status timer, advances.
    follower.ingest_stream_status(mode=1, tick_seq=100, command_position=CRX10IA_L_HOME)
    follower.ingest_stream_status(mode=1, tick_seq=125, command_position=CRX10IA_L_HOME)
    follower.ingest_supervisor_state(state=1)
    return follower, publisher


def test_sample_config_builds_without_ros_or_hardware_imports() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = TeleopConfig.from_yaml(repo / "configs" / "crx10ia_l.yaml")
    spec = get_robot_spec(config.follower.robot)
    follower = build_follower_device(config.follower, spec)
    gripper = build_gripper(config.follower.gripper, follower, config.follower.ip)

    assert isinstance(follower, FanucROS2Follower)
    assert follower.ros_joint_names == list(CRX10IA_L_JOINT_NAMES)
    assert follower.verify_executor_parameters
    assert isinstance(gripper, RobotiqGripper)
    assert gripper.com_port == "/dev/ttyUSB0"
    assert gripper.device_id == 9


def test_joint_state_buffer_reorders_extra_joints_and_rejects_stale() -> None:
    clock = _Clock()
    buffer = JointStateBuffer(clock=clock)
    names = ["camera", "J3", "J1", "J6", "J2", "J5", "J4"]
    values = [99.0, 3.0, 1.0, 6.0, 2.0, 5.0, 4.0]
    buffer.update(names, values)
    np.testing.assert_array_equal(buffer.read(max_age_s=0.3).positions, np.arange(1.0, 7.0))
    clock.now += 0.31
    with pytest.raises(JointStateUnavailable, match="stale"):
        buffer.read(max_age_s=0.3)
    buffer.clear()
    with pytest.raises(JointStateUnavailable, match="no complete"):
        buffer.read(max_age_s=0.3)


@pytest.mark.parametrize(
    ("names", "positions", "match"),
    [
        (["J1"], [0.0], "missing"),
        ([*CRX10IA_L_JOINT_NAMES, "J1"], [0.0] * 7, "duplicate"),
        (CRX10IA_L_JOINT_NAMES, [0.0] * 5, "names"),
        (CRX10IA_L_JOINT_NAMES, [0.0, 0.0, np.nan, 0.0, 0.0, 0.0], "non-finite"),
    ],
)
def test_joint_state_buffer_rejects_invalid_messages(names, positions, match) -> None:
    with pytest.raises(ValueError, match=match):
        JointStateBuffer().update(names, positions)


def test_l_model_position_shape_finite_and_step_gates() -> None:
    safety = FanucCommandSafety.crx10ia_l(max_step_rad=0.1, limit_margin_rad=0.017)
    np.testing.assert_allclose(
        safety.validate_step(CRX10IA_L_HOME + 0.05, CRX10IA_L_HOME),
        CRX10IA_L_HOME + 0.05,
    )
    with pytest.raises(FanucSafetyError, match="shape"):
        safety.validate_target(np.zeros((2, 3)))
    with pytest.raises(FanucSafetyError, match="NaN or Inf"):
        safety.validate_target([0, 0, 0, math.nan, 0, 0])
    outside = CRX10IA_L_HOME.copy()
    outside[0] = CRX10IA_L_UPPER_LIMITS[0]
    with pytest.raises(FanucSafetyError, match="position limit"):
        safety.validate_target(outside)
    with pytest.raises(FanucSafetyError, match="step exceeds"):
        safety.validate_step(CRX10IA_L_HOME + 0.101, CRX10IA_L_HOME)


def test_min_jerk_home_profile_is_bounded_and_hits_target() -> None:
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.5
    times, positions = min_jerk_profile(
        CRX10IA_L_HOME, target, max_speed_rad_s=0.2, min_duration_s=1.0, sample_hz=30
    )
    assert len(times) == len(positions) >= 2
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(positions[-1], target)
    samples = np.vstack([CRX10IA_L_HOME, positions])
    sample_times = np.concatenate([[0.0], times])
    speed = np.abs(np.diff(samples[:, 0]) / np.diff(sample_times))
    assert speed.max() <= 0.2 * 1.001


def test_stream_command_is_named_and_invalid_jump_latches_hold() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock, command_duration_s=0.04)
    follower.enter_servo_mode()
    target = CRX10IA_L_HOME + 0.01
    follower.send_joint_positions(target)
    message = publisher.messages[0]
    assert message.joint_names == list(CRX10IA_L_JOINT_NAMES)
    assert message.points[0].positions == pytest.approx(target)
    assert message.points[0].time_from_start.nanosec == 40_000_000

    with pytest.raises(FanucSafetyError, match="step exceeds"):
        follower.send_joint_positions(target + 0.11)
    assert follower.fault_reason is not None
    assert len(publisher.messages) == 2
    assert publisher.messages[-1].points[0].positions == pytest.approx(CRX10IA_L_HOME)


def test_stream_command_applies_speed_scaled_time_normalized_step() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(
        clock, max_step_rad=0.02, command_duration_s=0.04
    )
    follower.ingest_stream_status(
        mode=1,
        tick_seq=150,
        command_position=CRX10IA_L_HOME,
        speed_scale=0.25,
    )
    follower.enter_servo_mode()

    first_request = CRX10IA_L_HOME + 0.016
    first_applied = follower.send_joint_positions(first_request)
    np.testing.assert_allclose(first_applied, CRX10IA_L_HOME + 0.005)
    assert publisher.messages[-1].points[0].positions == pytest.approx(first_applied)

    # A second immediate call cannot multiply the per-cycle allowance.
    second_request = CRX10IA_L_HOME + 0.032
    second_applied = follower.send_joint_positions(second_request)
    np.testing.assert_allclose(second_applied, first_applied)

    # Half a nominal period permits half of the scaled step ceiling.
    clock.now += 0.02
    third_applied = follower.send_joint_positions(second_request)
    np.testing.assert_allclose(third_applied, CRX10IA_L_HOME + 0.0075)

    # The returned value is a copy, not an alias to the safety state.
    third_applied[0] += 1.0
    assert follower._last_command[0] == pytest.approx(CRX10IA_L_HOME[0] + 0.0075)


def test_stale_state_and_watchdog_fail_closed_to_executor_setpoint_hold() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock, watchdog_timeout_s=0.25)
    follower.enter_servo_mode()
    follower.send_joint_positions(CRX10IA_L_HOME + 0.01)
    clock.now += 0.26
    assert follower._watchdog_check()
    assert "watchdog expired" in follower.fault_reason
    assert publisher.messages[-1].points[0].positions == pytest.approx(CRX10IA_L_HOME)
    with pytest.raises(FanucSafetyError, match="latched off"):
        follower.enter_servo_mode()


def test_home_rolls_small_one_point_targets_through_executor() -> None:
    clock = _Clock()

    def advance(seconds: float) -> None:
        clock.now += seconds

    follower, publisher = _ready_follower(
        clock,
        home_min_duration_s=0.1,
        home_tolerance_rad=0.1,
        home_settle_s=0.0,
        home_timeout_s=2.0,
        state_timeout_s=2.0,
        status_timeout_s=2.0,
        watchdog_timeout_s=1.0,
        sleeper=advance,
    )
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.05
    follower.move_to_joint_positions(target, blocking=True)
    assert len(publisher.messages) >= 2
    assert all(m.joint_names == list(CRX10IA_L_JOINT_NAMES) for m in publisher.messages)
    assert all(len(m.points) == 1 for m in publisher.messages)
    assert publisher.messages[-1].points[0].positions == pytest.approx(target)


def test_home_progress_uses_executor_speed_scale() -> None:
    clock = _Clock()

    def advance(seconds: float) -> None:
        clock.now += seconds

    follower, _publisher = _ready_follower(
        clock,
        home_min_duration_s=0.1,
        home_tolerance_rad=0.1,
        home_settle_s=0.0,
        home_timeout_s=5.0,
        state_timeout_s=5.0,
        status_timeout_s=5.0,
        watchdog_timeout_s=1.0,
        sleeper=advance,
    )
    follower.ingest_stream_status(
        mode=1,
        tick_seq=150,
        command_position=CRX10IA_L_HOME,
        speed_scale=0.25,
    )
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.05
    nominal_times, _ = min_jerk_profile(
        CRX10IA_L_HOME,
        target,
        max_speed_rad_s=follower.home_speed_rad_s,
        min_duration_s=follower.home_min_duration_s,
        sample_hz=follower.home_sample_hz,
    )
    started_at = clock.now
    follower.move_to_joint_positions(target)
    assert clock.now - started_at >= nominal_times[-1] / 0.25


def test_stop_cancels_rolling_home_before_later_waypoints() -> None:
    clock = _Clock()
    follower = None
    stopped = False

    def advance(seconds: float) -> None:
        nonlocal stopped
        clock.now += seconds
        if not stopped and clock.now >= 10.01:
            stopped = True
            assert follower is not None
            follower.stop()

    follower, publisher = _ready_follower(
        clock,
        home_min_duration_s=0.1,
        home_timeout_s=2.0,
        state_timeout_s=2.0,
        status_timeout_s=2.0,
        watchdog_timeout_s=1.0,
        sleeper=advance,
    )
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.05
    with pytest.raises(FanucSafetyError, match="homing was cancelled"):
        follower.move_to_joint_positions(target)
    assert publisher.messages
    assert all(
        message.points[0].positions == pytest.approx(CRX10IA_L_HOME)
        for message in publisher.messages
    )
    assert follower.fault_reason is None


def test_home_setup_failure_latches_and_holds_current_executor_target() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)

    def fail_ready():
        raise FanucError("synthetic readiness failure")

    follower._wait_motion_ready = fail_ready
    with pytest.raises(FanucError, match="synthetic readiness failure"):
        follower.move_to_joint_positions(CRX10IA_L_HOME)
    assert follower.fault_reason is not None
    assert publisher.messages[-1].points[0].positions == pytest.approx(CRX10IA_L_HOME)


def test_invalid_home_target_does_not_disarm_existing_servo_watchdog() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)
    follower.enter_servo_mode()
    follower.send_joint_positions(CRX10IA_L_HOME + 0.01)
    invalid = CRX10IA_L_HOME.copy()
    invalid[0] = CRX10IA_L_UPPER_LIMITS[0]

    with pytest.raises(FanucSafetyError, match="position limit"):
        follower.move_to_joint_positions(invalid)
    assert follower._servo_mode
    assert follower._watchdog_armed
    assert len(publisher.messages) == 1


def test_stale_joint_state_stops_rolling_home() -> None:
    clock = _Clock()

    def advance(seconds: float) -> None:
        clock.now += seconds

    follower, publisher = _ready_follower(
        clock,
        home_min_duration_s=0.1,
        home_timeout_s=2.0,
        state_timeout_s=0.015,
        status_timeout_s=2.0,
        watchdog_timeout_s=1.0,
        sleeper=advance,
    )
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.05
    with pytest.raises(JointStateUnavailable, match="stale"):
        follower.move_to_joint_positions(target)
    assert follower.fault_reason is not None
    # Only the initial source heartbeat and the fault hold are permitted.
    assert all(
        message.points[0].positions == pytest.approx(CRX10IA_L_HOME)
        for message in publisher.messages
    )


def test_nonblocking_home_is_rejected_without_publishing() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)
    with pytest.raises(ValueError, match="non-blocking FANUC homing is disabled"):
        follower.move_to_joint_positions(CRX10IA_L_HOME, blocking=False)
    assert not publisher.messages


def test_target_cannot_run_far_ahead_of_executor_command() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(
        clock, max_step_rad=0.2, max_target_lead_rad=0.1
    )
    follower.enter_servo_mode()
    target = CRX10IA_L_HOME.copy()
    target[0] = 0.11
    with pytest.raises(FanucSafetyError, match="target lead"):
        follower.send_joint_positions(target)
    assert follower.fault_reason is not None
    assert len(publisher.messages) == 1  # current executor command setpoint hold


def test_stale_status_never_replays_old_measured_position_as_hold() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)
    follower.enter_servo_mode()
    follower.send_joint_positions(CRX10IA_L_HOME + 0.01)
    assert len(publisher.messages) == 1
    clock.now += 0.31
    with pytest.raises(JointStateUnavailable, match="stale"):
        follower.send_joint_positions(CRX10IA_L_HOME + 0.01)
    # No second message: both status and measured data are stale.
    assert len(publisher.messages) == 1


def test_publish_failure_is_latched_without_masking_transport_error() -> None:
    clock = _Clock()
    follower, _publisher = _ready_follower(clock)
    follower._publisher = _FailingPublisher()
    follower.enter_servo_mode()
    with pytest.raises(OSError, match="DDS publish failed"):
        follower.send_joint_positions(CRX10IA_L_HOME + 0.01)
    assert follower.fault_reason is not None
    assert "setpoint hold also failed" in follower.fault_reason


def test_stream_or_supervisor_not_run_blocks_motion() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)
    follower.ingest_supervisor_state(state=2, reason="operator resume required")
    problem = follower._health_error(require_run=True)
    assert problem is not None and "supervisor is not RUN" in problem
    with pytest.raises(FanucSafetyError, match="supervisor is not RUN"):
        follower._servo_mode = True
        follower._last_command = CRX10IA_L_HOME.copy()
        follower.send_joint_positions(CRX10IA_L_HOME)
    assert len(publisher.messages) == 1  # executor-command hold, never requested motion
    assert publisher.messages[0].points[0].positions == pytest.approx(CRX10IA_L_HOME)


def test_executor_parameter_preflight_accepts_only_l_500hz_waypoints() -> None:
    values = [
        _ParameterValue(string="crx10ia_l"),
        _ParameterValue(double=500.0),
        _ParameterValue(boolean=False),
        _ParameterValue(string="waypoints"),
        _ParameterValue(boolean=True),
    ]
    follower = FanucROS2Follower()
    follower._connected = True
    follower._parameter_client = _ParameterClient(values)
    follower._verify_executor_configuration()
    assert follower._parameter_verified

    values[0] = _ParameterValue(string="crx10ia")
    wrong = FanucROS2Follower()
    wrong._connected = True
    wrong._parameter_client = _ParameterClient(values)
    with pytest.raises(FanucSafetyError, match="robot_model='crx10ia'"):
        wrong._verify_executor_configuration()

    no_supervisor_values = [
        _ParameterValue(string="crx10ia_l"),
        _ParameterValue(double=500.0),
        _ParameterValue(boolean=False),
        _ParameterValue(string="waypoints"),
        _ParameterValue(boolean=False),
    ]
    no_supervisor = FanucROS2Follower()
    no_supervisor._connected = True
    no_supervisor._parameter_client = _ParameterClient(no_supervisor_values)
    with pytest.raises(FanucSafetyError, match="require_supervisor=false"):
        no_supervisor._verify_executor_configuration()


def test_stale_parameter_result_cannot_verify_a_replaced_ros_client() -> None:
    values = [
        _ParameterValue(string="crx10ia_l"),
        _ParameterValue(double=500.0),
        _ParameterValue(boolean=False),
        _ParameterValue(string="waypoints"),
        _ParameterValue(boolean=True),
    ]
    follower = FanucROS2Follower()
    follower._connected = True

    class _ReplacingFuture(_ParameterFuture):
        def result(self):
            follower._parameter_client = object()
            return super().result()

    class _ReplacingClient(_ParameterClient):
        def get_parameters(self, names):
            assert len(names) == 5
            return _ReplacingFuture(self.values)

    follower._parameter_client = _ReplacingClient(values)
    with pytest.raises(FanucSafetyError, match="connection changed"):
        follower._verify_executor_configuration()
    assert not follower._parameter_verified


def test_executor_timing_gate_rejects_slow_effective_loop() -> None:
    clock = _Clock()
    follower, _publisher = _ready_follower(clock)
    follower.ingest_stream_status(
        mode=1,
        tick_seq=150,
        command_position=CRX10IA_L_HOME,
        period_p99_us=3000.0,
    )
    assert "period_p99" in follower._health_error(require_run=True)


def test_failed_stop_publish_can_be_retried() -> None:
    clock = _Clock()
    follower, _publisher = _ready_follower(clock)
    publisher = _FailingOncePublisher()
    follower._publisher = publisher
    with pytest.raises(OSError, match="transient DDS failure"):
        follower.stop()
    assert not follower._stop_published
    follower.stop()
    assert follower._stop_published
    assert publisher.calls == 2


def test_stop_without_fresh_status_can_retry_after_status_recovers() -> None:
    clock = _Clock()
    follower, publisher = _ready_follower(clock)
    follower._stream = None
    follower.stop()
    assert not follower._stop_published
    assert not publisher.messages

    follower.ingest_stream_status(
        mode=1, tick_seq=150, command_position=CRX10IA_L_HOME
    )
    follower.stop()
    assert follower._stop_published
    assert len(publisher.messages) == 1


def test_enter_servo_mode_cannot_undo_stop_during_readiness_wait() -> None:
    clock = _Clock()
    follower, _publisher = _ready_follower(clock)

    def stop_while_waiting():
        snapshot = follower._states.read(max_age_s=follower.state_timeout_s)
        follower.stop()
        return snapshot

    follower._wait_motion_ready = stop_while_waiting
    with pytest.raises(FanucSafetyError, match="servo-mode entry was cancelled"):
        follower.enter_servo_mode()
    assert not follower._servo_mode


def test_enter_servo_mode_rejects_active_home() -> None:
    clock = _Clock()
    follower, _publisher = _ready_follower(clock)
    follower._homing_generation = follower._motion_generation
    with pytest.raises(FanucSafetyError, match="homing move is active"):
        follower.enter_servo_mode()


def test_connect_refuses_to_overwrite_retained_ros_resources() -> None:
    follower = FanucROS2Follower()
    retained_context = object()
    follower._context = retained_context
    follower._watchdog_stop.set()

    with pytest.raises(FanucError, match="cleanup is incomplete"):
        follower.connect()
    assert follower._context is retained_context
    assert follower._watchdog_stop.is_set()


def test_stream_status_command_is_name_mapped() -> None:
    clock = _Clock()
    follower = FanucROS2Follower(clock=clock)
    follower.ingest_stream_status(
        mode=1,
        tick_seq=1,
        joint_names=list(reversed(CRX10IA_L_JOINT_NAMES)),
        command_position=[6, 5, 4, 3, 2, 1],
    )
    np.testing.assert_array_equal(follower._fresh_stream_command(), np.arange(1.0, 7.0))
