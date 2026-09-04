"""FANUC CRX-10iA/L follower through the ROS 2 ``ws_fanuc`` stream stack.

This driver intentionally does **not** publish directly to
``forward_position_controller``.  Python supplies named, low-rate waypoints to
``fanuc_stream_executor``; its C++ loop performs the 500 Hz publication and the
ws_fanuc safety supervisor monitors the physical driver.

The ROS imports are lazy so kinematics/tests remain usable without ROS 2.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .base import FollowerInterface

CRX10IA_L_JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
CRX10IA_L_LOWER_LIMITS = np.array(
    [-3.139847324337799, -3.139847324337799, -4.71238898038469,
     -3.3161255787892263, -3.139847324337799, -3.9269908169872414],
    dtype=np.float64,
)
CRX10IA_L_UPPER_LIMITS = np.array(
    [3.139847324337799, 3.139847324337799, 4.71238898038469,
     3.3161255787892263, 3.139847324337799, 3.9269908169872414],
    dtype=np.float64,
)
CRX10IA_L_HOME = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2.0, 0.0])

DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_TRAJECTORY_TOPIC = "/stream_executor/joint_trajectory"
DEFAULT_STREAM_STATUS_TOPIC = "/stream_executor/status"
DEFAULT_SUPERVISOR_STATE_TOPIC = "/safety_supervisor/state"

STREAM_MODE_RUN = 1
SUPERVISOR_STATE_RUN = 1


class FanucError(RuntimeError):
    """Base error for an unavailable or unhealthy FANUC stream stack."""


class FanucSafetyError(FanucError):
    """A command was rejected or motion was latched off for safety."""


class JointStateUnavailable(FanucError):
    """No complete, fresh CRX joint state is available."""


@dataclass(frozen=True)
class JointStateSnapshot:
    positions: np.ndarray
    received_at: float
    source_timestamp: float | None = None


class JointStateBuffer:
    """Thread-safe, name-mapped latest-state buffer with a freshness gate."""

    def __init__(
        self,
        joint_names: Sequence[str] = CRX10IA_L_JOINT_NAMES,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self._clock = clock
        self._condition = threading.Condition()
        self._snapshot: JointStateSnapshot | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def update(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        received_at: float | None = None,
        source_timestamp: float | None = None,
    ) -> JointStateSnapshot:
        msg_names = tuple(names)
        msg_positions = tuple(positions)
        if len(msg_names) != len(msg_positions):
            raise ValueError(
                f"joint state has {len(msg_names)} names but {len(msg_positions)} positions"
            )
        if len(set(msg_names)) != len(msg_names):
            raise ValueError("joint state contains duplicate names")
        by_name = dict(zip(msg_names, msg_positions, strict=True))
        missing = [name for name in self.joint_names if name not in by_name]
        if missing:
            raise ValueError(f"joint state is missing required joints: {missing}")
        ordered = np.asarray([by_name[name] for name in self.joint_names], dtype=np.float64)
        if ordered.shape != (len(self.joint_names),) or not np.all(np.isfinite(ordered)):
            raise ValueError("joint state contains non-finite positions")
        stamp = self._clock() if received_at is None else float(received_at)
        if not math.isfinite(stamp):
            raise ValueError("joint state receive timestamp is not finite")
        source = None if source_timestamp is None else float(source_timestamp)
        if source is not None and not math.isfinite(source):
            raise ValueError("joint state source timestamp is not finite")
        snapshot = JointStateSnapshot(ordered.copy(), stamp, source)
        with self._condition:
            self._snapshot = snapshot
            self._last_error = None
            self._condition.notify_all()
        return snapshot

    def reject(self, error: BaseException | str) -> None:
        with self._condition:
            self._last_error = str(error)
            self._condition.notify_all()

    def clear(self) -> None:
        """Drop data from a previous ROS connection/session."""

        with self._condition:
            self._snapshot = None
            self._last_error = None
            self._condition.notify_all()

    def read(self, *, max_age_s: float, now: float | None = None) -> JointStateSnapshot:
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        current = self._clock() if now is None else float(now)
        with self._condition:
            snapshot = self._snapshot
            last_error = self._last_error
        if snapshot is None:
            detail = f"; last rejected message: {last_error}" if last_error else ""
            raise JointStateUnavailable(f"no complete joint state received{detail}")
        age = current - snapshot.received_at
        if age < 0 or age > max_age_s:
            raise JointStateUnavailable(
                f"joint state is stale (age={age:.3f}s, limit={max_age_s:.3f}s)"
            )
        return JointStateSnapshot(
            snapshot.positions.copy(), snapshot.received_at, snapshot.source_timestamp
        )

    def wait(self, *, timeout_s: float, max_age_s: float) -> JointStateSnapshot:
        deadline = self._clock() + timeout_s
        last_error: JointStateUnavailable | None = None
        while True:
            try:
                return self.read(max_age_s=max_age_s)
            except JointStateUnavailable as exc:
                last_error = exc
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise JointStateUnavailable(
                    f"timed out after {timeout_s:.2f}s waiting for a fresh joint state "
                    f"({last_error})"
                ) from last_error
            with self._condition:
                self._condition.wait(timeout=min(0.05, remaining))

    def latest(self) -> JointStateSnapshot | None:
        with self._condition:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        return JointStateSnapshot(
            snapshot.positions.copy(), snapshot.received_at, snapshot.source_timestamp
        )


@dataclass(frozen=True)
class FanucCommandSafety:
    lower: np.ndarray
    upper: np.ndarray
    max_step_rad: float = 0.02
    limit_margin_rad: float = 0.017

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (6,) or upper.shape != (6,):
            raise ValueError("CRX limits must each have shape (6,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("joint limits must be finite")
        if np.any(lower >= upper):
            raise ValueError("every lower joint limit must be below its upper limit")
        if not math.isfinite(self.max_step_rad) or self.max_step_rad <= 0:
            raise ValueError("max_step_rad must be a positive finite number")
        if not math.isfinite(self.limit_margin_rad) or self.limit_margin_rad < 0:
            raise ValueError("limit_margin_rad must be a non-negative finite number")
        if np.any(lower + self.limit_margin_rad >= upper - self.limit_margin_rad):
            raise ValueError("limit_margin_rad leaves an empty position range")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @classmethod
    def crx10ia_l(
        cls, *, max_step_rad: float = 0.02, limit_margin_rad: float = 0.017
    ) -> "FanucCommandSafety":
        return cls(
            CRX10IA_L_LOWER_LIMITS,
            CRX10IA_L_UPPER_LIMITS,
            max_step_rad=max_step_rad,
            limit_margin_rad=limit_margin_rad,
        )

    @staticmethod
    def vector(values: Sequence[float], what: str = "joint command") -> np.ndarray:
        q = np.asarray(values, dtype=np.float64)
        if q.shape != (6,):
            raise FanucSafetyError(f"{what} must have shape (6,), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise FanucSafetyError(f"{what} contains NaN or Inf")
        return q.copy()

    def validate_target(self, values: Sequence[float]) -> np.ndarray:
        q = self.vector(values)
        lower = self.lower + self.limit_margin_rad
        upper = self.upper - self.limit_margin_rad
        bad = np.flatnonzero((q < lower) | (q > upper))
        if len(bad):
            detail = ", ".join(
                f"J{i + 1}={q[i]:.5f} outside [{lower[i]:.5f}, {upper[i]:.5f}]"
                for i in bad
            )
            raise FanucSafetyError(f"CRX-10iA/L position limit violation: {detail}")
        return q

    def validate_step(
        self, target: Sequence[float], reference: Sequence[float]
    ) -> np.ndarray:
        q = self.validate_target(target)
        ref = self.vector(reference, "command reference")
        delta = np.abs(q - ref)
        bad = np.flatnonzero(delta > self.max_step_rad)
        if len(bad):
            detail = ", ".join(f"J{i + 1}={delta[i]:.5f}" for i in bad)
            raise FanucSafetyError(
                f"joint command step exceeds {self.max_step_rad:.5f} rad: {detail}"
            )
        return q

    def hold_target(self, values: Sequence[float]) -> np.ndarray:
        """Validate and clamp only to absolute limits for a best-effort hold."""

        q = self.vector(values, "hold target")
        return np.clip(q, self.lower, self.upper)


def min_jerk_profile(
    start: Sequence[float],
    target: Sequence[float],
    *,
    max_speed_rad_s: float = 0.25,
    min_duration_s: float = 2.0,
    sample_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, positions)`` for a speed-bounded minimum-jerk move."""

    q0 = FanucCommandSafety.vector(start, "home start")
    q1 = FanucCommandSafety.vector(target, "home target")
    for value, name in (
        (max_speed_rad_s, "max_speed_rad_s"),
        (min_duration_s, "min_duration_s"),
        (sample_hz, "sample_hz"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    # max(s'(u)) for 10u^3 - 15u^4 + 6u^5 is 1.875.
    duration = max(
        float(min_duration_s),
        1.875 * float(np.max(np.abs(q1 - q0))) / float(max_speed_rad_s),
    )
    count = max(2, int(math.ceil(duration * sample_hz)))
    times = np.linspace(duration / count, duration, count, dtype=np.float64)
    u = times / duration
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    positions = q0[None, :] + blend[:, None] * (q1 - q0)[None, :]
    positions[-1] = q1
    return times, positions


@dataclass(frozen=True)
class _StreamSnapshot:
    mode: int
    tick_seq: int
    plan_seq: int
    command_position: np.ndarray | None
    publish_rate_hz: float
    period_p99_us: float
    period_max_us: float
    speed_scale: float
    plan_progress: float
    received_at: float
    last_error: str


@dataclass(frozen=True)
class _SupervisorSnapshot:
    state: int
    received_at: float
    reason: str


def _duration_parts(seconds: float) -> tuple[int, int]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("trajectory time must be a positive finite number")
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return sec, nanosec


def _ordered_named_vector(
    expected_names: Sequence[str],
    names: Sequence[str],
    values: Sequence[float],
    *,
    what: str,
) -> np.ndarray:
    msg_names = tuple(names)
    msg_values = tuple(values)
    if len(msg_names) != len(msg_values):
        raise ValueError(f"{what} names/value lengths differ")
    if len(set(msg_names)) != len(msg_names):
        raise ValueError(f"{what} contains duplicate joint names")
    by_name = dict(zip(msg_names, msg_values, strict=True))
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise ValueError(f"{what} is missing required joints: {missing}")
    return FanucCommandSafety.vector(
        [by_name[name] for name in expected_names], what
    )


class FanucROS2Follower(FollowerInterface):
    """CRX-10iA/L backend for a separately launched ``ws_fanuc`` stack."""

    def __init__(
        self,
        ip: str | None = None,
        joint_names: Sequence[str] = CRX10IA_L_JOINT_NAMES,
        *,
        joint_prefix: str = "",
        joint_states_topic: str = DEFAULT_JOINT_STATES_TOPIC,
        trajectory_topic: str = DEFAULT_TRAJECTORY_TOPIC,
        stream_status_topic: str = DEFAULT_STREAM_STATUS_TOPIC,
        supervisor_state_topic: str = DEFAULT_SUPERVISOR_STATE_TOPIC,
        state_timeout_s: float = 0.30,
        status_timeout_s: float = 0.30,
        ready_timeout_s: float = 10.0,
        watchdog_timeout_s: float = 0.25,
        max_period_p99_us: float = 2400.0,
        max_period_max_us: float = 6000.0,
        max_step_rad: float = 0.02,
        max_target_lead_rad: float = 0.10,
        limit_margin_rad: float = 0.017,
        command_duration_s: float = 1.0 / 30.0,
        home_speed_rad_s: float = 0.25,
        home_min_duration_s: float = 2.0,
        home_sample_hz: float = 30.0,
        home_tolerance_rad: float = 0.01,
        home_settle_s: float = 0.20,
        home_timeout_s: float = 120.0,
        require_stream_status: bool = True,
        require_supervisor: bool = True,
        verify_executor_parameters: bool = True,
        stream_node_name: str = "/stream_executor",
        node_name: str = "anyteleop_fanuc_follower",
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ip = ip  # Informational; stream_physical.launch.py owns the connection.
        self.joint_names = list(joint_names)
        if tuple(self.joint_names) != CRX10IA_L_JOINT_NAMES:
            raise ValueError(
                f"CRX-10iA/L requires kinematic joint order {CRX10IA_L_JOINT_NAMES}, "
                f"got {tuple(self.joint_names)}"
            )
        self.ros_joint_names = [f"{joint_prefix}{name}" for name in self.joint_names]
        self.joint_states_topic = str(joint_states_topic)
        self.trajectory_topic = str(trajectory_topic)
        self.stream_status_topic = str(stream_status_topic)
        self.supervisor_state_topic = str(supervisor_state_topic)
        self.state_timeout_s = self._positive(state_timeout_s, "state_timeout_s")
        self.status_timeout_s = self._positive(status_timeout_s, "status_timeout_s")
        self.ready_timeout_s = self._positive(ready_timeout_s, "ready_timeout_s")
        self.watchdog_timeout_s = self._positive(watchdog_timeout_s, "watchdog_timeout_s")
        self.max_period_p99_us = self._positive(max_period_p99_us, "max_period_p99_us")
        self.max_period_max_us = self._positive(max_period_max_us, "max_period_max_us")
        self.max_target_lead_rad = self._positive(
            max_target_lead_rad, "max_target_lead_rad"
        )
        self.command_duration_s = self._positive(command_duration_s, "command_duration_s")
        self.home_speed_rad_s = self._positive(home_speed_rad_s, "home_speed_rad_s")
        self.home_min_duration_s = self._positive(home_min_duration_s, "home_min_duration_s")
        self.home_sample_hz = self._positive(home_sample_hz, "home_sample_hz")
        self.home_tolerance_rad = self._positive(home_tolerance_rad, "home_tolerance_rad")
        self.home_settle_s = self._positive(home_settle_s, "home_settle_s", allow_zero=True)
        self.home_timeout_s = self._positive(home_timeout_s, "home_timeout_s")
        self.require_stream_status = bool(require_stream_status)
        self.require_supervisor = bool(require_supervisor)
        self.verify_executor_parameters = bool(verify_executor_parameters)
        self.stream_node_name = "/" + str(stream_node_name).strip("/")
        self.node_name = str(node_name)
        self._clock = clock
        self._sleep = sleeper
        self._states = JointStateBuffer(self.ros_joint_names, clock=clock)
        self._safety = FanucCommandSafety.crx10ia_l(
            max_step_rad=max_step_rad, limit_margin_rad=limit_margin_rad
        )

        self._condition = threading.Condition()
        self._command_lock = threading.RLock()
        # Device lifecycle calls may join threads, so keep their serialization
        # separate from the command lock used by ROS callbacks/watchdog work.
        self._lifecycle_lock = threading.RLock()
        self._stream: _StreamSnapshot | None = None
        self._supervisor: _SupervisorSnapshot | None = None
        self._last_tick_seq: int | None = None
        self._tick_advanced_at: float | None = None
        self._thread_error: BaseException | None = None
        self._fault_reason: str | None = None
        self._last_command: np.ndarray | None = None
        self._last_requested_command: np.ndarray | None = None
        self._last_send_at: float | None = None
        self._connected = False
        self._servo_mode = False
        self._watchdog_armed = False
        self._stop_published = False
        # Incremented whenever motion starts or is cancelled. A homing loop may
        # only publish while its captured generation is still current.
        self._motion_generation = 0
        self._homing_generation: int | None = None

        self._rclpy: Any | None = None
        self._context: Any | None = None
        self._node: Any | None = None
        self._executor: Any | None = None
        self._executor_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._publisher: Any | None = None
        self._subscriptions: list[Any] = []
        self._trajectory_type: Any | None = None
        self._point_type: Any | None = None
        self._parameter_client: Any | None = None
        self._parameter_verified = False

    @staticmethod
    def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {relation} finite number")
        return result

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    @property
    def last_state_error(self) -> str | None:
        return self._states.last_error

    @staticmethod
    def _import_ros() -> dict[str, Any]:
        try:
            import rclpy
            from fanuc_stream_msgs.msg import StreamStatus, SupervisorState
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.parameter_client import AsyncParameterClient
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        except ImportError as exc:  # pragma: no cover - requires Ubuntu/ROS workspace
            raise RuntimeError(
                "ROS 2 Jazzy or ws_fanuc messages are unavailable. Source both "
                "/opt/ros/jazzy/setup.bash and ws_fanuc/install/setup.bash before "
                "starting anyteleop."
            ) from exc
        return locals()

    def _ros_resources_remain(self) -> bool:
        return any(
            resource is not None
            for resource in (
                self._context,
                self._node,
                self._executor,
                self._executor_thread,
                self._watchdog_thread,
                self._publisher,
                self._parameter_client,
            )
        ) or bool(self._subscriptions)

    def connect(self) -> None:
        with self._lifecycle_lock:
            self._connect_locked()

    def _connect_locked(self) -> None:
        if self._connected:
            return
        if self._ros_resources_remain():
            raise FanucError(
                "previous ROS cleanup is incomplete; call disconnect() again before reconnecting"
            )
        # A reconnect is an explicit recovery boundary; a fault remains latched
        # until all ROS resources have been torn down and connect() is called again.
        self._thread_error = None
        self._fault_reason = None
        self._stream = None
        self._supervisor = None
        self._last_tick_seq = None
        self._tick_advanced_at = None
        self._last_command = None
        self._last_requested_command = None
        self._last_send_at = None
        self._stop_published = False
        self._motion_generation += 1
        self._homing_generation = None
        self._states.clear()
        self._parameter_verified = False
        ros = self._import_ros()
        self._rclpy = ros["rclpy"]
        try:
            self._context = ros["Context"]()
            self._rclpy.init(args=None, context=self._context)
            self._node = self._rclpy.create_node(
                f"{self.node_name}_{os.getpid()}", context=self._context
            )
            qos = ros["QoSProfile"](
                depth=1,
                history=ros["QoSHistoryPolicy"].KEEP_LAST,
                reliability=ros["QoSReliabilityPolicy"].RELIABLE,
                durability=ros["QoSDurabilityPolicy"].VOLATILE,
            )
            self._trajectory_type = ros["JointTrajectory"]
            self._point_type = ros["JointTrajectoryPoint"]
            self._publisher = self._node.create_publisher(
                self._trajectory_type, self.trajectory_topic, qos
            )
            self._subscriptions = [
                self._node.create_subscription(
                    ros["JointState"], self.joint_states_topic, self._on_joint_state, qos
                ),
                self._node.create_subscription(
                    ros["StreamStatus"], self.stream_status_topic, self._on_stream_status, qos
                ),
                self._node.create_subscription(
                    ros["SupervisorState"],
                    self.supervisor_state_topic,
                    self._on_supervisor_state,
                    qos,
                ),
            ]
            self._executor = ros["SingleThreadedExecutor"](context=self._context)
            self._executor.add_node(self._node)
            self._parameter_client = ros["AsyncParameterClient"](
                self._node, self.stream_node_name
            )
            self._watchdog_stop.clear()
            self._executor_thread = threading.Thread(
                target=self._spin_executor,
                name="anyteleop-fanuc-ros",
                daemon=False,
            )
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="anyteleop-fanuc-watchdog",
                daemon=False,
            )
            self._executor_thread.start()
            self._watchdog_thread.start()
            with self._command_lock:
                if self._thread_error is not None:
                    raise FanucError(
                        f"ROS executor stopped during startup: {self._thread_error}"
                    )
                self._connected = True
        except BaseException as exc:
            try:
                self._cleanup_ros()
            except BaseException as cleanup_error:
                exc.add_note(f"ROS cleanup also failed: {cleanup_error}")
            raise

    def _spin_executor(self) -> None:
        try:
            self._executor.spin()
        except BaseException as exc:  # pragma: no cover - ROS runtime failure
            with self._condition:
                self._thread_error = exc
                self._condition.notify_all()
            self._latch_fault(f"ROS executor stopped: {exc}")

    def ingest_joint_state(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        received_at: float | None = None,
        source_timestamp: float | None = None,
    ) -> JointStateSnapshot:
        snapshot = self._states.update(
            names,
            positions,
            received_at=received_at,
            source_timestamp=source_timestamp,
        )
        with self._condition:
            self._condition.notify_all()
        return snapshot

    def ingest_stream_status(
        self,
        *,
        mode: int,
        tick_seq: int,
        plan_seq: int = 0,
        joint_names: Sequence[str] | None = None,
        command_position: Sequence[float] | None = None,
        publish_rate_hz: float = 500.0,
        period_p99_us: float = 2000.0,
        period_max_us: float = 2000.0,
        speed_scale: float = 1.0,
        plan_progress: float = 0.0,
        received_at: float | None = None,
        last_error: str = "",
    ) -> None:
        stamp = self._clock() if received_at is None else float(received_at)
        command = None
        if command_position is not None and len(command_position):
            status_names = self.ros_joint_names if joint_names is None else list(joint_names)
            command = _ordered_named_vector(
                self.ros_joint_names,
                status_names,
                command_position,
                what="executor command position",
            )
        publish_rate_hz = float(publish_rate_hz)
        period_p99_us = float(period_p99_us)
        period_max_us = float(period_max_us)
        for value, label in (
            (publish_rate_hz, "publish_rate_hz"),
            (period_p99_us, "period_p99_us"),
            (period_max_us, "period_max_us"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"executor {label} must be a non-negative finite number")
        speed_scale = float(speed_scale)
        plan_progress = float(plan_progress)
        if not math.isfinite(speed_scale) or not 0.0 <= speed_scale <= 1.0:
            raise ValueError("executor speed_scale must be finite and in [0, 1]")
        if not math.isfinite(plan_progress):
            raise ValueError("executor plan_progress must be finite")
        with self._condition:
            if self._last_tick_seq is not None and int(tick_seq) != self._last_tick_seq:
                self._tick_advanced_at = stamp
            self._last_tick_seq = int(tick_seq)
            self._stream = _StreamSnapshot(
                int(mode),
                int(tick_seq),
                int(plan_seq),
                command,
                publish_rate_hz,
                period_p99_us,
                period_max_us,
                speed_scale,
                plan_progress,
                stamp,
                str(last_error),
            )
            self._condition.notify_all()

    def ingest_supervisor_state(
        self, *, state: int, received_at: float | None = None, reason: str = ""
    ) -> None:
        stamp = self._clock() if received_at is None else float(received_at)
        with self._condition:
            self._supervisor = _SupervisorSnapshot(int(state), stamp, str(reason))
            self._condition.notify_all()

    def _on_joint_state(self, msg: Any) -> None:
        try:
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            source = None
            if stamp is not None:
                source = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            self.ingest_joint_state(msg.name, msg.position, source_timestamp=source)
        except (TypeError, ValueError) as exc:
            self._states.reject(exc)

    def _on_stream_status(self, msg: Any) -> None:
        try:
            self.ingest_stream_status(
                mode=msg.mode,
                tick_seq=msg.tick_seq,
                plan_seq=msg.plan_seq,
                joint_names=msg.joint_names,
                command_position=msg.command_position,
                publish_rate_hz=msg.publish_rate_hz,
                period_p99_us=msg.period_p99_us,
                period_max_us=msg.period_max_us,
                speed_scale=msg.speed_scale,
                plan_progress=msg.plan_progress,
                last_error=msg.last_error,
            )
        except (TypeError, ValueError, FanucSafetyError) as exc:
            self._latch_fault(f"invalid stream status: {exc}")

    def _on_supervisor_state(self, msg: Any) -> None:
        self.ingest_supervisor_state(state=msg.state, reason=msg.reason)

    def _health_error(self, *, require_run: bool, now: float | None = None) -> str | None:
        current = self._clock() if now is None else float(now)
        with self._condition:
            stream = self._stream
            supervisor = self._supervisor
            tick_at = self._tick_advanced_at
            thread_error = self._thread_error
        if thread_error is not None:
            return f"ROS executor failed: {thread_error}"
        if self.require_stream_status:
            if stream is None:
                return f"no status received on {self.stream_status_topic}"
            if current - stream.received_at > self.status_timeout_s:
                return f"stream status is stale ({current - stream.received_at:.3f}s)"
            if tick_at is None or current - tick_at > self.status_timeout_s:
                return "stream executor tick_seq is not advancing"
            if not math.isclose(stream.publish_rate_hz, 500.0, rel_tol=0.0, abs_tol=0.5):
                return (
                    f"stream executor reports publish_rate_hz={stream.publish_rate_hz:.1f}, "
                    "expected 500.0"
                )
            if stream.period_p99_us <= 0 or stream.period_max_us <= 0:
                return "stream executor timing statistics are not ready"
            if stream.period_p99_us > self.max_period_p99_us:
                return (
                    f"stream executor period_p99={stream.period_p99_us:.0f}us exceeds "
                    f"{self.max_period_p99_us:.0f}us"
                )
            if stream.period_max_us > self.max_period_max_us:
                return (
                    f"stream executor period_max={stream.period_max_us:.0f}us exceeds "
                    f"{self.max_period_max_us:.0f}us"
                )
            if require_run and stream.mode != STREAM_MODE_RUN:
                suffix = f": {stream.last_error}" if stream.last_error else ""
                return f"stream executor is not RUN (mode={stream.mode}){suffix}"
            if require_run and stream.command_position is None:
                return "stream executor status has no command_position"
        if self.require_supervisor:
            if supervisor is None:
                return f"no state received on {self.supervisor_state_topic}"
            if current - supervisor.received_at > self.status_timeout_s:
                return f"safety supervisor state is stale ({current - supervisor.received_at:.3f}s)"
            if require_run and supervisor.state != SUPERVISOR_STATE_RUN:
                suffix = f": {supervisor.reason}" if supervisor.reason else ""
                return f"safety supervisor is not RUN (state={supervisor.state}){suffix}"
        return None

    def _fresh_stream_command(self, now: float | None = None) -> np.ndarray:
        current = self._clock() if now is None else float(now)
        with self._condition:
            stream = self._stream
        if stream is None or stream.command_position is None:
            raise FanucSafetyError("no executor command position is available")
        age = current - stream.received_at
        if age < 0 or age > self.status_timeout_s:
            raise FanucSafetyError(
                f"executor command position is stale (age={age:.3f}s)"
            )
        return stream.command_position.copy()

    def _validate_target_lead(self, target: Sequence[float]) -> np.ndarray:
        q = self._safety.validate_target(target)
        if not self.require_stream_status:
            return q
        command = self._fresh_stream_command()
        delta = np.abs(q - command)
        bad = np.flatnonzero(delta > self.max_target_lead_rad)
        if len(bad):
            detail = ", ".join(f"J{i + 1}={delta[i]:.5f}" for i in bad)
            raise FanucSafetyError(
                f"target lead over executor command exceeds "
                f"{self.max_target_lead_rad:.5f} rad: {detail}"
            )
        return q

    def _subscriber_ready(self) -> bool:
        if self._publisher is None:
            return False
        getter = getattr(self._publisher, "get_subscription_count", None)
        return True if getter is None else getter() > 0

    def _verify_executor_configuration(self) -> None:
        """Fail closed if ws_fanuc was launched for the wrong robot/mode."""

        if self._parameter_verified or not self.verify_executor_parameters:
            self._parameter_verified = True
            return
        parameter_client = self._parameter_client
        if parameter_client is None:
            raise FanucError("stream executor parameter client is unavailable")
        if not parameter_client.wait_for_services(timeout_sec=self.ready_timeout_s):
            raise FanucError(
                f"parameter service for {self.stream_node_name} did not appear within "
                f"{self.ready_timeout_s:.1f}s"
            )
        names = [
            "robot_model",
            "publish_rate_hz",
            "join_enabled",
            "setpoint_source",
            "require_supervisor",
        ]
        future = parameter_client.get_parameters(names)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout=self.ready_timeout_s):
            raise FanucError(
                f"timed out reading parameters from {self.stream_node_name}"
            )
        try:
            values = future.result().values
        except Exception as exc:
            raise FanucError(
                f"failed to read parameters from {self.stream_node_name}: {exc}"
            ) from exc
        if len(values) != len(names):
            raise FanucError(
                f"{self.stream_node_name} returned {len(values)} parameters, expected {len(names)}"
            )
        actual = {
            "robot_model": values[0].string_value,
            "publish_rate_hz": values[1].double_value,
            "join_enabled": values[2].bool_value,
            "setpoint_source": values[3].string_value,
            "require_supervisor": values[4].bool_value,
        }
        problems = []
        if actual["robot_model"] != "crx10ia_l":
            problems.append(f"robot_model={actual['robot_model']!r} (must be 'crx10ia_l')")
        if not math.isclose(float(actual["publish_rate_hz"]), 500.0, rel_tol=0.0, abs_tol=0.5):
            problems.append(
                f"publish_rate_hz={actual['publish_rate_hz']!r} (must be 500.0)"
            )
        if bool(actual["join_enabled"]):
            problems.append("join_enabled=true (must be false for rolling one-point teleop)")
        if actual["setpoint_source"] != "waypoints":
            problems.append(
                f"setpoint_source={actual['setpoint_source']!r} (must be 'waypoints')"
            )
        if not bool(actual["require_supervisor"]):
            problems.append("require_supervisor=false (must be true for the safety dead-man)")
        if problems:
            raise FanucSafetyError(
                f"unsafe {self.stream_node_name} configuration: " + "; ".join(problems)
            )
        with self._command_lock:
            if not self._connected or self._parameter_client is not parameter_client:
                raise FanucSafetyError(
                    "FANUC ROS connection changed during executor parameter verification"
                )
            self._parameter_verified = True

    def _wait_motion_ready(self) -> JointStateSnapshot:
        self._verify_executor_configuration()
        deadline = self._clock() + self.ready_timeout_s
        last_problem = "waiting for ROS discovery"
        while True:
            try:
                state = self._states.read(max_age_s=self.state_timeout_s)
                state_problem = None
            except JointStateUnavailable as exc:
                state = None
                state_problem = str(exc)
            health_problem = self._health_error(require_run=True)
            subscriber_problem = None if self._subscriber_ready() else (
                f"no subscriber on {self.trajectory_topic}"
            )
            problems = [p for p in (state_problem, health_problem, subscriber_problem) if p]
            if state is not None and not problems:
                return state
            last_problem = "; ".join(problems)
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise FanucError(
                    f"FANUC stream stack not ready after {self.ready_timeout_s:.1f}s: "
                    f"{last_problem}. Start stream_physical with robot_model:=crx10ia_l "
                    "and resume the safety supervisor."
                )
            with self._condition:
                self._condition.wait(timeout=min(0.05, remaining))

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("FANUC follower is not connected")
        if self._fault_reason is not None:
            raise FanucSafetyError(f"FANUC follower is latched off: {self._fault_reason}")

    def get_joint_positions(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("FANUC follower is not connected")
        try:
            snapshot = self._states.read(max_age_s=self.state_timeout_s)
        except JointStateUnavailable:
            snapshot = self._states.wait(
                timeout_s=self.ready_timeout_s, max_age_s=self.state_timeout_s
            )
        return snapshot.positions

    def _make_trajectory(
        self, positions: Sequence[Sequence[float]], times_s: Sequence[float]
    ) -> Any:
        if self._trajectory_type is None or self._point_type is None:
            raise RuntimeError("FANUC follower is not connected")
        if len(positions) != len(times_s) or len(positions) == 0:
            raise ValueError("trajectory positions/times must have equal non-zero length")
        message = self._trajectory_type()
        message.joint_names = list(self.ros_joint_names)
        previous_time = 0.0
        for row, seconds in zip(positions, times_s, strict=True):
            q = self._safety.vector(row, "trajectory point")
            seconds = float(seconds)
            if seconds <= previous_time:
                raise ValueError("trajectory times must be strictly increasing and positive")
            previous_time = seconds
            point = self._point_type()
            point.positions = q.tolist()
            sec, nanosec = _duration_parts(seconds)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            message.points.append(point)
        return message

    def _publish(self, message: Any) -> None:
        if self._publisher is None:
            raise RuntimeError("FANUC follower is not connected")
        self._publisher.publish(message)

    def move_to_joint_positions(self, q: np.ndarray, blocking: bool = True) -> None:
        if not blocking:
            raise ValueError(
                "non-blocking FANUC homing is disabled: it cannot supervise a rolling "
                "setpoint stream"
            )
        # Validate before changing an existing control mode. An invalid caller
        # target must not silently disarm the source watchdog.
        target = self._safety.validate_target(q)
        with self._command_lock:
            self._require_connected()
            if self._homing_generation is not None:
                raise FanucSafetyError("a FANUC homing move is already active")
            if self._servo_mode or self._watchdog_armed:
                if not self._publish_hold_locked():
                    raise FanucSafetyError(
                        "cannot leave servo mode for homing without a fresh executor hold target"
                    )
            self._motion_generation += 1
            motion_generation = self._motion_generation
            self._homing_generation = motion_generation
            self._servo_mode = False
            self._watchdog_armed = False
            self._stop_published = False

        try:
            # Keep setup inside the ownership/fault boundary. In particular, a
            # readiness failure after servo mode was disabled must fail closed.
            start = self._wait_motion_ready().positions
            times, points = min_jerk_profile(
                start,
                target,
                max_speed_rad_s=self.home_speed_rad_s,
                min_duration_s=self.home_min_duration_s,
                sample_hz=self.home_sample_hz,
            )
            started_at = self._clock()
            deadline = started_at + self.home_timeout_s
            previous = start.copy()
            with self._command_lock:
                self._check_homing_health(deadline, motion_generation)
                self._last_command = previous.copy()
                self._last_requested_command = previous.copy()
                self._last_send_at = started_at
                self._watchdog_armed = True

            # Publish a rolling series of one-point holds instead of handing the
            # complete path to the executor. If this process disappears, only
            # the latest small, lead-bounded increment remains outstanding.
            profile_time = 0.0
            profile_updated_at = started_at
            heartbeat_period = min(0.10, self.watchdog_timeout_s / 2.0)
            next_heartbeat = started_at
            for relative_time, row in zip(times, points, strict=True):
                relative_time = float(relative_time)
                while profile_time < relative_time:
                    self._check_homing_health(deadline, motion_generation)
                    now = self._clock()
                    elapsed = now - profile_updated_at
                    if elapsed < 0:
                        raise FanucSafetyError("monotonic clock moved backwards while homing")
                    speed_scale = self._fresh_speed_scale(now)
                    profile_time += elapsed * speed_scale
                    profile_updated_at = now
                    if now >= next_heartbeat:
                        with self._command_lock:
                            self._check_homing_health(deadline, motion_generation)
                            hold = self._validate_target_lead(previous)
                            self._publish(
                                self._make_trajectory([hold], [self.command_duration_s])
                            )
                            self._last_send_at = self._clock()
                            self._stop_published = False
                        next_heartbeat = now + heartbeat_period
                    if profile_time < relative_time:
                        remaining = relative_time - profile_time
                        wait = 0.01 if speed_scale <= 0 else min(0.01, remaining / speed_scale)
                        self._sleep(max(0.0005, wait))
                with self._command_lock:
                    self._check_homing_health(deadline, motion_generation)
                    waypoint = self._safety.validate_step(row, previous)
                    waypoint = self._validate_target_lead(waypoint)
                    self._publish(
                        self._make_trajectory([waypoint], [self.command_duration_s])
                    )
                    previous = waypoint.copy()
                    self._last_command = previous.copy()
                    self._last_send_at = self._clock()
                    self._stop_published = False
                next_heartbeat = self._clock() + heartbeat_period

            settled_at: float | None = None
            next_heartbeat = self._clock()
            while True:
                self._check_homing_health(deadline, motion_generation)
                now = self._clock()
                measured = self._states.read(max_age_s=self.state_timeout_s).positions
                if float(np.max(np.abs(measured - target))) <= self.home_tolerance_rad:
                    settled_at = now if settled_at is None else settled_at
                    if now - settled_at >= self.home_settle_s:
                        with self._command_lock:
                            self._check_homing_health(deadline, motion_generation)
                            self._watchdog_armed = False
                            self._last_command = target.copy()
                            self._last_requested_command = target.copy()
                        return
                else:
                    settled_at = None
                if now >= next_heartbeat:
                    with self._command_lock:
                        self._check_homing_health(deadline, motion_generation)
                        final_target = self._validate_target_lead(target)
                        self._publish(
                            self._make_trajectory(
                                [final_target], [self.command_duration_s]
                            )
                        )
                        self._last_send_at = self._clock()
                        self._stop_published = False
                    next_heartbeat = now + heartbeat_period
                self._sleep(0.02)
        except Exception as exc:
            # A deliberate stop/newer motion owns the new generation and must
            # not be poisoned by this cancelled loop as it unwinds.
            with self._command_lock:
                still_owner = (
                    self._homing_generation == motion_generation
                    and self._motion_generation == motion_generation
                )
            if still_owner:
                self._latch_fault(f"homing failed: {exc}")
            raise
        finally:
            with self._command_lock:
                if self._homing_generation == motion_generation:
                    self._homing_generation = None

    def _check_homing_health(self, deadline: float, motion_generation: int) -> None:
        with self._command_lock:
            self._require_connected()
            if self._motion_generation != motion_generation:
                raise FanucSafetyError("homing was cancelled by a stop or newer motion")
        if self._clock() >= deadline:
            raise FanucSafetyError(
                f"homing timed out after {self.home_timeout_s:.1f}s"
            )
        problem = self._health_error(require_run=True)
        if problem:
            raise FanucSafetyError(f"health failure while homing: {problem}")
        # Unlike the status checks, this is deliberately performed for every
        # rolling waypoint: stale measured state must stop the home path too.
        self._states.read(max_age_s=self.state_timeout_s)

    def _fresh_speed_scale(self, now: float | None = None) -> float:
        if not self.require_stream_status:
            return 1.0
        current = self._clock() if now is None else float(now)
        with self._condition:
            stream = self._stream
        if stream is None or current - stream.received_at < 0:
            raise FanucSafetyError("no fresh executor speed_scale is available")
        if current - stream.received_at > self.status_timeout_s:
            raise FanucSafetyError("executor speed_scale is stale")
        return stream.speed_scale

    def enter_servo_mode(self) -> None:
        with self._command_lock:
            self._require_connected()
            if self._homing_generation is not None:
                raise FanucSafetyError(
                    "cannot enter servo mode while a FANUC homing move is active"
                )
            if self._servo_mode:
                return
            self._motion_generation += 1
            motion_generation = self._motion_generation
        self._wait_motion_ready()
        with self._command_lock:
            self._require_connected()
            if (
                self._motion_generation != motion_generation
                or self._homing_generation is not None
            ):
                raise FanucSafetyError(
                    "servo-mode entry was cancelled by a stop or newer motion"
                )
            state = self._states.read(max_age_s=self.state_timeout_s)
            problem = self._health_error(require_run=True)
            if problem:
                raise FanucSafetyError(
                    f"FANUC stream stack changed before servo-mode entry: {problem}"
                )
            self._last_command = state.positions.copy()
            self._last_requested_command = state.positions.copy()
            self._last_send_at = None
            # Arm the source watchdog only after the first accepted setpoint;
            # first-call JAX compilation can legitimately take several seconds.
            self._watchdog_armed = False
            self._servo_mode = True
            self._stop_published = False

    def send_joint_positions(self, q: np.ndarray) -> np.ndarray:
        with self._command_lock:
            self._require_connected()
            if not self._servo_mode:
                raise RuntimeError("enter_servo_mode() must be called before streaming")
            try:
                state = self._states.read(max_age_s=self.state_timeout_s)
                problem = self._health_error(require_run=True)
                if problem:
                    raise FanucSafetyError(problem)
                requested_reference = (
                    self._last_requested_command
                    if self._last_requested_command is not None
                    else state.positions
                )
                requested = self._safety.validate_step(q, requested_reference)

                # ws_fanuc treats a one-point trajectory as an immediate hold,
                # so its plan-time speed_scale does not affect that message.
                # Enforce a conservative, scale-aware velocity ceiling here.
                # The elapsed-time factor prevents callers from bypassing the
                # per-cycle bound by invoking this method faster than 30 Hz.
                applied_reference = (
                    self._last_command
                    if self._last_command is not None
                    else state.positions
                )
                now = self._clock()
                if self._last_send_at is None:
                    elapsed_fraction = 1.0
                else:
                    elapsed = now - self._last_send_at
                    if elapsed < 0:
                        raise FanucSafetyError("monotonic clock moved backwards while servoing")
                    elapsed_fraction = min(elapsed / self.command_duration_s, 1.0)
                speed_scale = self._fresh_speed_scale(now)
                allowed_step = (
                    self._safety.max_step_rad * speed_scale * elapsed_fraction
                )
                target = applied_reference + np.clip(
                    requested - applied_reference, -allowed_step, allowed_step
                )
                target = self._safety.validate_target(target)
                target = self._validate_target_lead(target)
                message = self._make_trajectory([target], [self.command_duration_s])
                # Recheck after allocation and immediately before the side effect.
                self._states.read(max_age_s=self.state_timeout_s)
                problem = self._health_error(require_run=True)
                if problem:
                    raise FanucSafetyError(problem)
                self._publish(message)
            except Exception as exc:
                self._latch_fault_locked(f"command rejected: {exc}")
                raise
            self._last_requested_command = requested.copy()
            self._last_command = target.copy()
            self._last_send_at = self._clock()
            self._watchdog_armed = True
            self._stop_published = False
            return target.copy()

    def _hold_position(self) -> np.ndarray | None:
        """Return only a fresh executor command, never an old measured pose."""

        now = self._clock()
        with self._condition:
            stream = self._stream
        if (
            stream is None
            or stream.mode != STREAM_MODE_RUN
            or stream.command_position is None
            or now - stream.received_at < 0
            or now - stream.received_at > self.status_timeout_s
        ):
            return None
        return self._safety.hold_target(stream.command_position)

    def _publish_hold_locked(self) -> bool:
        hold = self._hold_position()
        if hold is None or self._publisher is None or self._trajectory_type is None:
            return False
        self._publish(self._make_trajectory([hold], [self.command_duration_s]))
        self._last_command = hold.copy()
        self._stop_published = True
        return True

    def _latch_fault_locked(self, reason: str) -> None:
        if self._fault_reason is not None:
            return
        # Latch before attempting any ROS side effect. A failed hold must never
        # leave this object accepting more motion commands.
        self._fault_reason = str(reason)
        self._motion_generation += 1
        self._homing_generation = None
        self._servo_mode = False
        self._watchdog_armed = False
        try:
            if self._connected and not self._stop_published:
                self._publish_hold_locked()
        except Exception as hold_error:
            self._fault_reason += f"; setpoint hold also failed: {hold_error}"
        with self._condition:
            self._condition.notify_all()

    def _latch_fault(self, reason: str) -> None:
        with self._command_lock:
            self._latch_fault_locked(reason)

    def _watchdog_check(self, now: float | None = None) -> bool:
        current = self._clock() if now is None else float(now)
        with self._command_lock:
            if (
                not self._connected
                or not self._watchdog_armed
                or self._last_send_at is None
                or self._fault_reason is not None
                or current - self._last_send_at <= self.watchdog_timeout_s
            ):
                return False
            self._latch_fault_locked(
                f"setpoint watchdog expired after {current - self._last_send_at:.3f}s"
            )
            return True

    def _watchdog_loop(self) -> None:
        interval = min(0.05, max(0.01, self.watchdog_timeout_s / 4.0))
        while not self._watchdog_stop.wait(interval):
            try:
                self._watchdog_check()
            except BaseException as exc:  # pragma: no cover - ROS runtime failure
                self._latch_fault(f"watchdog failed: {exc}")

    def stop(self) -> None:
        with self._command_lock:
            # Cancel a rolling home even if a hold was already published.
            self._motion_generation += 1
            self._homing_generation = None
            self._servo_mode = False
            self._watchdog_armed = False
            if not self._connected or self._stop_published:
                return
            self._publish_hold_locked()
            # If no fresh executor target existed, leave _stop_published false.
            # A later stop()/disconnect() can then retry after status recovers.

    def disconnect(self) -> None:
        with self._lifecycle_lock:
            self._disconnect_locked()

    def _disconnect_locked(self) -> None:
        if not self._connected and not self._ros_resources_remain():
            return
        stop_error: BaseException | None = None
        try:
            self.stop()
        except BaseException as exc:  # cleanup must still continue
            stop_error = exc
        cleanup_error: BaseException | None = None
        try:
            self._cleanup_ros()
        except BaseException as exc:
            cleanup_error = exc
        if stop_error is not None:
            if cleanup_error is not None:
                stop_error.add_note(f"ROS cleanup also failed: {cleanup_error}")
            raise stop_error
        if cleanup_error is not None:
            raise cleanup_error

    def _cleanup_ros(self) -> None:
        with self._command_lock:
            self._motion_generation += 1
            self._homing_generation = None
            self._connected = False
            self._servo_mode = False
            self._watchdog_armed = False
            self._last_requested_command = None
        self._watchdog_stop.set()
        if (
            self._watchdog_thread is not None
            and self._watchdog_thread is not threading.current_thread()
            and self._watchdog_thread.ident is not None
        ):
            self._watchdog_thread.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
        if (
            self._executor_thread is not None
            and self._executor_thread is not threading.current_thread()
            and self._executor_thread.ident is not None
        ):
            self._executor_thread.join(timeout=2.0)
        # A context shutdown is the fallback wake-up if executor.shutdown did
        # not release spin(). Do it before destroying resources used by a live thread.
        if self._context is not None:
            try:
                self._context.try_shutdown()
            except Exception:
                pass
        if (
            self._executor_thread is not None
            and self._executor_thread is not threading.current_thread()
            and self._executor_thread.ident is not None
            and self._executor_thread.is_alive()
        ):
            self._executor_thread.join(timeout=2.0)
        alive = []
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            alive.append("watchdog thread did not stop")
        if self._executor_thread is not None and self._executor_thread.is_alive():
            alive.append("ROS executor thread did not stop")
        if alive:
            # Do not destroy handles that a surviving thread may still be using;
            # retain them so a later disconnect() can retry the joins safely.
            raise FanucError("; ".join(alive))
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        self._publisher = None
        self._subscriptions = []
        self._node = None
        self._executor = None
        self._executor_thread = None
        self._watchdog_thread = None
        self._context = None
        self._parameter_client = None
        self._parameter_verified = False


__all__ = [
    "CRX10IA_L_HOME",
    "CRX10IA_L_JOINT_NAMES",
    "CRX10IA_L_LOWER_LIMITS",
    "CRX10IA_L_UPPER_LIMITS",
    "FanucCommandSafety",
    "FanucError",
    "FanucROS2Follower",
    "FanucSafetyError",
    "JointStateBuffer",
    "JointStateSnapshot",
    "JointStateUnavailable",
    "min_jerk_profile",
]
