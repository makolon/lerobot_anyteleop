"""Franka Hand (Franka Emika gripper) via ``panda_py.libfranka.Gripper``.

Franka gripper commands are **slow and blocking** (homing ~1-2 s; move/grasp take
hundreds of ms) and must NOT be called at the arm's control rate. So commands run
on a background worker thread with *latest-wins* semantics: ``set_normalized``
just updates the target; the worker dispatches it when the previous command
finished. Fully-closing commands use ``grasp`` (applies force) rather than
``move`` so objects are actually held.

Width is in meters (max ~0.08 m), speed m/s, force N.
"""

from __future__ import annotations

import threading

from .base import GripperInterface


class FrankaHand(GripperInterface):
    deadband = 0.05

    def __init__(
        self,
        ip: str,
        *,
        speed: float = 0.1,
        force: float = 40.0,
        grasp_below: float = 0.15,  # normalized; below this, grasp (with force)
    ) -> None:
        self.ip = ip
        self.speed = float(speed)
        self.force = float(force)
        self.grasp_below = float(grasp_below)
        self._gripper = None
        self._max_width = 0.08
        self._target_v: float | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        from panda_py import libfranka  # type: ignore

        self._gripper = libfranka.Gripper(self.ip)
        self._gripper.homing()  # blocking; estimates max_width
        self._max_width = float(self._gripper.read_once().max_width)
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="franka-gripper", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception:
                pass
        self._gripper = None

    @property
    def is_connected(self) -> bool:
        return self._gripper is not None

    def set_normalized(self, value: float) -> None:
        with self._lock:
            self._target_v = self._clamp01(value)
        self._wake.set()

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                v = self._target_v
            if v is None or self._gripper is None:
                continue
            try:
                if v < self.grasp_below:
                    # close & grasp; eps_outer = max so any closure counts as a grasp
                    self._gripper.grasp(0.0, self.speed, self.force,
                                        epsilon_inner=self._max_width, epsilon_outer=self._max_width)
                else:
                    self._gripper.move(v * self._max_width, self.speed)
            except Exception:
                pass  # don't let a transient gripper error kill the worker
