"""Manage a set of cameras as one unit."""

from __future__ import annotations

from .base import CameraFrame, CameraInterface


class MultiCameraManager:
    """Start/stop and read a group of cameras together.

    Reads are sequential. For many high-rate cameras consider running each in its
    own thread; for recording at ~30 Hz the sequential path is sufficient.
    """

    def __init__(self, cameras: list[CameraInterface]) -> None:
        self.cameras = list(cameras)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cameras]

    def start(self) -> None:
        for cam in self.cameras:
            cam.start()

    def stop(self) -> None:
        for cam in self.cameras:
            try:
                cam.stop()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def read(self) -> dict[str, CameraFrame]:
        return {cam.name: cam.read() for cam in self.cameras}

    def __len__(self) -> int:
        return len(self.cameras)

    def __enter__(self) -> "MultiCameraManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
