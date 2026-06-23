"""Intel RealSense D435 camera via ``pyrealsense2`` (imported lazily).

On Apple Silicon install the community wheel ``pyrealsense2-macosx`` (the import
name is still ``pyrealsense2``); on Linux x86_64 use the official ``pyrealsense2``.
"""

from __future__ import annotations

import numpy as np

from .base import CameraFrame, CameraInterface


def list_realsense_devices() -> list[dict]:
    """Return ``[{serial, name, firmware}, ...]`` for connected RealSense devices."""
    import pyrealsense2 as rs

    out = []
    for dev in rs.context().query_devices():
        out.append(
            {
                "serial": dev.get_info(rs.camera_info.serial_number),
                "name": dev.get_info(rs.camera_info.name),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
        )
    return out


class RealSenseCamera(CameraInterface):
    def __init__(
        self,
        name: str,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
        align_depth_to_color: bool = True,
    ) -> None:
        self.name = name
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.enable_depth = bool(enable_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self._pipeline = None
        self._align = None

    def start(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        # rgb8 so frames come back already as RGB (no channel swap needed).
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline = pipeline
        self._pipeline.start(config)
        if self.enable_depth and self.align_depth_to_color:
            self._align = rs.align(rs.stream.color)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None

    def read(self) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError(f"RealSenseCamera({self.name!r}) not started.")
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())  # (H, W, 3) RGB
        depth = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())  # (H, W) uint16
        ts = frames.get_timestamp() / 1000.0  # ms -> s
        return CameraFrame(color=color, depth=depth, timestamp=ts)
