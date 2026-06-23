"""Camera interface and frame container."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraFrame:
    #: color image, ``(H, W, 3)`` uint8, **RGB**.
    color: np.ndarray
    #: optional depth image, ``(H, W)`` uint16 (raw device units), or ``None``.
    depth: np.ndarray | None
    #: capture timestamp (seconds).
    timestamp: float


class CameraInterface(abc.ABC):
    name: str
    width: int
    height: int
    enable_depth: bool

    @abc.abstractmethod
    def start(self) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    @abc.abstractmethod
    def read(self) -> CameraFrame:
        ...
