from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mss import mss


@dataclass(slots=True)
class CaptureFrame:
    frame_bgr: np.ndarray
    timestamp: float
    sequence: int


class FPSCounter:
    def __init__(self, window_sec: float = 1.0) -> None:
        self.window_sec = window_sec
        self._count = 0
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._count += 1
        delta = now - self._last
        if delta >= self.window_sec:
            self._fps = self._count / delta
            self._count = 0
            self._last = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class LatestFrameStore:
    """Keep only newest frame to avoid queue buildup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[CaptureFrame] = None

    def set(self, frame: CaptureFrame) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[CaptureFrame]:
        with self._lock:
            return self._frame


class CaptureWorker:
    def __init__(self, monitor_index: int, target_fps: int, frame_store: LatestFrameStore) -> None:
        self.monitor_index = monitor_index
        self.target_fps = max(1, target_fps)
        self.frame_store = frame_store
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fps_counter = FPSCounter(window_sec=1.0)
        self.capture_fps = 0.0
        self.monitor_rect: Optional[dict] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        seq = 0
        frame_interval = 1.0 / self.target_fps
        with mss() as sct:
            monitors = sct.monitors
            idx = self.monitor_index if 1 <= self.monitor_index < len(monitors) else 1
            monitor = monitors[idx]
            self.monitor_rect = {
                "left": int(monitor["left"]),
                "top": int(monitor["top"]),
                "width": int(monitor["width"]),
                "height": int(monitor["height"]),
            }

            while not self._stop.is_set():
                t0 = time.perf_counter()
                grab = sct.grab(self.monitor_rect)
                frame_bgr = np.asarray(grab)[:, :, :3]

                seq += 1
                self.frame_store.set(
                    CaptureFrame(frame_bgr=frame_bgr, timestamp=time.time(), sequence=seq)
                )
                self.capture_fps = self._fps_counter.tick()

                elapsed = time.perf_counter() - t0
                remaining = frame_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
