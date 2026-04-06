from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from capture import FPSCounter, LatestFrameStore


@dataclass(slots=True)
class DetectionResult:
    frame_sequence: int
    minimap_roi: np.ndarray
    center_roi: np.ndarray
    inferred_area: str
    inferred_point: str
    minimap_conf: float
    center_conf: float


class DetectionWorker:
    def __init__(
        self,
        frame_store: LatestFrameStore,
        minimap_roi_rel: tuple[float, float, float, float],
        center_roi_rel: tuple[float, float, float, float],
        idle_sleep_ms: int = 1,
    ) -> None:
        self.frame_store = frame_store
        self.minimap_roi_rel = minimap_roi_rel
        self.center_roi_rel = center_roi_rel
        self.idle_sleep_sec = max(0.001, idle_sleep_ms / 1000)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._result: Optional[DetectionResult] = None
        self._last_seen_seq = 0

        self._fps_counter = FPSCounter(window_sec=1.0)
        self.detection_fps = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="det-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_latest_result(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._result

    def _loop(self) -> None:
        while not self._stop.is_set():
            capture = self.frame_store.get()
            if capture is None or capture.sequence == self._last_seen_seq:
                time.sleep(self.idle_sleep_sec)
                continue

            frame = capture.frame_bgr
            h, w = frame.shape[:2]
            mm = self._crop_roi(frame, self.minimap_roi_rel, w, h)
            center = self._crop_roi(frame, self.center_roi_rel, w, h)

            area, point, mm_conf, center_conf = self._infer_area_and_point(mm, center)
            result = DetectionResult(
                frame_sequence=capture.sequence,
                minimap_roi=mm,
                center_roi=center,
                inferred_area=area,
                inferred_point=point,
                minimap_conf=mm_conf,
                center_conf=center_conf,
            )
            with self._lock:
                self._result = result

            self._last_seen_seq = capture.sequence
            self.detection_fps = self._fps_counter.tick()

    @staticmethod
    def _crop_roi(
        frame: np.ndarray,
        rel_roi: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        x_rel, y_rel, w_rel, h_rel = rel_roi
        x = int(width * x_rel)
        y = int(height * y_rel)
        rw = int(width * w_rel)
        rh = int(height * h_rel)
        x2 = min(width, x + max(4, rw))
        y2 = min(height, y + max(4, rh))
        return frame[y:y2, x:x2]

    @staticmethod
    def _infer_area_and_point(minimap: np.ndarray, center: np.ndarray) -> tuple[str, str, float, float]:
        if minimap.size == 0 or center.size == 0:
            return "unknown", "unknown", 0.0, 0.0

        mm_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

        mm_edge = cv2.Canny(mm_gray, 60, 130)
        center_edge = cv2.Canny(center_gray, 80, 150)

        mm_ratio = float(np.count_nonzero(mm_edge)) / float(mm_edge.size)
        center_ratio = float(np.count_nonzero(center_edge)) / float(center_edge.size)

        if mm_ratio < 0.045:
            area = "open_area"
        elif mm_ratio < 0.11:
            area = "corridor"
        else:
            area = "dense_area"

        if center_ratio < 0.05:
            point = "holding_angle"
        elif center_ratio < 0.12:
            point = "default_peek"
        else:
            point = "engagement"

        return area, point, min(1.0, mm_ratio * 8.0), min(1.0, center_ratio * 6.0)


class TutorialImageCache:
    def __init__(self, tutorial_dir: Path) -> None:
        self.tutorial_dir = tutorial_dir
        self._cache: dict[str, np.ndarray] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            return self._cache[key]

        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = self.tutorial_dir / f"{key}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    self._cache[key] = img
                    return img
        return None
