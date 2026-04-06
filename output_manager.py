from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from detector import DetectionResult, TutorialImageCache
from display_renderer import DisplayRenderer
from output_backend import OutputBackend


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OutputManagerConfig:
    stable_frames_required: int = 3
    switch_cooldown_sec: float = 0.8
    map_name: str = "DE_DUST2"
    team_name: str = "T"


class OutputManager:
    def __init__(
        self,
        backend: OutputBackend,
        renderer: DisplayRenderer,
        tutorial_cache: TutorialImageCache,
        cfg: OutputManagerConfig,
    ) -> None:
        self.backend = backend
        self.renderer = renderer
        self.tutorial_cache = tutorial_cache
        self.cfg = cfg

        self._candidate_key = ""
        self._candidate_count = 0
        self._displayed_key = ""
        self._last_switch_time = 0.0

        self._last_lineup_info: Optional[dict] = None

    def initialize(self) -> bool:
        ok = self.backend.initialize()
        self.show_idle(status="INIT")
        return ok

    def update_from_detection(self, result: Optional[DetectionResult]) -> None:
        if result is None:
            self.show_idle(status="NO_DETECTION")
            return

        if result.inferred_area == "unknown" or result.inferred_point == "unknown":
            self.show_idle(status="NO_LINEUP")
            return

        key = f"{result.inferred_area}|{result.inferred_point}"
        if key != self._candidate_key:
            self._candidate_key = key
            self._candidate_count = 1
            return

        self._candidate_count += 1
        if self._candidate_count < self.cfg.stable_frames_required:
            return

        now = time.perf_counter()
        if key != self._displayed_key and (now - self._last_switch_time) < self.cfg.switch_cooldown_sec:
            return

        area, point = result.inferred_area, result.inferred_point
        lineup_info = {
            "map": self.cfg.map_name,
            "team": self.cfg.team_name,
            "area": area,
            "point": point,
            "throw_type": "SMOKE",
            "confidence": min(result.minimap_conf, result.center_conf),
        }

        image_key = f"{area}_{point}"
        preview = self.tutorial_cache.get(image_key)
        rendered = self.renderer.render_lineup_screen(lineup_info, preview)

        self.backend.show_lineup(lineup_info, rendered)
        self._displayed_key = key
        self._last_switch_time = now
        self._last_lineup_info = lineup_info

    def show_idle(self, status: str = "WAITING") -> None:
        image = self.renderer.render_idle_screen(self.cfg.map_name, "HUD", status)
        self.backend.show_idle(image, status)

    def send_test_pattern(self) -> None:
        image = self.renderer.render_test_pattern()
        self.backend.show_status("send_test_pattern")
        self.backend.show_idle(image, status_text="TEST")

    def resend_current(self) -> None:
        self.backend.resend_current()

    def close(self) -> None:
        self.backend.close()

    def runtime_status(self) -> dict:
        return self.backend.get_runtime_status()
