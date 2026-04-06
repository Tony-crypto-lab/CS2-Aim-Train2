from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for CS2 visual assistant."""

    window_width: int = 1200
    window_height: int = 800
    title: str = "CS2 Vision Trainer Console"

    monitor_index: int = 1
    capture_target_fps: int = 120
    ui_preview_fps: int = 8
    detection_idle_sleep_ms: int = 1

    minimap_roi: tuple[float, float, float, float] = (0.76, 0.73, 0.23, 0.25)
    center_roi: tuple[float, float, float, float] = (0.35, 0.28, 0.30, 0.40)

    full_preview_size: tuple[int, int] = (420, 236)
    minimap_preview_size: tuple[int, int] = (256, 256)
    center_preview_size: tuple[int, int] = (320, 240)

    assets_dir: Path = Path("assets")
    tutorial_dir: Path = Path("assets/tutorial")


def clamp_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = roi
    x = max(0.0, min(x, 1.0))
    y = max(0.0, min(y, 1.0))
    w = max(0.01, min(w, 1.0 - x))
    h = max(0.01, min(h, 1.0 - y))
    return x, y, w, h


def build_default_config() -> AppConfig:
    cfg = AppConfig()
    cfg.minimap_roi = clamp_roi(cfg.minimap_roi)
    cfg.center_roi = clamp_roi(cfg.center_roi)
    return cfg
