from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RenderConfig:
    width: int = 800
    height: int = 480


class DisplayRenderer:
    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg

    def render_idle_screen(self, map_name: str, region: str, status: str = "WAITING LINEUP") -> np.ndarray:
        canvas = self._base_canvas()
        self._draw_title(canvas, "CS2 HUD READY")
        self._draw_kv(canvas, 64, "CURRENT MAP", map_name)
        self._draw_kv(canvas, 104, "CURRENT REGION", region)
        self._draw_kv(canvas, 144, "STATUS", status)
        cv2.putText(canvas, "NO LINEUP", (38, 228), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (60, 180, 240), 3)
        self._draw_preview_placeholder(canvas, "WAITING FOR MATCH")
        return canvas

    def render_test_pattern(self) -> np.ndarray:
        canvas = self._base_canvas()
        self._draw_title(canvas, "SERIAL 5INCH TEST")
        for i, color in enumerate([(255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 220, 80)]):
            cv2.rectangle(canvas, (40 + i * 90, 120), (110 + i * 90, 190), color, -1)
        self._draw_kv(canvas, 216, "MESSAGE", "LINK TEST OK")
        self._draw_preview_placeholder(canvas, "TEST IMAGE")
        return canvas

    def render_lineup_screen(self, lineup_info: dict[str, Any], preview_image: Optional[np.ndarray]) -> np.ndarray:
        canvas = self._base_canvas()
        self._draw_title(canvas, "CS2 LINEUP")

        self._draw_kv(canvas, 56, "MAP", lineup_info.get("map", "UNKNOWN"))
        self._draw_kv(canvas, 94, "TEAM", lineup_info.get("team", "UNKNOWN"))
        self._draw_kv(canvas, 132, "AREA", lineup_info.get("area", "UNKNOWN"))
        self._draw_kv(canvas, 170, "POINT", lineup_info.get("point", "UNKNOWN"))
        self._draw_kv(canvas, 208, "THROW", lineup_info.get("throw_type", "UNKNOWN"))
        conf = lineup_info.get("confidence", 0.0)
        self._draw_kv(canvas, 246, "CONF", f"{conf:.2f}")

        self._blit_preview(canvas, preview_image)
        return canvas

    def load_preview_image(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        if img is None:
            LOGGER.warning("Cannot load preview image: %s", path)
            return None
        return img

    def _base_canvas(self) -> np.ndarray:
        canvas = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 24)
        cv2.rectangle(canvas, (0, 0), (self.cfg.width - 1, self.cfg.height - 1), (70, 80, 130), 2)
        return canvas

    @staticmethod
    def _draw_title(canvas: np.ndarray, title: str) -> None:
        cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 240), 2)

    @staticmethod
    def _draw_kv(canvas: np.ndarray, y: int, k: str, v: str) -> None:
        cv2.putText(canvas, f"{k: <13}: {v}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (198, 210, 220), 2)

    def _blit_preview(self, canvas: np.ndarray, preview_image: Optional[np.ndarray]) -> None:
        px, py, pw, ph = 430, 62, self.cfg.width - 450, self.cfg.height - 90
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (50, 70, 100), 2)

        if preview_image is None or preview_image.size == 0:
            self._draw_preview_placeholder(canvas, "PREVIEW MISSING")
            return

        fitted = self._fit_letterbox(preview_image, pw - 8, ph - 8)
        fh, fw = fitted.shape[:2]
        ox = px + (pw - fw) // 2
        oy = py + (ph - fh) // 2
        canvas[oy : oy + fh, ox : ox + fw] = fitted

    def _draw_preview_placeholder(self, canvas: np.ndarray, text: str) -> None:
        px, py, pw, ph = 430, 62, self.cfg.width - 450, self.cfg.height - 90
        cv2.rectangle(canvas, (px + 8, py + 8), (px + pw - 8, py + ph - 8), (26, 26, 34), -1)
        cv2.putText(canvas, text, (px + 20, py + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 140, 180), 2)

    @staticmethod
    def _fit_letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (max(1, nw), max(1, nh)), interpolation=cv2.INTER_AREA)
