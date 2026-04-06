from __future__ import annotations

import hashlib
import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from output_backend import OutputBackend


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SecondScreenConfig:
    enabled: bool = True
    window_title: str = "CS2 Second Monitor Output"
    width: int = 800
    height: int = 480


class SecondMonitorBackend(OutputBackend):
    def __init__(self, root: tk.Tk, cfg: SecondScreenConfig) -> None:
        self.root = root
        self.cfg = cfg
        self.window: Optional[tk.Toplevel] = None
        self.image_label: Optional[ttk.Label] = None
        self.info_var: Optional[tk.StringVar] = None
        self.last_hash = ""
        self.last_image: Optional[np.ndarray] = None
        self.last_status = "not_initialized"

    def initialize(self) -> bool:
        if not self.cfg.enabled:
            self.last_status = "disabled"
            return False

        self.window = tk.Toplevel(self.root)
        self.window.title(self.cfg.window_title)
        self.window.geometry(f"{self.cfg.width}x{self.cfg.height + 40}")
        self.window.minsize(420, 260)

        self.info_var = tk.StringVar(value="Second monitor backend ready")
        ttk.Label(self.window, textvariable=self.info_var).pack(anchor="w", padx=8, pady=(6, 2))
        self.image_label = ttk.Label(self.window)
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.last_status = "ready"
        return True

    def show_idle(self, rendered_image: np.ndarray, status_text: str = "") -> None:
        self._update_image(rendered_image, status_text or "Idle")

    def show_lineup(self, lineup_info: dict[str, Any], rendered_image: np.ndarray) -> None:
        status = f"Lineup: {lineup_info.get('point', 'unknown')}"
        self._update_image(rendered_image, status)

    def show_status(self, text: str) -> None:
        if self.info_var is not None:
            self.info_var.set(text)

    def resend_current(self) -> None:
        if self.last_image is not None:
            self._display_image(self.last_image)

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
        self.last_status = "closed"

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "backend": "second_monitor",
            "status": self.last_status,
            "connected": self.window is not None,
            "last_hash": self.last_hash,
            "last_send_result": "updated" if self.last_hash else "idle",
        }

    def _update_image(self, image: np.ndarray, status: str) -> None:
        current_hash = hashlib.sha1(image.tobytes()).hexdigest()[:16]
        if current_hash == self.last_hash:
            return

        self.last_hash = current_hash
        self.last_image = image.copy()
        self._display_image(image)
        if self.info_var is not None:
            self.info_var.set(status)

    def _display_image(self, image: np.ndarray) -> None:
        if self.image_label is None:
            LOGGER.warning("SecondMonitorBackend image_label missing")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image
