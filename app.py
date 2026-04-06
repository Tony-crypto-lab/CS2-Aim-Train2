from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from capture import FPSCounter, LatestFrameStore, CaptureWorker
from config import build_default_config
from detector import DetectionWorker, DetectionResult, TutorialImageCache


class App:
    def __init__(self) -> None:
        self.cfg = build_default_config()
        self.frame_store = LatestFrameStore()
        self.capture_worker = CaptureWorker(
            monitor_index=self.cfg.monitor_index,
            target_fps=self.cfg.capture_target_fps,
            frame_store=self.frame_store,
        )
        self.det_worker = DetectionWorker(
            frame_store=self.frame_store,
            minimap_roi_rel=self.cfg.minimap_roi,
            center_roi_rel=self.cfg.center_roi,
            idle_sleep_ms=self.cfg.detection_idle_sleep_ms,
        )

        self.tutorial_cache = TutorialImageCache(self.cfg.tutorial_dir)
        self._last_tutorial_key: Optional[str] = None

        self.ui_fps_counter = FPSCounter(window_sec=1.0)
        self.ui_fps = 0.0

        self.root = tk.Tk()
        self.root.title(self.cfg.title)
        self.root.geometry(f"{self.cfg.window_width}x{self.cfg.window_height}")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="状态: 初始化")
        self.fps_var = tk.StringVar(value="capture: 0.0 | detection: 0.0 | ui: 0.0")
        self.roi_var = tk.StringVar(value="ROI: minimap=(0,0,0,0), center=(0,0,0,0)")

        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(top, textvariable=self.fps_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(top, textvariable=self.roi_var).pack(side=tk.LEFT)

        panel = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        panel.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(panel, padding=8)
        right = ttk.Frame(panel, padding=8)
        panel.add(left, weight=3)
        panel.add(right, weight=2)

        ttk.Label(left, text="整屏缩略预览（确认抓取完整显示器）").pack(anchor="w")
        self.full_preview_label = ttk.Label(left)
        self.full_preview_label.pack(anchor="w", pady=(4, 10))

        roi_frame = ttk.Frame(left)
        roi_frame.pack(fill="x")

        ttk.Label(roi_frame, text="Minimap ROI").grid(row=0, column=0, sticky="w")
        ttk.Label(roi_frame, text="Center ROI").grid(row=0, column=1, sticky="w")

        self.minimap_preview_label = ttk.Label(roi_frame)
        self.minimap_preview_label.grid(row=1, column=0, padx=(0, 16), pady=4)
        self.center_preview_label = ttk.Label(roi_frame)
        self.center_preview_label.grid(row=1, column=1, pady=4)

        ttk.Label(right, text="副屏教学内容（缓存+变化更新）").pack(anchor="w")
        self.tutorial_label = ttk.Label(right)
        self.tutorial_label.pack(anchor="w", pady=(4, 10))

        self.detect_var = tk.StringVar(value="area: unknown | point: unknown")
        ttk.Label(right, textvariable=self.detect_var, font=("Consolas", 12)).pack(anchor="w")

    def run(self) -> None:
        self.capture_worker.start()
        self.det_worker.start()
        self.status_var.set("状态: 运行中")
        self._schedule_ui_refresh()
        self.root.mainloop()

    def _schedule_ui_refresh(self) -> None:
        interval_ms = int(1000 / max(1, self.cfg.ui_preview_fps))
        self._refresh_ui()
        self.root.after(interval_ms, self._schedule_ui_refresh)

    def _refresh_ui(self) -> None:
        capture = self.frame_store.get()
        if capture is not None:
            self._update_full_preview(capture.frame_bgr)

        result = self.det_worker.get_latest_result()
        if result is not None:
            self._update_roi_preview(result)
            self._update_tutorial_if_changed(result)

        self.ui_fps = self.ui_fps_counter.tick()
        self.fps_var.set(
            f"capture: {self.capture_worker.capture_fps:.1f} | "
            f"detection: {self.det_worker.detection_fps:.1f} | ui: {self.ui_fps:.1f}"
        )

    def _update_full_preview(self, frame_bgr):
        img = cv2.resize(frame_bgr, self.cfg.full_preview_size, interpolation=cv2.INTER_AREA)
        self._set_label_image(self.full_preview_label, img)

    def _update_roi_preview(self, result: DetectionResult) -> None:
        if result.minimap_roi.size:
            mm = cv2.resize(result.minimap_roi, self.cfg.minimap_preview_size, interpolation=cv2.INTER_AREA)
            self._set_label_image(self.minimap_preview_label, mm)

        if result.center_roi.size:
            center = cv2.resize(result.center_roi, self.cfg.center_preview_size, interpolation=cv2.INTER_AREA)
            self._set_label_image(self.center_preview_label, center)

        self.detect_var.set(
            f"area: {result.inferred_area} ({result.minimap_conf:.2f}) | "
            f"point: {result.inferred_point} ({result.center_conf:.2f})"
        )

        capture = self.frame_store.get()
        if capture is not None:
            h, w = capture.frame_bgr.shape[:2]
            mm_roi = self._abs_roi(self.cfg.minimap_roi, w, h)
            c_roi = self._abs_roi(self.cfg.center_roi, w, h)
            self.roi_var.set(f"ROI: minimap={mm_roi}, center={c_roi}")

    def _update_tutorial_if_changed(self, result: DetectionResult) -> None:
        key = f"{result.inferred_area}_{result.inferred_point}"
        if key == self._last_tutorial_key:
            return

        img = self.tutorial_cache.get(key)
        if img is None:
            img = self._build_fallback_tutorial(key)

        preview = cv2.resize(img, (360, 220), interpolation=cv2.INTER_AREA)
        self._set_label_image(self.tutorial_label, preview)
        self._last_tutorial_key = key

    @staticmethod
    def _abs_roi(rel_roi: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
        x = int(rel_roi[0] * w)
        y = int(rel_roi[1] * h)
        rw = int(rel_roi[2] * w)
        rh = int(rel_roi[3] * h)
        return x, y, rw, rh

    @staticmethod
    def _build_fallback_tutorial(key: str):
        canvas = cv2.imread(str(Path(__file__).parent / "assets" / "fallback.png"))
        if canvas is None:
            canvas = 255 * (cv2.UMat(220, 360, cv2.CV_8UC3).get())
        cv2.putText(canvas, "No tutorial asset", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(canvas, key, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 80, 150), 2)
        return canvas

    @staticmethod
    def _set_label_image(label: ttk.Label, bgr_img) -> None:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        tk_im = ImageTk.PhotoImage(im)
        label.configure(image=tk_im)
        label.image = tk_im

    def _on_close(self) -> None:
        self.status_var.set("状态: 正在停止")
        self.capture_worker.stop()
        self.det_worker.stop()
        self.root.destroy()


if __name__ == "__main__":
    App().run()
