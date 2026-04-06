from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from capture import FPSCounter, LatestFrameStore, CaptureWorker
from config import build_default_config
from detector import DetectionWorker, DetectionResult, TutorialImageCache
from display_renderer import DisplayRenderer, RenderConfig
from output_manager import OutputManager, OutputManagerConfig
from second_screen_window import SecondMonitorBackend, SecondScreenConfig
from serial_5inch_backend import Serial5InchBackend, Serial5InchConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


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
        self.root.minsize(1000, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.output_backend = None
        self.output_manager: Optional[OutputManager] = None

        self._build_layout()
        self._rebuild_output_backend()

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

        ttk.Label(right, text="输出控制 / 5寸屏调试").pack(anchor="w")
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(4, 8))

        ttk.Label(controls, text="输出模式").grid(row=0, column=0, sticky="w")
        self.output_mode_var = tk.StringVar(value=self.cfg.output_mode)
        self.output_mode_combo = ttk.Combobox(
            controls,
            textvariable=self.output_mode_var,
            values=("second_monitor", "serial_5inch"),
            state="readonly",
            width=20,
        )
        self.output_mode_combo.grid(row=0, column=1, sticky="ew", padx=4)
        self.output_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_change())

        ttk.Label(controls, text="COM").grid(row=1, column=0, sticky="w")
        self.serial_port_var = tk.StringVar(value=self.cfg.serial_port)
        ttk.Entry(controls, textvariable=self.serial_port_var, width=20).grid(row=1, column=1, sticky="ew", padx=4)

        ttk.Label(controls, text="Baud").grid(row=2, column=0, sticky="w")
        self.baud_var = tk.StringVar(value=str(self.cfg.baud_rate))
        ttk.Entry(controls, textvariable=self.baud_var, width=20).grid(row=2, column=1, sticky="ew", padx=4)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Connect", command=self._connect_serial).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(btns, text="Reconnect", command=self._reconnect_serial).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(btns, text="Send Idle", command=self._send_idle).grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(btns, text="Send Test Pattern", command=self._send_test).grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        ttk.Button(btns, text="Send Current Lineup", command=self._send_current_lineup).grid(
            row=2, column=0, padx=2, pady=2, sticky="ew"
        )
        ttk.Button(btns, text="Resend Current", command=self._resend_current).grid(row=2, column=1, padx=2, pady=2, sticky="ew")

        self.detect_var = tk.StringVar(value="area: unknown | point: unknown")
        self.output_status_var = tk.StringVar(value="backend: unknown")
        self.serial_status_var = tk.StringVar(value="serial: disconnected")

        ttk.Label(right, textvariable=self.detect_var, font=("Consolas", 12)).pack(anchor="w", pady=(6, 2))
        ttk.Label(right, textvariable=self.output_status_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.serial_status_var).pack(anchor="w")

    def _build_output_manager(self) -> OutputManager:
        mode = self.output_mode_var.get().strip() or "second_monitor"
        self.cfg.output_mode = mode

        if mode == "serial_5inch":
            serial_cfg = Serial5InchConfig(
                serial_port=self.serial_port_var.get().strip(),
                baud_rate=int(self.baud_var.get().strip() or self.cfg.baud_rate),
                timeout=self.cfg.serial_timeout,
                retry_count=self.cfg.serial_retry_count,
                screen_width=self.cfg.screen_width,
                screen_height=self.cfg.screen_height,
                send_only_on_change=self.cfg.send_only_on_change,
                image_format=self.cfg.image_format,
                handshake_enabled=self.cfg.handshake_enabled,
                connect_on_startup=self.cfg.connect_on_startup,
                connect_retry_sec=self.cfg.serial_connect_retry_sec,
            )
            backend = Serial5InchBackend(serial_cfg)
        else:
            second_cfg = SecondScreenConfig(
                enabled=self.cfg.second_monitor_enabled,
                width=self.cfg.screen_width,
                height=self.cfg.screen_height,
            )
            backend = SecondMonitorBackend(self.root, second_cfg)

        renderer = DisplayRenderer(RenderConfig(width=self.cfg.screen_width, height=self.cfg.screen_height))
        manager = OutputManager(
            backend=backend,
            renderer=renderer,
            tutorial_cache=self.tutorial_cache,
            cfg=OutputManagerConfig(
                stable_frames_required=self.cfg.stable_frames_required,
                switch_cooldown_sec=self.cfg.switch_cooldown_sec,
                idle_resend_sec=self.cfg.idle_resend_sec,
                map_name=self.cfg.map_name,
                team_name=self.cfg.team_name,
            ),
        )
        return manager

    def _rebuild_output_backend(self) -> None:
        if self.output_manager is not None:
            self.output_manager.close()

        self.output_manager = self._build_output_manager()
        self.output_manager.initialize()

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

        if self.output_manager is not None:
            self.output_manager.update_from_detection(result)
            runtime = self.output_manager.runtime_status()
            self.output_status_var.set(
                f"backend: {runtime.get('backend')} | last_hash: {runtime.get('last_hash', '')[:8]} | "
                f"last: {runtime.get('last_send_result', '-') }"
            )
            self.serial_status_var.set(
                f"serial_status: {runtime.get('status')} | port: {runtime.get('port', '-') } | "
                f"send_events: {runtime.get('send_events', 0)} | err: {runtime.get('last_error', '')[:36]}"
            )

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

    def _on_mode_change(self) -> None:
        self._rebuild_output_backend()

    def _connect_serial(self) -> None:
        if self.output_manager is None:
            return
        backend = self.output_manager.backend
        if isinstance(backend, Serial5InchBackend):
            backend.cfg.serial_port = self.serial_port_var.get().strip()
            backend.cfg.baud_rate = int(self.baud_var.get().strip() or backend.cfg.baud_rate)
            backend.connect()
        else:
            self.status_var.set("当前不是 serial_5inch 模式")

    def _reconnect_serial(self) -> None:
        if self.output_manager is None:
            return
        backend = self.output_manager.backend
        if isinstance(backend, Serial5InchBackend):
            backend.reconnect()

    def _send_idle(self) -> None:
        if self.output_manager is not None:
            self.output_manager.show_idle(status="MANUAL_IDLE")

    def _send_test(self) -> None:
        if self.output_manager is not None:
            self.output_manager.send_test_pattern()

    def _send_current_lineup(self) -> None:
        if self.output_manager is None:
            return
        result = self.det_worker.get_latest_result()
        self.output_manager.update_from_detection(result)

    def _resend_current(self) -> None:
        if self.output_manager is not None:
            self.output_manager.resend_current()

    def _on_close(self) -> None:
        LOGGER.info("Stopping app")
        self.status_var.set("状态: 正在停止")
        self.capture_worker.stop()
        self.det_worker.stop()
        if self.output_manager is not None:
            self.output_manager.close()
        self.root.destroy()


if __name__ == "__main__":
    App().run()
