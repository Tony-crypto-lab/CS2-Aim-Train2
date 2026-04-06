from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from output_backend import OutputBackend


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Serial5InchConfig:
    serial_port: str = ""
    baud_rate: int = 921600
    timeout: float = 1.0
    retry_count: int = 2
    screen_width: int = 800
    screen_height: int = 480
    send_only_on_change: bool = True
    image_format: str = "jpg"
    handshake_enabled: bool = False
    connect_on_startup: bool = False
    connect_retry_sec: float = 2.0


class Serial5InchBackend(OutputBackend):
    def __init__(self, cfg: Serial5InchConfig) -> None:
        self.cfg = cfg
        self.serial_client: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._pending_payload: Optional[bytes] = None
        self._pending_hash = ""
        self._current_payload: Optional[bytes] = None
        self._current_hash = ""

        self._last_enqueued_hash = ""
        self._next_connect_allowed = 0.0
        self._last_error_signature = ""

        self.last_send_result = "idle"
        self.last_error = ""
        self.connected = False
        self.send_events = 0

    def initialize(self) -> bool:
        self._stop.clear()
        self._thread = threading.Thread(target=self._send_loop, name="serial-5inch-sender", daemon=True)
        self._thread.start()
        if self.cfg.connect_on_startup:
            return self.connect(force=True)
        return True

    def connect(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now < self._next_connect_allowed:
            self.last_send_result = "connect_backoff"
            return False

        port = self.cfg.serial_port or self._auto_detect_port()
        if not port:
            self.connected = False
            self.last_error = "No serial port found"
            self.last_send_result = "connect_failed"
            self._next_connect_allowed = now + self.cfg.connect_retry_sec
            return False

        try:
            self.serial_client = serial.Serial(
                port=port,
                baudrate=self.cfg.baud_rate,
                timeout=self.cfg.timeout,
                write_timeout=self.cfg.timeout,
            )
            self.cfg.serial_port = port
            self.connected = True
            self.last_error = ""
            self.last_send_result = f"connected:{port}"
            self._next_connect_allowed = 0.0
            LOGGER.info("Connected serial port %s", port)
            return True
        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)
            self.last_send_result = "connect_failed"
            self._next_connect_allowed = now + self.cfg.connect_retry_sec
            self._log_connect_error_once(exc, port)
            return False

    def reconnect(self) -> bool:
        self.disconnect()
        return self.connect(force=True)

    def disconnect(self) -> None:
        if self.serial_client is not None:
            try:
                self.serial_client.close()
            except Exception:
                LOGGER.exception("Error while closing serial port")
        self.serial_client = None
        self.connected = False

    def show_idle(self, rendered_image: np.ndarray, status_text: str = "") -> None:
        self._enqueue_image(rendered_image)

    def show_lineup(self, lineup_info: dict[str, Any], rendered_image: np.ndarray) -> None:
        self._enqueue_image(rendered_image)

    def show_status(self, text: str) -> None:
        self.last_send_result = text

    def resend_current(self) -> None:
        with self._lock:
            if self._current_payload is None:
                return
            self._pending_payload = self._current_payload
            self._pending_hash = self._current_hash
            self._event.set()

    def close(self) -> None:
        self._stop.set()
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.disconnect()

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "backend": "serial_5inch",
            "status": "connected" if self.connected else "disconnected",
            "connected": self.connected,
            "port": self.cfg.serial_port,
            "last_hash": self._current_hash,
            "last_send_result": self.last_send_result,
            "last_error": self.last_error,
            "send_only_on_change": self.cfg.send_only_on_change,
            "send_events": self.send_events,
            "next_connect_in": max(0.0, self._next_connect_allowed - time.monotonic()),
        }

    def _enqueue_image(self, image: np.ndarray) -> None:
        prepared = self._prepare_image(image)
        payload, hash_value = self._encode_image(prepared)

        with self._lock:
            if self.cfg.send_only_on_change:
                if hash_value == self._current_hash:
                    self.last_send_result = "skip_same_hash"
                    return
                if hash_value == self._pending_hash:
                    self.last_send_result = "skip_pending_same_hash"
                    return
                if hash_value == self._last_enqueued_hash and not self.connected:
                    self.last_send_result = "skip_duplicate_while_disconnected"
                    return

            self._pending_payload = payload
            self._pending_hash = hash_value
            self._last_enqueued_hash = hash_value
            self._event.set()

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            self._event.wait(timeout=0.2)
            self._event.clear()
            if self._stop.is_set():
                break

            with self._lock:
                payload = self._pending_payload
                hash_value = self._pending_hash
                self._pending_payload = None
                self._pending_hash = ""

            if payload is None:
                continue

            if not self.connected and not self.connect():
                with self._lock:
                    self._pending_payload = payload
                    self._pending_hash = hash_value
                continue

            ok = self._send_payload(payload, hash_value)
            if ok:
                with self._lock:
                    self._current_payload = payload
                    self._current_hash = hash_value
                self.last_send_result = f"sent:{hash_value[:8]}"
                self.send_events += 1
            else:
                self.last_send_result = "send_failed"
                with self._lock:
                    if self._pending_payload is None:
                        self._pending_payload = payload
                        self._pending_hash = hash_value

    def _send_payload(self, payload: bytes, hash_value: str) -> bool:
        if self.serial_client is None:
            return False

        for _ in range(max(1, self.cfg.retry_count)):
            try:
                header = (
                    f"META|{self.cfg.screen_width}|{self.cfg.screen_height}|{self.cfg.image_format}|"
                    f"{len(payload)}|{hash_value}\n"
                ).encode("utf-8")
                self.serial_client.write(header)
                self.serial_client.flush()

                if self.cfg.handshake_enabled:
                    ready = self.serial_client.readline().decode("utf-8", errors="ignore").strip()
                    if ready not in ("READY", "OK"):
                        continue

                self.serial_client.write(payload)
                self.serial_client.write(b"\nEND\n")
                self.serial_client.flush()

                if self.cfg.handshake_enabled:
                    ack = self.serial_client.readline().decode("utf-8", errors="ignore").strip()
                    if ack not in ("OK", "ACK"):
                        continue
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.connected = False
                self.disconnect()
                self._next_connect_allowed = time.monotonic() + self.cfg.connect_retry_sec
                self._log_connect_error_once(exc, self.cfg.serial_port)
                return False

        return False

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tw, th = self.cfg.screen_width, self.cfg.screen_height
        scale = min(tw / max(1, w), th / max(1, h))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x = (tw - nw) // 2
        y = (th - nh) // 2
        canvas[y : y + nh, x : x + nw] = resized
        return canvas

    def _encode_image(self, image: np.ndarray) -> tuple[bytes, str]:
        fmt = self.cfg.image_format.lower()
        ext = ".jpg" if fmt in {"jpg", "jpeg"} else ".png"
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85] if ext == ".jpg" else []
        ok, enc = cv2.imencode(ext, image, encode_param)
        if not ok:
            placeholder = self._build_placeholder("ENCODE FAILED")
            ok, enc = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        payload = enc.tobytes() if ok else b""
        hash_value = hashlib.sha1(payload).hexdigest()
        return payload, hash_value

    def _build_placeholder(self, text: str) -> np.ndarray:
        img = np.zeros((self.cfg.screen_height, self.cfg.screen_width, 3), dtype=np.uint8)
        cv2.putText(img, text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(img, "SERIAL 5INCH", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    @staticmethod
    def list_available_ports() -> list[str]:
        return [p.device for p in list_ports.comports()]

    @staticmethod
    def _auto_detect_port() -> str:
        ports = list_ports.comports()
        if not ports:
            return ""

        preferred = [
            p for p in ports if any(k in (p.description or "").lower() for k in ("ch340", "cp210", "usb serial", "arduino"))
        ]
        return (preferred[0] if preferred else ports[0]).device

    def _log_connect_error_once(self, exc: Exception, port: str) -> None:
        signature = f"{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature:
            return
        self._last_error_signature = signature

        text = str(exc)
        if "PermissionError" in text or "拒绝访问" in text:
            LOGGER.warning("Serial port %s busy or access denied. Close other app and retry.", port)
        elif "FileNotFoundError" in text or "系统找不到指定的文件" in text:
            LOGGER.warning("Serial port %s not found. Check cable and COM number.", port)
        else:
            LOGGER.exception("Failed to connect serial backend")
