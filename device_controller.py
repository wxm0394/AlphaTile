"""ADB device control and Scrcpy video streaming.

Communication only: connect to Android, capture/stream frames, and send input.
Board recognition and game logic live elsewhere.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import uiautomator2 as u2
from adbutils import AdbClient, AdbError


def load_config(config_path: str | Path = "config.json") -> dict[str, Any]:
    """Read config.json. The path is relative to the current working directory."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class DeviceController:
    """Wrap one Android device connected through the local ADB server."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path = "config.json",
    ) -> None:
        self.config = config if config is not None else load_config(config_path)
        device_cfg = self.config.get("device", {})
        self.serial: str = str(device_cfg.get("serial") or "").strip()
        self.adb_path: str = str(device_cfg.get("adb_path") or "adb")
        self.screenshot_method: str = str(
            device_cfg.get("screenshot_method") or "exec_out"
        )

        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self._pick_device()
        self.serial = self.device.serial
        self.d = u2.connect(self.serial)
        self.screen_width, self.screen_height = self._read_screen_size()

        self._scrcpy_client: Any | None = None
        self._frame_condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._frame_sequence = 0
        self._last_frame_time = 0.0
        self._frame_interval = 0.1

    def _pick_device(self):
        """Select by configured serial, or use the only online device."""
        try:
            devices = self.client.device_list()
        except AdbError as exc:
            raise RuntimeError(
                "Cannot connect to the ADB server. Start adb and enable USB or TCP/IP debugging."
            ) from exc

        if not devices:
            raise RuntimeError(
                "No online Android device found. Connect it and check `adb devices`."
            )

        if self.serial:
            for dev in devices:
                if dev.serial == self.serial:
                    return dev
            online = ", ".join(dev.serial for dev in devices)
            raise RuntimeError(
                f"Configured serial={self.serial!r} is not online. Online devices: {online}"
            )

        if len(devices) > 1:
            online = ", ".join(dev.serial for dev in devices)
            raise RuntimeError(
                "More than one device is online; set device.serial in config.json. "
                f"Online devices: {online}"
            )
        return devices[0]

    def _adb_base_cmd(self) -> list[str]:
        cmd = [self.adb_path]
        if self.device.serial:
            cmd.extend(["-s", self.device.serial])
        return cmd

    def _read_screen_size(self) -> tuple[int, int]:
        """Read the device resolution, falling back to config.screen."""
        screen_cfg = self.config.get("screen", {})
        fallback = (
            int(screen_cfg.get("width", 1080)),
            int(screen_cfg.get("height", 2400)),
        )
        try:
            raw = self.device.shell("wm size")
        except AdbError:
            return fallback

        # Prefer override size when both physical and override sizes are present.
        override = re.search(r"Override size:\s*(\d+)x(\d+)", raw)
        physical = re.search(r"Physical size:\s*(\d+)x(\d+)", raw)
        match = override or physical
        if not match:
            return fallback
        return int(match.group(1)), int(match.group(2))

    def get_screenshot(self) -> np.ndarray:
        """Capture a single screen as an OpenCV BGR uint8 array (H, W, 3)."""
        if self.screenshot_method == "adbutils":
            return self._screenshot_via_adbutils()
        try:
            return self._screenshot_via_exec_out()
        except RuntimeError:
            # Some older devices truncate exec-out binary data; fall back to adbutils.
            return self._screenshot_via_adbutils()

    def _screenshot_via_exec_out(self) -> np.ndarray:
        """Capture via `adb exec-out screencap -p`, without writing a file."""
        cmd = self._adb_base_cmd() + ["exec-out", "screencap", "-p"]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or "exec-out screencap failed")

        encoded = np.frombuffer(proc.stdout, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Screenshot decode failed: response is not a valid PNG.")
        return image

    def _screenshot_via_adbutils(self) -> np.ndarray:
        """Capture with adbutils, which returns a PIL RGB/RGBA image."""
        pil_image = self.device.screenshot()
        rgb = np.asarray(pil_image)
        if rgb.ndim != 3:
            raise RuntimeError(f"Unexpected screenshot shape: {rgb.shape}")
        if rgb.shape[2] == 4:
            return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def start_video_stream(self, fps: float = 10.0) -> None:
        """Start Scrcpy decoding in a worker thread.

        Incoming frames are rate-limited to at most ``fps`` and retained as BGR
        uint8 numpy arrays. ``iter_video_frames`` provides a convenient consumer.
        Scrcpy uses the device selected by the existing local ADB connection, so
        USB and already-connected TCP/IP ADB devices are both supported.
        """
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if self._scrcpy_client is not None:
            raise RuntimeError("The Scrcpy stream is already running.")

        try:
            import pyscrcpy
        except ImportError as exc:
            raise RuntimeError(
                "Scrcpy streaming dependency is missing. Install requirements.txt "
                "inside the active Python environment."
            ) from exc

        self._frame_interval = 1.0 / float(fps)
        self._last_frame_time = 0.0
        self._latest_frame = None
        self._frame_sequence = 0
        client = pyscrcpy.Client(device=self.device, max_fps=int(fps))

        def on_frame(client: Any, frame: np.ndarray | None) -> None:
            if frame is None:
                return
            now = time.monotonic()
            with self._frame_condition:
                if now - self._last_frame_time < self._frame_interval:
                    return
                image = np.asarray(frame)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.ndim != 3 or image.shape[2] != 3:
                    return
                # pyscrcpy's EVENT_FRAME is already BGR for OpenCV consumers.
                self._latest_frame = np.ascontiguousarray(image.copy())
                self._last_frame_time = now
                self._frame_sequence += 1
                self._frame_condition.notify_all()

        client.on_frame(on_frame)
        self._scrcpy_client = client
        try:
            # pyscrcpy stores its worker thread privately; join it on stop to
            # avoid orphan decode threads when the preview exits.
            self._scrcpy_worker = threading.Thread(
                target=client.start, kwargs={"threaded": True}, daemon=True
            )
            self._scrcpy_worker.start()
            self._scrcpy_ready = threading.Event()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if client.alive:
                    break
                if not self._scrcpy_worker.is_alive():
                    raise RuntimeError("Scrcpy client stopped before the video stream started.")
                time.sleep(0.01)
            else:
                client.stop()
                self._scrcpy_worker.join(timeout=2.0)
                raise TimeoutError("Timed out while starting the Scrcpy video stream.")
        except Exception:
            self._scrcpy_client = None
            raise

    def iter_video_frames(self, fps: float = 10.0) -> Iterator[np.ndarray]:
        """Yield fresh BGR frames at up to ``fps`` until the stream is stopped."""
        self.start_video_stream(fps=fps)
        last_sequence = 0
        try:
            while self._scrcpy_client is not None:
                with self._frame_condition:
                    self._frame_condition.wait_for(
                        lambda: self._frame_sequence != last_sequence
                        or self._scrcpy_client is None,
                        timeout=1.0,
                    )
                    if self._scrcpy_client is None:
                        break
                    if self._frame_sequence == last_sequence:
                        continue
                    last_sequence = self._frame_sequence
                    frame = self._latest_frame
                if frame is not None:
                    yield frame
        finally:
            self.stop_video_stream()

    def stop_video_stream(self) -> None:
        """Stop the Scrcpy worker if it is running."""
        client = self._scrcpy_client
        if client is None:
            return
        self._scrcpy_client = None
        with self._frame_condition:
            self._frame_condition.notify_all()
        client.stop()
        worker = getattr(self, "_scrcpy_worker", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        self._scrcpy_worker = None

    def swipe(
        self,
        x1: int | float,
        y1: int | float,
        x2: int | float,
        y2: int | float,
        duration: int = 120,
    ) -> None:
        """Simulate a swipe through the uiautomator2 input driver.

        Duration is supplied in milliseconds and converted to seconds for u2.
        """
        sx, sy, ex, ey = (int(x1), int(y1), int(x2), int(y2))
        self.d.swipe(sx, sy, ex, ey, duration=duration / 1000.0)

    def tap(self, x: int | float, y: int | float) -> None:
        """Simulate a short tap through the uiautomator2 input driver."""
        self.d.click(int(x), int(y))
