"""Phase 2: Live board grid calibration tool.

Starts the Scrcpy stream, overlays the configured board ROI and grid lines,
and supports hot-reloading config.json with the 'r' key.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from device_controller import DeviceController, load_config
from vision_engine import VisionEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Live board grid calibration")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Maximum frame extraction rate (default: 10)"
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    config = load_config("config.json")
    controller = DeviceController(config)
    vision = VisionEngine(config)

    print(f"Connected device: {controller.device.serial}")
    print(f"Device resolution: {controller.screen_width}x{controller.screen_height}")
    print(f"Board ROI: {vision.top_left} -> {vision.bottom_right}")
    print(f"Grid: {vision.rows} rows x {vision.cols} cols")
    print("Starting Scrcpy stream...")
    print("Controls:")
    print("  q / Escape : Quit")
    print("  r          : Reload config.json (hot-reload board coordinates)")

    window_name = "Phase 2 - Board Grid Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    started_at = time.monotonic()
    try:
        for frame in controller.iter_video_frames(fps=args.fps):
            if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Expected BGR uint8 frame, received {frame.dtype} {frame.shape}")

            annotated = vision.draw_debug_grid(frame)
            cv2.imshow(window_name, annotated)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                try:
                    new_config = load_config("config.json")
                    vision.update_config(new_config)
                    print(f"[{time.strftime('%H:%M:%S')}] Config reloaded: "
                          f"ROI={vision.top_left}->{vision.bottom_right}, "
                          f"Grid={vision.rows}x{vision.cols}")
                except Exception as exc:
                    print(f"[{time.strftime('%H:%M:%S')}] Config reload failed: {exc}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        controller.stop_video_stream()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(f"Received {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.2f} FPS).")


if __name__ == "__main__":
    main()