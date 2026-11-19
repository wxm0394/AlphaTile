"""Live Scrcpy video preview and frame-rate smoke test.

Run with the project virtualenv active:
    python test_phase1.py

Press q or Escape in the preview window to stop. Each frame yielded by the
controller is a BGR uint8 numpy array suitable for OpenCV processing.
"""

from __future__ import annotations

import argparse
import time

import cv2

from device_controller import DeviceController, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the Scrcpy stream with OpenCV")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Maximum frame extraction rate (default: 10)"
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    controller = DeviceController(load_config("config.json"))
    print(f"Connected device: {controller.device.serial}")
    print(f"Device resolution: {controller.screen_width}x{controller.screen_height}")
    print(f"Starting Scrcpy stream; extracting up to {args.fps:g} FPS.")
    print("Press q or Escape in the preview window to quit.")

    window_name = "Scrcpy live stream (BGR)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_count = 0
    started_at = time.monotonic()
    try:
        for frame in controller.iter_video_frames(fps=args.fps):
            if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Expected BGR uint8 frame, received {frame.dtype} {frame.shape}")
            cv2.imshow(window_name, frame)
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        controller.stop_video_stream()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(f"Received {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.2f} FPS).")


if __name__ == "__main__":
    main()
