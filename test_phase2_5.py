"""Phase 2.5: Live board matrix monitoring with template classification.

Starts the Scrcpy stream, extracts the board, splits into tiles, classifies
each tile against loaded templates, and prints the numeric matrix to the
console. Optionally overlays the predicted IDs on the OpenCV preview.

Controls:
  q / Escape : Quit
  r          : Reload config.json (hot-reload board coordinates)
  t          : Reload templates from templates/ directory
  v          : Toggle visual ID overlay on/off
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from device_controller import DeviceController, load_config
from vision_engine import VisionEngine


def print_matrix(matrix: list[list[int]]) -> None:
    """Pretty-print a 2D integer matrix to the console."""
    if not matrix:
        print("(empty)")
        return
    rows = len(matrix)
    cols = len(matrix[0])
    # Column headers
    header = "    " + " ".join(f"{c:2d}" for c in range(cols))
    print(header)
    print("   " + "-" * (cols * 3 - 1))
    for r, row in enumerate(matrix):
        line = f"{r:2d} |" + " ".join(f"{v:2d}" for v in row)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5: Live board matrix monitoring")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Maximum frame extraction rate (default: 10)"
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("templates"),
        help="Template directory (default: templates)",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable visual ID overlay on the preview window",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    config = load_config("config.json")
    controller = DeviceController(config)
    vision = VisionEngine(config)

    # Load templates
    try:
        vision.load_templates(args.templates)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}. Run tool_extract_tiles.py first, then organize templates/ manually.")

    print(f"Connected device: {controller.device.serial}")
    print(f"Device resolution: {controller.screen_width}x{controller.screen_height}")
    print(f"Board ROI: {vision.top_left} -> {vision.bottom_right}")
    print(f"Grid: {vision.rows} rows x {vision.cols} cols")
    print(f"Templates loaded: {list(vision._template_histograms.keys())}")
    print(f"Match threshold: {vision._match_threshold}")
    print(f"Center crop ratio: {vision._center_ratio}")
    print("Starting Scrcpy stream...")
    print("Controls:")
    print("  q / Escape : Quit")
    print("  r          : Reload config.json (hot-reload board coordinates)")
    print("  t          : Reload templates from templates/ directory")
    print("  v          : Toggle visual ID overlay on/off")

    window_name = "Phase 2.5 - Board Matrix Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_overlay = not args.no_overlay
    frame_count = 0
    started_at = time.monotonic()
    last_print_time = 0.0
    print_interval = 1.0  # seconds between console matrix prints

    try:
        for frame in controller.iter_video_frames(fps=args.fps):
            if frame.dtype.name != "uint8" or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Expected BGR uint8 frame, received {frame.dtype} {frame.shape}")

            board = vision.extract_board(frame)
            tiles = vision.split_grid(board)
            matrix = vision.predict_board(tiles)

            now = time.monotonic()
            if now - last_print_time >= print_interval:
                print(f"\n[{time.strftime('%H:%M:%S')}] Board matrix ({vision.rows}x{vision.cols}):")
                print_matrix(matrix)
                last_print_time = now

            if show_overlay:
                annotated = vision.draw_debug_grid(frame)
                annotated = vision.draw_board_matrix(annotated, matrix)
            else:
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
            if key == ord("t"):
                try:
                    vision.load_templates(args.templates)
                    print(f"[{time.strftime('%H:%M:%S')}] Templates reloaded: "
                          f"{list(vision._template_histograms.keys())}")
                except Exception as exc:
                    print(f"[{time.strftime('%H:%M:%S')}] Template reload failed: {exc}")
            if key == ord("v"):
                show_overlay = not show_overlay
                print(f"[{time.strftime('%H:%M:%S')}] Visual overlay: {'ON' if show_overlay else 'OFF'}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        controller.stop_video_stream()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.2f} FPS).")


if __name__ == "__main__":
    main()