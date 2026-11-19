"""Phase 2.5: Tile extraction tool for building the template library.

Starts the Scrcpy stream, overlays the board grid, and saves all tiles
when the user presses 's'. Tiles are written to debug_output/raw_tiles/
with names like frame{N}_row{R}_col{C}.png.

Controls:
  s          : Save current frame's tiles
  q / Escape : Quit
  r          : Reload config.json (hot-reload board coordinates)
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
    parser = argparse.ArgumentParser(description="Extract tiles for template library")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Maximum frame extraction rate (default: 10)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("debug_output/raw_tiles"),
        help="Directory to save extracted tiles (default: debug_output/raw_tiles)",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("config.json")
    controller = DeviceController(config)
    vision = VisionEngine(config)

    print(f"Connected device: {controller.device.serial}")
    print(f"Device resolution: {controller.screen_width}x{controller.screen_height}")
    print(f"Board ROI: {vision.top_left} -> {vision.bottom_right}")
    print(f"Grid: {vision.rows} rows x {vision.cols} cols")
    print(f"Output directory: {args.out_dir.resolve()}")
    print("Starting Scrcpy stream...")
    print("Controls:")
    print("  s          : Save current frame's tiles")
    print("  q / Escape : Quit")
    print("  r          : Reload config.json (hot-reload board coordinates)")

    window_name = "Phase 2.5 - Tile Extractor (press 's' to save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    saved_count = 0
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
            if key == ord("s"):
                board = vision.extract_board(frame)
                tiles = vision.split_grid(board)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                for row_idx, row in enumerate(tiles):
                    for col_idx, tile in enumerate(row):
                        filename = f"frame{saved_count}_row{row_idx}_col{col_idx}_{timestamp}.png"
                        out_path = args.out_dir / filename
                        cv2.imwrite(str(out_path), tile)
                print(f"[{time.strftime('%H:%M:%S')}] Saved {len(tiles)}x{len(tiles[0])} tiles "
                      f"to {args.out_dir} (batch {saved_count})")
                saved_count += 1
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        controller.stop_video_stream()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_at, 1e-9)
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.2f} FPS).")
    print(f"Saved {saved_count} tile batches.")


if __name__ == "__main__":
    main()