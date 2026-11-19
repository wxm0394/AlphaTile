"""Main bot loop for automated match-three gameplay.

Integrates DeviceController, VisionEngine, and LogicSolver into a complete
auto-play system with state validation, randomization, and safe shutdown.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from device_controller import DeviceController, load_config as load_device_config
from logic_solver import LogicSolver
from vision_engine import VisionEngine


def load_config(config_path: str | Path = "config.json") -> dict[str, Any]:
    """Load configuration from config.json and add bot-specific timing defaults."""
    cfg = load_device_config(config_path)
    # Override timing with bot-specific values
    timing = cfg.get("timing", {})
    cfg["timing"] = {
        "swipe_duration_ms": timing.get("swipe_duration_ms", 120),
        "swipe_duration_jitter_ms": timing.get("swipe_duration_jitter_ms", [40, 160]),
        "swipe_duration_range_ms": timing.get("swipe_duration_range_ms", [100, 250]),
        "coordinate_jitter_px": timing.get("coordinate_jitter_px", 5),
        "post_swipe_wait_sec": timing.get("post_swipe_wait_sec", [0.7, 1.2]),
        "dead_board_wait_sec": timing.get("dead_board_wait_sec", 1.0),
        "invalid_board_wait_sec": timing.get("invalid_board_wait_sec", 1.0),
        "min_valid_tile_ratio": timing.get("min_valid_tile_ratio", 0.3),
    }
    return cfg


def random_range(min_val: float, max_val: float) -> float:
    """Return a random float in [min_val, max_val]."""
    return random.uniform(min_val, max_val)


def random_int(min_val: int, max_val: int) -> int:
    """Return a random integer in [min_val, max_val]."""
    return random.randint(min_val, max_val)


class BotLoop:
    """Main auto-play loop for match-three game."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path = "config.json") -> None:
        self.config = config if config is not None else load_config(config_path)
        self.timing = self.config.get("timing", {})
        
        # Initialize components
        self.controller = DeviceController(self.config)
        self.vision = VisionEngine(self.config)
        self.solver = LogicSolver()
        
        # Load template library for tile classification
        template_dir = self.config.get("templates", {}).get("dir", "templates")
        match_threshold = self.config.get("templates", {}).get("match_threshold", 0.82)
        self.vision._match_threshold = match_threshold
        self.vision.load_templates(template_dir)
        
        # Timing parameters
        self.swipe_duration_range = self.timing.get("swipe_duration_range_ms", [100, 250])
        self.coordinate_jitter = self.timing.get("coordinate_jitter_px", 5)
        self.post_swipe_wait = self.timing.get("post_swipe_wait_sec", [0.7, 1.2])
        self.dead_board_wait = self.timing.get("dead_board_wait_sec", 1.0)
        self.invalid_board_wait = self.timing.get("invalid_board_wait_sec", 1.0)
        self.min_valid_ratio = self.timing.get("min_valid_tile_ratio", 0.3)
        
        # Debug output
        self.debug_enabled = self.config.get("debug", {}).get("enabled", False)
        self.debug_dir = Path(self.config.get("debug", {}).get("save_dir", "debug_output"))
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "dead_frames": 0,
            "moves_executed": 0,
            "start_time": 0.0,
        }
        
        # Banned moves: short-term memory to prevent dead-loop on same invalid swap
        self.banned_moves: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.last_matrix_hash: str | None = None
        self.last_move: tuple[tuple[int, int], tuple[int, int]] | None = None

    def _get_valid_tile_ratio(self, matrix: list[list[int]]) -> float:
        """Calculate the ratio of non-zero tiles in the matrix."""
        if not matrix or not matrix[0]:
            return 0.0
        total = len(matrix) * len(matrix[0])
        valid = sum(1 for row in matrix for cell in row if cell != 0)
        return valid / total

    def _apply_jitter(self, x: int, y: int) -> tuple[int, int]:
        """Add random jitter to coordinates to simulate human input."""
        jitter = self.coordinate_jitter
        dx = random_int(-jitter, jitter)
        dy = random_int(-jitter, jitter)
        return x + dx, y + dy

    def _execute_move(self, move: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """Execute a move with randomized timing and coordinates."""
        (r1, c1), (r2, c2) = move
        
        # Convert grid to screen coordinates
        x1, y1 = self.vision.grid_to_screen(r1, c1)
        x2, y2 = self.vision.grid_to_screen(r2, c2)
        
        # Apply jitter
        x1, y1 = self._apply_jitter(x1, y1)
        x2, y2 = self._apply_jitter(x2, y2)
        
        # Randomize swipe duration
        duration = random_int(self.swipe_duration_range[0], self.swipe_duration_range[1])
        
        # Execute swipe
        self.controller.swipe(x1, y1, x2, y2, duration)
        
        # Log action
        print(
            f"[Action] Swipe ({x1}, {y1}) -> ({x2}, {y2}), "
            f"duration={duration}ms, wait={self.post_swipe_wait[0]:.1f}s-{self.post_swipe_wait[1]:.1f}s"
        )
        
        # Wait for animation to complete
        wait_time = random_range(self.post_swipe_wait[0], self.post_swipe_wait[1])
        time.sleep(wait_time)
        
        self.stats["moves_executed"] += 1

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame: extract board, predict, find move, execute."""
        self.stats["total_frames"] += 1
        
        # Extract board ROI
        board = self.vision.extract_board(frame)
        
        # Split into grid
        tiles = self.vision.split_grid(board)
        
        # Predict tile classes
        matrix = self.vision.predict_board(tiles)
        
        # Validate board state
        valid_ratio = self._get_valid_tile_ratio(matrix)
        
        if valid_ratio < self.min_valid_ratio:
            self.stats["invalid_frames"] += 1
            print(f"[State] Invalid board: {valid_ratio:.1%} valid tiles, waiting...")
            time.sleep(self.invalid_board_wait)
            return
        
        self.stats["valid_frames"] += 1

        # Compare with the previous board observation to identify rejected moves.
        matrix_hash = str(tuple(tuple(row) for row in matrix))
        if self.last_matrix_hash is not None:
            if matrix_hash == self.last_matrix_hash and self.last_move is not None:
                if self.last_move not in self.banned_moves:
                    self.banned_moves.add(self.last_move)
                    print(f"[Banned] Move {self.last_move} left the board unchanged; banning it")
                    self.vision.save_debug_image(
                        frame, matrix, "debug_output/hallucination_check.png"
                    )
            elif matrix_hash != self.last_matrix_hash:
                if self.banned_moves:
                    print(f"[State] Board changed; clearing {len(self.banned_moves)} banned move(s)")
                self.banned_moves.clear()
        self.last_matrix_hash = matrix_hash
        
        # Find the best unbanned move
        move = self.solver.find_best_move(matrix, self.banned_moves)
        
        if move is None and self.banned_moves:
            print(f"[Banned] All legal moves exhausted ({len(self.banned_moves)} banned); clearing blacklist")
            self.vision.save_debug_image(
                frame, matrix, "debug_output/hallucination_check.png"
            )
            self.banned_moves.clear()
            rows = len(matrix)
            cols = len(matrix[0]) if rows else 0
            adjacent_moves = [
                ((r, c), (nr, nc))
                for r in range(rows)
                for c in range(cols)
                for nr, nc in ((r, c + 1), (r + 1, c))
                if nr < rows and nc < cols
            ]
            if adjacent_moves:
                move = random.choice(adjacent_moves)
                print(f"[Trying next] Random adjacent swap to break stalemate: {move}")
            else:
                print("[State] Board too small for a fallback adjacent swipe")
                time.sleep(self.dead_board_wait)
                return
        
        if move is None:
            self.stats["dead_frames"] += 1
            print("[State] No valid moves found (dead board), waiting...")
            time.sleep(self.dead_board_wait)
            return
        
        # Remember the attempted move for comparison with the next board observation.
        self.last_move = move
        # Execute the move
        self._execute_move(move)

    def run(self) -> None:
        """Run the main bot loop until interrupted."""
        print("=" * 60)
        print("Starting automated match-three bot loop")
        print("=" * 60)
        
        self.stats["start_time"] = time.time()
        
        try:
            # Main loop - iter_video_frames starts the stream internally
            print("[System] Starting Scrcpy video stream and entering main loop (Ctrl+C to stop)...")
            for frame in self.controller.iter_video_frames(fps=10.0):
                self._process_frame(frame)
                
        except KeyboardInterrupt:
            print("\n[System] Interrupted by user, shutting down...")
        except Exception as e:
            print(f"\n[Error] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Cleanup resources and print statistics."""
        print("\n" + "=" * 60)
        print("Bot shutdown complete")
        print("=" * 60)
        
        # Stop video stream
        if hasattr(self.controller, "_scrcpy_client") and self.controller._scrcpy_client is not None:
            print("[System] Stopping video stream...")
            self.controller.stop_video_stream()
        
        # Print statistics
        elapsed = time.time() - self.stats["start_time"]
        print(f"\nStatistics:")
        print(f"  Total frames processed: {self.stats['total_frames']}")
        print(f"  Valid frames: {self.stats['valid_frames']} ({self.stats['valid_frames']/max(self.stats['total_frames'],1)*100:.1f}%)")
        print(f"  Invalid frames: {self.stats['invalid_frames']} ({self.stats['invalid_frames']/max(self.stats['total_frames'],1)*100:.1f}%)")
        print(f"  Dead frames: {self.stats['dead_frames']} ({self.stats['dead_frames']/max(self.stats['total_frames'],1)*100:.1f}%)")
        print(f"  Moves executed: {self.stats['moves_executed']}")
        print(f"  Runtime: {elapsed:.1f}s")
        print(f"  Moves per second: {self.stats['moves_executed']/max(elapsed, 0.1):.1f}")
        
        print("\n[System] Bot stopped gracefully")


def main() -> None:
    """Entry point for the bot."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated match-three game bot with Scrcpy video streaming"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Video stream FPS (default: 10.0)",
    )
    
    args = parser.parse_args()
    
    try:
        bot = BotLoop(config_path=args.config)
        bot.run()
    except Exception as e:
        print(f"[Fatal] Failed to start bot: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
