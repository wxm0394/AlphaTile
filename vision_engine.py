"""Board ROI extraction, grid geometry, and template-based tile classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VisionEngine:
    """Crop and split a configured rectangular board from OpenCV BGR frames,
    and classify tiles using HSV histogram template matching.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._templates: dict[int, list[np.ndarray]] = {}
        self._template_histograms: dict[int, list[np.ndarray]] = {}
        self._match_threshold: float = 0.6
        self._center_ratio: float = 0.6
        self.update_config(config)

    def update_config(self, config: dict[str, Any]) -> None:
        """Replace configuration, validating board geometry before use."""
        board = config.get("board", {})
        self.top_left = self._point(board.get("board_top_left", [40, 620]), "board_top_left")
        self.bottom_right = self._point(
            board.get("board_bottom_right", [1040, 1820]), "board_bottom_right"
        )
        self.rows = int(board.get("grid_rows", 9))
        self.cols = int(board.get("grid_cols", 9))
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers")
        if self.bottom_right[0] <= self.top_left[0] or self.bottom_right[1] <= self.top_left[1]:
            raise ValueError("board_bottom_right must be right and below board_top_left")

        # Pre-compute cell dimensions for fast coordinate mapping
        self._cell_width = (self.bottom_right[0] - self.top_left[0]) / self.cols
        self._cell_height = (self.bottom_right[1] - self.top_left[1]) / self.rows

        # Optional vision-specific config
        vision_cfg = config.get("vision", {})
        self._match_threshold = float(vision_cfg.get("match_threshold", 0.6))
        self._center_ratio = float(vision_cfg.get("center_crop_ratio", 0.6))

    @staticmethod
    def _point(value: Any, name: str) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a two-element [x, y] coordinate")
        return int(value[0]), int(value[1])

    def _clamped_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        if frame.ndim < 2:
            raise ValueError(f"Expected an image frame, got shape {frame.shape}")
        height, width = frame.shape[:2]
        x1 = min(max(self.top_left[0], 0), width)
        y1 = min(max(self.top_left[1], 0), height)
        x2 = min(max(self.bottom_right[0], 0), width)
        y2 = min(max(self.bottom_right[1], 0), height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Board ROI is outside the frame or empty: {(x1, y1, x2, y2)}, "
                f"frame size is {width}x{height}"
            )
        return x1, y1, x2, y2

    def extract_board(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of the configured board ROI from a BGR frame."""
        x1, y1, x2, y2 = self._clamped_roi(frame)
        return frame[y1:y2, x1:x2].copy()

    def draw_debug_grid(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of ``frame`` with ROI and equal grid lines overlaid."""
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected a color BGR frame, got shape {frame.shape}")
        output = frame.copy()
        x1, y1, x2, y2 = self._clamped_roi(frame)
        green = (0, 255, 0)
        red = (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), green, 2)

        for col in range(1, self.cols):
            x = x1 + (col * (x2 - x1)) // self.cols
            cv2.line(output, (x, y1), (x, y2 - 1), red, 1)
        for row in range(1, self.rows):
            y = y1 + (row * (y2 - y1)) // self.rows
            cv2.line(output, (x1, y), (x2 - 1, y), red, 1)
        return output

    def split_grid(self, board_image: np.ndarray) -> list[list[np.ndarray]]:
        """Split a board image into rows x columns BGR tile crops."""
        if board_image.ndim < 2:
            raise ValueError(f"Expected a board image, got shape {board_image.shape}")
        height, width = board_image.shape[:2]
        if height < self.rows or width < self.cols:
            raise ValueError(
                f"Board image ({width}x{height}) is too small for "
                f"{self.rows}x{self.cols} grid"
            )
        x_edges = [(col * width) // self.cols for col in range(self.cols + 1)]
        y_edges = [(row * height) // self.rows for row in range(self.rows + 1)]
        return [
            [
                board_image[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]].copy()
                for col in range(self.cols)
            ]
            for row in range(self.rows)
        ]

    # ---------------------------------------------------------------------
    # Template loading and classification
    # ---------------------------------------------------------------------

    @staticmethod
    def _center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
        """Return the central ``ratio`` portion of an image (e.g., 0.6 = 60%)."""
        if image.ndim < 2:
            raise ValueError(f"Expected an image, got shape {image.shape}")
        h, w = image.shape[:2]
        if ratio <= 0 or ratio > 1:
            raise ValueError("center_crop_ratio must be in (0, 1]")
        ch = int(h * ratio)
        cw = int(w * ratio)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        return image[y1 : y1 + ch, x1 : x1 + cw]

    @staticmethod
    def _compute_hist(image: np.ndarray) -> np.ndarray:
        """Compute a normalized 2D HSV histogram (H x S) for a BGR image."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image, got shape {image.shape}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def load_templates(self, template_dir: str | Path = "templates") -> None:
        """Load template images from subdirectories named by integer class ID.

        Directory structure:
            templates/
                1/
                    tile_a.png
                    tile_b.png
                2/
                    tile_c.png
                ...

        For each template image, the central ``center_crop_ratio`` region is
        extracted and its HSV histogram is cached for fast matching.
        """
        path = Path(template_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {path}")

        self._templates.clear()
        self._template_histograms.clear()

        for class_dir in sorted(path.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999):
            if not class_dir.is_dir():
                continue
            try:
                class_id = int(class_dir.name)
            except ValueError:
                print(f"[VisionEngine] Skipping non-numeric template folder: {class_dir.name}")
                continue

            histograms: list[np.ndarray] = []
            for img_path in class_dir.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[VisionEngine] Failed to read: {img_path}")
                    continue
                cropped = self._center_crop(img, self._center_ratio)
                hist = self._compute_hist(cropped)
                histograms.append(hist)

            if histograms:
                self._template_histograms[class_id] = histograms
                print(f"[VisionEngine] Loaded {len(histograms)} templates for class {class_id}")
            else:
                print(f"[VisionEngine] No valid images in {class_dir}")

        if not self._template_histograms:
            print("[VisionEngine] WARNING: No templates loaded. predict_board will return all zeros.")

    def predict_board(self, tiles: list[list[np.ndarray]]) -> list[list[int]]:
        """Classify each tile against loaded templates, returning a matrix of class IDs.

        Returns 0 for tiles that don't match any template above the threshold.
        """
        if not self._template_histograms:
            return [[0 for _ in row] for row in tiles]

        result: list[list[int]] = []
        for row in tiles:
            result_row: list[int] = []
            for tile in row:
                cropped = self._center_crop(tile, self._center_ratio)
                tile_hist = self._compute_hist(cropped)

                best_id = 0
                best_score = -1.0
                for class_id, templates in self._template_histograms.items():
                    for tmpl_hist in templates:
                        score = cv2.compareHist(tile_hist, tmpl_hist, cv2.HISTCMP_CORREL)
                        if score > best_score:
                            best_score = score
                            best_id = class_id

                if best_score >= self._match_threshold:
                    result_row.append(best_id)
                else:
                    result_row.append(0)
            result.append(result_row)
        return result

    def draw_board_matrix(self, frame: np.ndarray, matrix: list[list[int]]) -> np.ndarray:
        """Overlay the predicted class IDs on the grid cells in the frame."""
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected a color BGR frame, got shape {frame.shape}")
        output = frame.copy()
        x1, y1, x2, y2 = self._clamped_roi(frame)

        cell_w = (x2 - x1) / self.cols
        cell_h = (y2 - y1) / self.rows

        for row_idx, row in enumerate(matrix):
            for col_idx, val in enumerate(row):
                cx = int(x1 + (col_idx + 0.5) * cell_w)
                cy = int(y1 + (row_idx + 0.5) * cell_h)
                text = str(val)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(
                    output,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
        return output

    def grid_to_screen(self, row: int, col: int) -> tuple[int, int]:
        """Convert grid (row, col) to physical screen pixel coordinates (X, Y).

        Returns the center point of the specified cell.
        Raises ValueError if row/col are out of bounds.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"Grid coordinates ({row}, {col}) out of bounds for {self.rows}x{self.cols} board"
            )
        x = int(self.top_left[0] + (col + 0.5) * self._cell_width)
        y = int(self.top_left[1] + (row + 0.5) * self._cell_height)
        return x, y

    def save_debug_image(
        self,
        frame: np.ndarray,
        matrix: list[list[int]],
        output_path: str | Path = "debug_output/hallucination_check.png",
    ) -> None:
        """Save a debug image showing the predicted class IDs overlaid on the grid.

        Draws green grid lines over the board ROI and writes the predicted class
        number (1-6, or 0 for unrecognized) in large red text at each cell centre.
        """
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected a color BGR frame, got shape {frame.shape}")

        output = frame.copy()
        x1, y1, x2, y2 = self._clamped_roi(frame)

        # Draw green grid lines
        green = (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), green, 2)
        for col in range(1, self.cols):
            x = x1 + (col * (x2 - x1)) // self.cols
            cv2.line(output, (x, y1), (x, y2 - 1), green, 1)
        for row in range(1, self.rows):
            y = y1 + (row * (y2 - y1)) // self.rows
            cv2.line(output, (x1, y), (x2 - 1, y), green, 1)

        # Draw predicted class IDs in red at each cell centre
        cell_w = (x2 - x1) / self.cols
        cell_h = (y2 - y1) / self.rows
        red = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        for row_idx, row in enumerate(matrix):
            for col_idx, val in enumerate(row):
                cx = int(x1 + (col_idx + 0.5) * cell_w)
                cy = int(y1 + (row_idx + 0.5) * cell_h)
                text = str(val)
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                # Black outline for readability, then red text on top
                cv2.putText(
                    output,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    font,
                    font_scale,
                    red,
                    thickness,
                    cv2.LINE_AA,
                )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), output)
        print(f"[VisionEngine] Debug image saved to {out_path}")
