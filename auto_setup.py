#!/usr/bin/env python3
"""Fully automated template library builder using K-means clustering.

This script replaces the manual 's' key workflow:
1. Captures a single frame from Scrcpy
2. Extracts 70 tiles (10x7 grid) using VisionEngine
3. Computes HSV histograms from center 50% of each tile
4. Filters out black/low-brightness invalid tiles
5. Runs K-means (K=6) on histogram features
6. Creates templates/1..6 directories with the most representative tile per cluster
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from device_controller import DeviceController, load_config
from vision_engine import VisionEngine


def load_config_file(config_path: str = "config.json") -> dict[str, Any]:
    """Load configuration from config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tile_features(tiles: list[list[np.ndarray]], center_ratio: float = 0.5) -> tuple[list[np.ndarray], list[tuple[int, int]], list[np.ndarray]]:
    """
    Extract HSV histogram features from the center region of each tile.
    
    Returns:
        - features: List of flattened HSV histograms (valid tiles only)
        - positions: List of (row, col) positions for valid tiles
        - tile_images: List of the original tile images (valid tiles only)
    """
    features = []
    positions = []
    tile_images = []
    
    for row_idx, row in enumerate(tiles):
        for col_idx, tile in enumerate(row):
            if tile.ndim != 3 or tile.shape[2] != 3:
                continue
            
            h, w = tile.shape[:2]
            if h < 10 or w < 10:
                continue
            
            # Extract center region (50% by default)
            ch = int(h * center_ratio)
            cw = int(w * center_ratio)
            y1 = (h - ch) // 2
            x1 = (w - cw) // 2
            center_crop = tile[y1:y1+ch, x1:x1+cw]
            
            # Convert to HSV and compute 2D histogram (H x S)
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_flat = hist.flatten().astype(np.float32)
            
            # Filter out pure black or very low brightness tiles
            # Check V channel (brightness) mean
            v_mean = np.mean(hsv[:, :, 2])
            s_mean = np.mean(hsv[:, :, 1])
            
            # Skip if too dark (V < 30) or too low saturation (S < 20) - likely empty/background
            if v_mean < 30 or s_mean < 20:
                print(f"  [Filter] Skipping tile ({row_idx},{col_idx}): V_mean={v_mean:.1f}, S_mean={s_mean:.1f}")
                continue
            
            features.append(hist_flat)
            positions.append((row_idx, col_idx))
            tile_images.append(tile.copy())
    
    return features, positions, tile_images


def run_kmeans_clustering(features: list[np.ndarray], k: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run K-means clustering on histogram features.
    
    Returns:
        - labels: Cluster assignment for each feature
        - centers: Cluster centroids
        - distances: Distance of each point to its assigned centroid
    """
    if len(features) < k:
        raise ValueError(f"Not enough valid tiles ({len(features)}) for K={k} clusters")
    
    data = np.array(features, dtype=np.float32)
    
    # K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    
    # Run K-means with multiple attempts for stability
    compactness, labels, centers = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    # Compute distance of each point to its assigned centroid
    distances = np.zeros(len(data), dtype=np.float32)
    for i in range(len(data)):
        cluster_id = labels[i][0]
        distances[i] = np.linalg.norm(data[i] - centers[cluster_id])
    
    return labels.flatten(), centers, distances


def find_representative_tiles(
    labels: np.ndarray,
    distances: np.ndarray,
    positions: list[tuple[int, int]],
    tile_images: list[np.ndarray],
    k: int = 6
) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    """
    Find the most representative tile (closest to centroid) for each cluster.
    
    Returns:
        Dict mapping cluster_id -> (tile_image, (row, col))
    """
    representatives = {}
    
    for cluster_id in range(k):
        # Find indices belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            print(f"  [Warning] Cluster {cluster_id} is empty!")
            continue
        
        # Find the one with minimum distance to centroid
        best_idx = cluster_indices[np.argmin(distances[cluster_indices])]
        representatives[cluster_id] = (tile_images[best_idx], positions[best_idx])
        print(f"  [Cluster {cluster_id}] Representative: tile at {positions[best_idx]}, "
              f"distance={distances[best_idx]:.4f}, cluster_size={len(cluster_indices)}")
    
    return representatives


def save_templates(
    representatives: dict[int, tuple[np.ndarray, tuple[int, int]]],
    template_dir: Path = Path("templates")
) -> None:
    """Save representative tiles to templates/1, templates/2, ... directories."""
    
    # Replace only numeric class folders; preserve unrelated assets (e.g. UI templates).
    template_dir.mkdir(parents=True, exist_ok=True)
    for existing in template_dir.iterdir():
        if existing.is_dir() and existing.name.isdigit():
            for child in existing.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    import shutil
                    shutil.rmtree(child)
            existing.rmdir()
    
    # Sort by cluster ID for consistent numbering (1-based)
    for cluster_id in sorted(representatives.keys()):
        tile_img, (row, col) = representatives[cluster_id]
        class_id = cluster_id + 1  # 1-based for template folders
        
        class_dir = template_dir / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG
        out_path = class_dir / f"template_{class_id}_row{row}_col{col}.png"
        cv2.imwrite(str(out_path), tile_img)
        print(f"  [Saved] Class {class_id} -> {out_path}")


def main() -> None:
    print("=" * 60)
    print("AUTO SETUP: Building template library via K-means clustering")
    print("=" * 60)
    
    # Load config
    config = load_config_file("config.json")
    print(f"[Config] Board: {config['board']['grid_rows']}x{config['board']['grid_cols']}")
    print(f"[Config] ROI: {config['board']['board_top_left']} -> {config['board']['board_bottom_right']}")
    
    # Initialize device and vision
    print("\n[System] Connecting to device...")
    controller = DeviceController(config)
    print(f"[System] Device: {controller.device.serial}")
    print(f"[System] Resolution: {controller.screen_width}x{controller.screen_height}")
    
    vision = VisionEngine(config)
    print(f"[Vision] Grid: {vision.rows}x{vision.cols}")
    print(f"[Vision] Cell size: ~{vision._cell_width:.0f}x{vision._cell_height:.0f}px")
    
    # Capture a single frame
    print("\n[Capture] Getting frame from Scrcpy...")
    frame = None
    try:
        # iter_video_frames starts and stops the Scrcpy stream itself.
        for captured_frame in controller.iter_video_frames(fps=5.0):
            frame = captured_frame
            break
    finally:
        controller.stop_video_stream()
    
    if frame is None:
        raise RuntimeError("Failed to capture frame from device")
    
    print(f"[Capture] Frame shape: {frame.shape}")
    
    # Extract board and split into tiles
    print("\n[Vision] Extracting board and splitting grid...")
    board = vision.extract_board(frame)
    tiles = vision.split_grid(board)
    print(f"[Vision] Extracted {len(tiles)}x{len(tiles[0])} = {len(tiles)*len(tiles[0])} tiles")
    
    # Extract features from center 50% of each tile
    print("\n[Features] Computing HSV histograms (center 50%)...")
    features, positions, tile_images = extract_tile_features(tiles, center_ratio=0.5)
    print(f"[Features] Valid tiles after filtering: {len(features)} / {len(tiles)*len(tiles[0])}")
    
    if len(features) < 6:
        raise RuntimeError(f"Too few valid tiles ({len(features)}). Check board ROI and game state.")
    
    # Run K-means clustering (K=6 for 6 animal types)
    print("\n[Clustering] Running K-means (K=6)...")
    labels, centers, distances = run_kmeans_clustering(features, k=6)
    
    # Find representative tile for each cluster
    print("\n[Selection] Finding most representative tile per cluster...")
    representatives = find_representative_tiles(labels, distances, positions, tile_images, k=6)
    
    # Save templates
    print("\n[Save] Writing template images...")
    save_templates(representatives, Path("templates"))
    
    print("\n" + "=" * 60)
    print("AUTO SETUP COMPLETE!")
    print("=" * 60)
    print(f"Templates created in: {Path('templates').resolve()}")
    print("You can now run: python main_bot.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        exit(1)