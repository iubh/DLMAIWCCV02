"""Calibrate camera intrinsics from captured chessboard images.

Reads images from: data/calibration/chessboard_images/
Outputs:
  - data/calibration/camera_matrix.yaml
  - data/calibration/dist_coeffs.yaml

Also prints mean reprojection error.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml


def _reprojection_error(
    objpoints,
    imgpoints,
    rvecs,
    tvecs,
    mtx,
    dist,
) -> float:
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        n = len(imgpoints2)
        total_error += error * error
        total_points += n
    return float(np.sqrt(total_error / max(total_points, 1)))


def calibrate_from_folder(
    images_dir: Path,
    pattern_size: Tuple[int, int],
    square_size_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (camera_matrix, dist_coeffs, rmse_reproj_px)."""

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No calibration images found in {images_dir}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints = []
    imgpoints = []
    image_size = None

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

    if len(objpoints) < 8:
        raise RuntimeError(
            f"Not enough valid chessboard detections ({len(objpoints)}). Capture more images."
        )

    assert image_size is not None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    if not ret:
        raise RuntimeError("cv2.calibrateCamera failed")

    rmse = _reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    return mtx, dist, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/calibration/chessboard_images",
    )
    parser.add_argument("--pattern_cols", type=int, default=9)
    parser.add_argument("--pattern_rows", type=int, default=6)
    parser.add_argument(
        "--square_size_m",
        type=float,
        default=0.025,
        help="Chessboard square size in meters (e.g., 0.025 for 25mm)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/calibration",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / args.images_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = (args.pattern_cols, args.pattern_rows)
    mtx, dist, rmse = calibrate_from_folder(images_dir, pattern, args.square_size_m)

    cam_path = output_dir / "camera_matrix.yaml"
    dist_path = output_dir / "dist_coeffs.yaml"
    with open(cam_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"camera_matrix": mtx.tolist()}, f)
    with open(dist_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"dist_coeffs": dist.flatten().tolist()}, f)

    print(f"[OK] Saved camera matrix to {cam_path}")
    print(f"[OK] Saved dist coeffs to {dist_path}")
    print(f"[INFO] Mean reprojection RMSE: {rmse:.4f} px")


if __name__ == "__main__":
    main()
