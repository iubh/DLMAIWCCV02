from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detection.aruco_pose import detect_aruco_pose, load_calibration
from .depth.backprojection import Intrinsics, backproject_depth_to_points
from .depth.factory import create_depth_backend
from .depth.plane_alignment import align_points_to_plane, fit_plane_svd
from .evaluation.metrics import compute_metrics
from .io.capture_image import capture_one_frame
from .visualization.draw_pose import draw_aruco_overlay


def _default_output_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.jpg"


def run_on_image(
    image_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_m: float,
    marker_id: Optional[int],
    depth_backend: str,
    device: Optional[str],
    out_dir: Path,
) -> None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 1) ArUco pose + GT plane
    det = detect_aruco_pose(
        bgr=bgr,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_length_m=marker_length_m,
        marker_id=marker_id,
    )
    if det is None:
        raise RuntimeError("No ArUco marker detected (or marker_id not found).")

    # 2) Relative depth
    model = create_depth_backend(depth_backend, device=device)
    pred = model.predict(bgr)
    depth_rel = pred.depth

    # 3) Backproject marker region
    intr = Intrinsics.from_camera_matrix(camera_matrix)
    pts_rel = backproject_depth_to_points(depth_rel, intr, mask=det.mask)
    if len(pts_rel) < 50:
        raise RuntimeError(
            f"Not enough 3D points inside marker mask ({len(pts_rel)}). "
            "Check marker detection / depth output."
        )

    # 4) Plane fit in relative space (optional, mainly for debugging)
    _plane_rel = fit_plane_svd(pts_rel)

    # 5) Scale alignment to GT plane
    pts_metric, s = align_points_to_plane(pts_rel, det.n_gt, det.d_gt)
    plane_metric = fit_plane_svd(pts_metric)

    # 6) Metrics
    metrics = compute_metrics(
        n_gt=det.n_gt,
        d_gt=det.d_gt,
        n_pred=plane_metric.n,
        d_pred=plane_metric.d,
        points_metric=pts_metric,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    vis = draw_aruco_overlay(
        bgr=bgr,
        corners_px=det.corners_px,
        rvec=det.rvec,
        tvec=det.tvec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        metrics=metrics,
    )
    out_path = out_dir / _default_output_name("overlay")
    cv2.imwrite(str(out_path), vis)

    # Log summary
    print("\n=== Metric Evaluation Result ===")
    print(f"image: {image_path}")
    print(f"aruco_id: {det.marker_id}")
    print(f"marker_length_m: {marker_length_m}")
    print(f"depth_backend: {depth_backend} ({pred.meta})")
    print(f"scale_factor_s: {s:.6f}")
    print(f"normal_angle_deg: {metrics.normal_angle_deg:.3f}")
    print(f"plane_offset_abs_m: {metrics.plane_offset_abs_m:.6f}")
    print(f"rmse_point_to_gt_plane_m: {metrics.rmse_point_to_gt_plane_m:.6f}")
    print(f"saved_overlay: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric evaluation of monocular depth via ArUco plane geometry"
    )

    # IO
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture one image from webcam into data/images/ and exit.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an existing image to analyze. If omitted and --capture is not set, uses latest image in data/images/.",
    )
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--capture_output",
        type=str,
        default="data/images/capture.jpg",
        help="Where to store captured image (when --capture).",
    )

    # Calibration
    parser.add_argument(
        "--camera_matrix",
        type=str,
        default="data/calibration/camera_matrix.yaml",
    )
    parser.add_argument(
        "--dist_coeffs",
        type=str,
        default="data/calibration/dist_coeffs.yaml",
    )

    # Marker
    parser.add_argument("--marker_id", type=int, default=None)
    parser.add_argument("--marker_length_m", type=float, default=0.048)

    # Depth
    parser.add_argument(
        "--depth_backend",
        type=str,
        default="midas",
        help="midas | depth_anything_v2 (stub)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu|cuda (default: auto)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Capture mode
    if args.capture:
        out_path = project_root / args.capture_output
        res = capture_one_frame(out_path, camera_index=args.camera_index)
        if res is None:
            print("[INFO] Capture aborted.")
        else:
            print(f"[OK] Captured image: {res.path} ({res.width}x{res.height})")
        return

    # Load calibration
    cam_path = project_root / args.camera_matrix
    dist_path = project_root / args.dist_coeffs
    K, dist = load_calibration(str(cam_path), str(dist_path))

    # Determine image
    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = project_root / image_path
    else:
        # use latest image in data/images
        img_dir = project_root / "data/images"
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not imgs:
            raise FileNotFoundError(
                "No images found. Either run with --capture or pass --image path/to/image.jpg"
            )
        image_path = imgs[-1]

    run_on_image(
        image_path=image_path,
        camera_matrix=K,
        dist_coeffs=dist,
        marker_length_m=float(args.marker_length_m),
        marker_id=args.marker_id,
        depth_backend=args.depth_backend,
        device=args.device,
        out_dir=project_root / args.output_dir,
    )


if __name__ == "__main__":
    main()
