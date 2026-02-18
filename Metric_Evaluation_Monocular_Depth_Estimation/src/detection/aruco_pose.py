from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ArucoDetection:
    marker_id: int
    corners_px: np.ndarray  # (4,2) in image pixel coordinates
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,) in meters
    R: np.ndarray  # (3,3)
    n_gt: np.ndarray  # (3,) plane normal (unit)
    d_gt: float  # plane offset (n^T X + d = 0)
    mask: np.ndarray  # (H,W) uint8 mask for pixels inside marker polygon


def _aruco_module() -> "cv2.aruco":
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available. Install opencv-contrib-python (not opencv-python)."
        )
    return cv2.aruco


def load_calibration(camera_matrix_path: str, dist_coeffs_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load calibration from yaml produced by calibrate_camera.py."""
    import yaml

    with open(camera_matrix_path, "r", encoding="utf-8") as f:
        cam = yaml.safe_load(f)
    with open(dist_coeffs_path, "r", encoding="utf-8") as f:
        dist = yaml.safe_load(f)

    mtx = np.array(cam["camera_matrix"], dtype=np.float64)
    d = np.array(dist["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return mtx, d


def detect_aruco_pose(
    bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_m: float,
    dictionary_name: str = "DICT_4X4_100",
    marker_id: Optional[int] = None,
) -> Optional[ArucoDetection]:
    """Detect an ArUco marker, estimate pose, and compute GT plane.

    If marker_id is provided, returns the best detection for that id.
    Otherwise returns the first detected marker.
    """

    aruco = _aruco_module()

    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unknown dictionary: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    corners_list, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    ids = ids.flatten().astype(int)

    # pick index
    idx = 0
    if marker_id is not None:
        matches = np.where(ids == int(marker_id))[0]
        if len(matches) == 0:
            return None
        idx = int(matches[0])

    corners = corners_list[idx].reshape(4, 2).astype(np.float64)
    mid = int(ids[idx])

    # --- Pose estimation ---
    # Some OpenCV Python wheels expose cv2.aruco detection utilities but do NOT expose
    # aruco.estimatePoseSingleMarkers. We therefore estimate pose via plain cv2.solvePnP.
    L = float(marker_length_m)
    objp = np.array(
        [
            [-L / 2.0, +L / 2.0, 0.0],
            [+L / 2.0, +L / 2.0, 0.0],
            [+L / 2.0, -L / 2.0, 0.0],
            [-L / 2.0, -L / 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Prefer IPPE for planar squares if available
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE
    else:
        pnp_flag = cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=objp,
        imagePoints=corners,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=pnp_flag,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for detected marker corners")

    rvec = rvec.reshape(3).astype(np.float64)
    tvec = tvec.reshape(3).astype(np.float64)

    R, _ = cv2.Rodrigues(rvec)

    # Marker plane normal in camera frame: marker's +Z axis
    n_gt = R[:, 2].astype(np.float64)
    n_norm = np.linalg.norm(n_gt)
    if n_norm > 0:
        n_gt = n_gt / n_norm
    d_gt = float(-n_gt.T @ tvec)

    # Build mask for pixels inside marker polygon
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = corners.astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)

    return ArucoDetection(
        marker_id=mid,
        corners_px=corners,
        rvec=rvec,
        tvec=tvec,
        R=R,
        n_gt=n_gt,
        d_gt=d_gt,
        mask=mask,
    )
