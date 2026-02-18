from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..evaluation.metrics import PlaneMetrics


def draw_aruco_overlay(
    bgr: np.ndarray,
    corners_px: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    metrics: Optional[PlaneMetrics] = None,
    axis_length_m: float = 0.03,
) -> np.ndarray:
    out = bgr.copy()

    # Draw marker polygon
    pts = corners_px.reshape(4, 2).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw axis
    try:
        cv2.drawFrameAxes(out, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)
    except Exception:
        # drawFrameAxes may fail if OpenCV built without it; ignore
        pass

    if metrics is not None:
        lines = [
            f"normal_angle_deg: {metrics.normal_angle_deg:.2f}",
            f"plane_offset_abs_m: {metrics.plane_offset_abs_m:.4f}",
            f"rmse_point_to_gt_plane_m: {metrics.rmse_point_to_gt_plane_m:.4f}",
        ]
        y = 25
        for ln in lines:
            cv2.putText(
                out,
                ln,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            y += 22

    return out
