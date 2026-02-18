from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PlaneMetrics:
    normal_angle_deg: float
    plane_offset_abs_m: float
    rmse_point_to_gt_plane_m: float


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def plane_normal_angle_deg(n_gt: np.ndarray, n_pred: np.ndarray) -> float:
    n1 = _safe_unit(n_gt.reshape(3).astype(np.float64))
    n2 = _safe_unit(n_pred.reshape(3).astype(np.float64))
    # account for sign ambiguity: n and -n represent same plane
    cosang = float(np.clip(abs(n1 @ n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def rmse_point_to_plane(points: np.ndarray, n: np.ndarray, d: float) -> float:
    if len(points) == 0:
        return float("nan")
    n = _safe_unit(n.reshape(3).astype(np.float64))
    dist = points @ n + float(d)
    return float(np.sqrt(np.mean(dist * dist)))


def compute_metrics(
    n_gt: np.ndarray,
    d_gt: float,
    n_pred: np.ndarray,
    d_pred: float,
    points_metric: np.ndarray,
) -> PlaneMetrics:
    ang = plane_normal_angle_deg(n_gt, n_pred)
    offset = float(abs(float(d_gt) - float(d_pred)))
    rmse = rmse_point_to_plane(points_metric, n_gt, d_gt)
    return PlaneMetrics(
        normal_angle_deg=ang,
        plane_offset_abs_m=offset,
        rmse_point_to_gt_plane_m=rmse,
    )
