from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Plane:
    n: np.ndarray  # (3,) unit normal
    d: float       # offset in n^T X + d = 0


def fit_plane_svd(points: np.ndarray) -> Plane:
    """Fit plane to 3D points via SVD.

    Returns plane in form n^T X + d = 0 with ||n||=1.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if len(points) < 3:
        raise ValueError("need at least 3 points")

    centroid = points.mean(axis=0)
    Q = points - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(-n.T @ centroid)
    return Plane(n=n.astype(np.float64), d=d)


def compute_scale_ls(points_rel: np.ndarray, n_gt: np.ndarray, d_gt: float) -> float:
    """Compute scale s that best aligns scaled points to GT plane.

    Minimize sum_i ( n_gt^T (s P_i) + d_gt )^2.

    Closed form:
      Let a_i = n_gt^T P_i.
      Minimize sum (s a_i + d_gt)^2 -> derivative = 2 sum a_i (s a_i + d_gt) = 0
      s = - d_gt * sum a_i / sum a_i^2
    """

    if len(points_rel) == 0:
        raise ValueError("no points")
    n_gt = n_gt.astype(np.float64).reshape(3)
    a = points_rel @ n_gt
    denom = float(np.sum(a * a))
    if denom < 1e-12:
        raise ValueError("degenerate configuration: denom ~ 0")
    s = -float(d_gt) * float(np.sum(a)) / denom
    return float(s)


def align_points_to_plane(points_rel: np.ndarray, n_gt: np.ndarray, d_gt: float) -> Tuple[np.ndarray, float]:
    s = compute_scale_ls(points_rel, n_gt, d_gt)
    return points_rel * s, s
