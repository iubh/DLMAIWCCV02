from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_camera_matrix(K: np.ndarray) -> "Intrinsics":
        return Intrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )


def backproject_depth_to_points(
    depth: np.ndarray,
    intr: Intrinsics,
    mask: np.ndarray | None = None,
    max_points: int = 20000,
) -> np.ndarray:
    """Backproject depth map to 3D points in camera coordinates.

    depth: (H,W) float depth (relative or metric)
    mask: (H,W) uint8/bool mask. If given, only masked pixels are used.

    Returns:
      points: (N,3)
    """

    if depth.ndim != 2:
        raise ValueError("depth must be (H,W)")

    H, W = depth.shape
    if mask is None:
        ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    else:
        m = mask.astype(bool)
        ys, xs = np.where(m & np.isfinite(depth) & (depth > 0))

    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # subsample to keep things fast
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    Z = depth[ys, xs].astype(np.float64)
    X = (xs.astype(np.float64) - intr.cx) / intr.fx * Z
    Y = (ys.astype(np.float64) - intr.cy) / intr.fy * Z
    pts = np.stack([X, Y, Z], axis=1)
    return pts
