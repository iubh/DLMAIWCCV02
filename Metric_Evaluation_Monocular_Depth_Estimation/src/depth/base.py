from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthPrediction:
    depth: np.ndarray  # (H,W) float32/float64
    meta: dict


class DepthModelBase(ABC):
    """Minimal interface for pluggable monocular depth estimators."""

    @abstractmethod
    def predict(self, bgr: np.ndarray) -> DepthPrediction:
        """Return a depth map for the input BGR image.

        Expected output:
          - depth is relative (unknown scale) unless backend is metric.
        """


def normalize_depth_to_positive(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize an arbitrary depth-like output to positive values.

    Some models return inverse depth or unnormalized values.
    This function enforces positivity and roughly normalizes the range.
    """

    d = depth.astype(np.float64)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # Shift to >=0
    d = d - float(np.min(d))
    # Avoid all-zero
    mx = float(np.max(d))
    if mx < eps:
        return np.ones_like(d, dtype=np.float64)
    d = d / mx
    # Keep away from 0 (backprojection requires >0)
    d = d + eps
    return d
