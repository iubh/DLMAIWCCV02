from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import DepthModelBase, DepthPrediction


class DepthAnythingV2Backend(DepthModelBase):
    """Optional backend placeholder for Depth Anything V2.

    This project intentionally keeps DA-v2 integration *pluggable*.
    The official Depth Anything V2 code/weights can be added under:
      models/depth_anything_v2/

    In practice, you would:
      - vendor the official repo code, or install it as a dependency
      - load weights
      - implement predict() returning a (H,W) depth map

    For now this backend raises a clear error with instructions.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | None = None):
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = device

    def predict(self, bgr: np.ndarray) -> DepthPrediction:
        raise NotImplementedError(
            "DepthAnythingV2Backend is a stub.\n"
            "Please integrate the official Depth Anything V2 implementation and weights, "
            "then implement predict().\n"
            "Alternatively run with --depth_backend midas (default)."
        )
