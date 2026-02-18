from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .base import DepthModelBase, DepthPrediction, normalize_depth_to_positive


class MiDaSBackend(DepthModelBase):
    """Out-of-the-box depth backend using MiDaS via torch.hub.

    Pros: easy to run.
    Cons: returns relative depth only.
    """

    def __init__(self, model_type: str = "DPT_Hybrid", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Download on first use (cached by torch)
        # NOTE: Newer torch versions may require explicit trust_repo=True.
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray) -> DepthPrediction:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float64)
        depth = normalize_depth_to_positive(depth)
        return DepthPrediction(depth=depth, meta={"backend": "midas", "model_type": self.model_type})
