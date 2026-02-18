from __future__ import annotations

from .base import DepthModelBase
from .midas_backend import MiDaSBackend
from .depth_anything_v2_backend import DepthAnythingV2Backend


def create_depth_backend(name: str, device: str | None = None) -> DepthModelBase:
    name = name.lower().strip()
    if name == "midas":
        return MiDaSBackend(device=device)
    if name in ("depth_anything_v2", "dav2", "depthanythingv2"):
        return DepthAnythingV2Backend(device=device)
    raise ValueError(f"Unknown depth backend: {name}")
