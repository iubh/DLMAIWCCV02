from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class CaptureResult:
    path: Path
    width: int
    height: int


def capture_one_frame(
    output_path: str | Path,
    camera_index: int = 0,
    window_name: str = "Capture (press SPACE to save, q to quit)",
    requested_width: Optional[int] = None,
    requested_height: Optional[int] = None,
) -> Optional[CaptureResult]:
    """Open webcam and capture exactly one frame.

    Controls:
      - SPACE: save current frame to output_path and exit
      - q: exit without saving
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index={camera_index}).")

    if requested_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    if requested_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read from webcam.")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return None
            if key == 32:  # SPACE
                ok = cv2.imwrite(str(output_path), frame)
                if not ok:
                    raise RuntimeError(f"Could not write image to {output_path}")
                h, w = frame.shape[:2]
                return CaptureResult(path=output_path, width=w, height=h)
    finally:
        cap.release()
        cv2.destroyAllWindows()
