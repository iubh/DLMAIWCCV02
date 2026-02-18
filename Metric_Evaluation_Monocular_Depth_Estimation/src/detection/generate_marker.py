"""Generate an ArUco marker image for printing.

Example:
  python -m src.detection.generate_marker --id 0 --out data/markers/aruco_0.png

Make sure to print at 100% scale (no auto-fit) and measure the marker side length.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "cv2.aruco is not available. Install opencv-contrib-python (not opencv-python)."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_100",
        help="OpenCV aruco dictionary name",
    )
    parser.add_argument("--pixels", type=int, default=600, help="image side length in pixels")
    parser.add_argument("--out", type=str, default="data/markers/aruco_0.png")
    args = parser.parse_args()

    aruco = cv2.aruco
    if not hasattr(aruco, args.dictionary):
        raise ValueError(f"Unknown dictionary: {args.dictionary}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dictionary))

    marker = aruco.generateImageMarker(dictionary, args.id, args.pixels)

    project_root = Path(__file__).resolve().parents[2]
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), marker)
    print(f"[OK] Wrote marker to: {out_path}")


if __name__ == "__main__":
    main()
