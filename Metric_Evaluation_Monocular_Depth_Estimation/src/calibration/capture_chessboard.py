"""Capture chessboard images for calibration.

Saves images to: data/calibration/chessboard_images/

Controls:
  - s: save current frame
  - q: quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--pattern_cols", type=int, default=9, help="inner corners")
    parser.add_argument("--pattern_rows", type=int, default=6, help="inner corners")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/calibration/chessboard_images",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("[INFO] Press 's' to save a frame, 'q' to quit")
    idx = 0
    pattern = (args.pattern_cols, args.pattern_rows)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)

            cv2.imshow("Calibration Capture", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                out_path = output_dir / f"chess_{idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"[SAVED] {out_path}")
                idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
