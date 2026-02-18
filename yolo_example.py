#!/usr/bin/env python3
"""
YOLO Object Detection Example (Ultralytics)
Usage:
    python yolo_example.py --image_path path/to/image.jpg
    
    e.g.
    python yolo_example.py \
    --image_path bus.jpg \
    --model_size yolo26s.pt \
    --confidence_threshold 0.3
    
    You can get the model yolo26s.pt, e.g. via https://github.com/ultralytics/ultralytics (tried on 18th of Feb. 2026)
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import torch

# -------------------------------
# Load YOLO Model
# -------------------------------
def load_yolo_model(model_size="yolo26s.pt", device="cpu"):
    """
    model_size options:
        yolo26s.pt  (small, fast, strong)
        yolo26m.pt  (medium, higher accuracy)
    """
    model = YOLO(model_size)
    model.to(device)
    print(f"Loaded YOLOv11 model: {model_size} on {device}")
    return model


# -------------------------------
# Run Inference
# -------------------------------
def detect_objects(model, image_path, confidence_threshold=0.25):
    results = model(
        source=image_path,
        conf=confidence_threshold,
        verbose=False
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls,
            "class_name": model.names[cls]
        })

    return detections


# -------------------------------
# Draw Results
# -------------------------------
def draw_detections(image_path, detections):
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="yolo26s.pt",
                        help="yolo26s.pt or yolo26m.pt")
    parser.add_argument("--confidence_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model = load_yolo_model(args.model_size, args.device)
    detections = detect_objects(model, args.image_path, args.confidence_threshold)

    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")

    result_img = draw_detections(args.image_path, detections)

    output_path = Path(args.image_path).parent / f"detected_{Path(args.image_path).name}"
    cv2.imwrite(str(output_path), result_img)

    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
