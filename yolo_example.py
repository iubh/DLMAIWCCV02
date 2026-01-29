#!/usr/bin/env python3
"""
YOLO-based Object Detection Example
Usage:
    python yolo_example.py --image_path path/to/image.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

def load_yolo_model(model_size='yolov5s', device='cpu'):
    """Load a pre-trained YOLOv5 model."""
    try:
        # Load YOLOv5 from torch.hub
        model = torch.hub.load('ultralytics/yolov5:v6.1', model_size, pretrained=True)
        model.to(device)
        model.eval()
        print(f"Loaded YOLOv5 {model_size} model on {device}")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        print("Please install ultralytics YOLOv5: pip install ultralytics")
        return None

def detect_objects(model, image_path, confidence_threshold=0.25):
    """Perform object detection on an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    results = model(img)

    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf >= confidence_threshold:
            detections.append({
                'bbox': box,              # [x1, y1, x2, y2]
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': results.names[int(cls)]
            })

    return detections

def draw_detections(image_path, detections):
    """Draw bounding boxes and labels on the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection Example')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_size', type=str, default='yolov5s',
                        help='YOLO model size: yolov5s, yolov5m, yolov5l, yolov5x')
    parser.add_argument('--confidence_threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model')
    args = parser.parse_args()

    model = load_yolo_model(args.model_size, device=args.device)
    if model is None:
        return

    try:
        detections = detect_objects(model, args.image_path, args.confidence_threshold)
        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

        result_img = draw_detections(args.image_path, detections)
        output_path = Path(args.image_path).parent / f"detected_{Path(args.image_path).name}"
        cv2.imwrite(str(output_path), result_img)
        print(f"Result saved to {output_path}")

    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    main()
