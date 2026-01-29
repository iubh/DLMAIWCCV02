#!/usr/bin/env python3
"""
YOLO-based Object Detection Example

This script demonstrates how to use YOLO (You Only Look Once) object detection
for computer vision tasks. It loads a pre-trained YOLO model and performs
object detection on input images.

Requirements:
- torch>=1.8.0
- torchvision>=0.9.0
- opencv-python
- numpy

Usage:
    python yolo_example.py --image_path path/to/image.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

def load_yolo_model(model_size='yolov5s'):
    """
    Load a pre-trained YOLO model
    
    Args:
        model_size (str): Size of YOLO model ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
    
    Returns:
        torch.nn.Module: Loaded YOLO model
    """
    try:
        # Try to load YOLOv5 from torch.hub
        model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=True)
        print(f"Loaded YOLOv5 {model_size} model successfully")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        print("Please install yolov5: pip install ultralytics")
        return None

def detect_objects(model, image_path, confidence_threshold=0.25):
    """
    Perform object detection on an image
    
    Args:
        model (torch.nn.Module): YOLO model
        image_path (str): Path to input image
        confidence_threshold (float): Minimum confidence for detection
    
    Returns:
        list: List of detected objects with bounding boxes and labels
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Perform detection
    results = model(img)
    
    # Process results
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf >= confidence_threshold:
            detections.append({
                'bbox': box,  # [x1, y1, x2, y2]
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': results.names[int(cls)]
            })
    
    return detections

def draw_detections(image_path, detections):
    """
    Draw bounding boxes and labels on the image
    
    Args:
        image_path (str): Path to input image
        detections (list): List of detected objects
    
    Returns:
        numpy.ndarray: Image with detections drawn
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Draw detections
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{detection['class_name']}: {detection['confidence']:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection Example')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image for object detection')
    parser.add_argument('--model_size', type=str, default='yolov5s',
                       help='YOLO model size (yolov5s, yolov5m, yolov5l, yolov5x)')
    parser.add_argument('--confidence_threshold', type=float, default=0.25,
                       help='Minimum confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Load YOLO model
    model = load_yolo_model(args.model_size)
    if model is None:
        return
    
    try:
        # Perform object detection
        detections = detect_objects(model, args.image_path, args.confidence_threshold)
        
        print(f"Detected {len(detections)} objects:")
        for detection in detections:
            print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")
        
        # Draw detections on image
        result_image = draw_detections(args.image_path, detections)
        
        # Save result
        output_path = Path(args.image_path).parent / f"detected_{Path(args.image_path).name}"
        cv2.imwrite(str(output_path), result_image)
        print(f"Result saved to {output_path}")
        
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    main()