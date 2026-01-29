#!/usr/bin/env python3
"""
Example usage of the diffusion-based depth estimation system.

This file demonstrates how to use the implemented components
for both training and inference of depth estimation models.
"""

import torch
import torch.nn as nn
from main import DepthDiffuser, DepthEstimator, train_step, infer, evaluate
from utils import generate_noise_schedule, load_sample_data, calculate_metrics
import argparse

def demo_training():
    """Demonstrate the training process"""
    print("=== Training Demo ===")
    
    # Initialize components
    diffuser = DepthDiffuser()
    model = DepthEstimator()
    
    # Create dummy data
    batch_size = 2
    images = torch.rand(batch_size, 3, 256, 256)
    depths = torch.rand(batch_size, 1, 256, 256)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Perform one training step
    loss = train_step(diffuser, model, images, depths, optimizer, device='cpu')
    
    print(f"Training step completed. Loss: {loss:.6f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
def demo_inference():
    """Demonstrate the inference process"""
    print("\n=== Inference Demo ===")
    
    # Initialize components
    diffuser = DepthDiffuser()
    model = DepthEstimator()
    
    # Create a sample image for inference
    sample_image = torch.rand(1, 3, 256, 256)
    
    # Perform inference
    result = infer(diffuser, model, sample_image, device='cpu', num_steps=50)
    
    print(f"Inference completed.")
    print(f"Input shape: {sample_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

def demo_metrics():
    """Demonstrate evaluation metrics calculation"""
    print("\n=== Metrics Demo ===")
    
    # Create sample predictions and ground truth
    pred_depth = torch.rand(1, 1, 256, 256)
    true_depth = torch.rand(1, 1, 256, 256)
    
    # Calculate metrics
    metrics = calculate_metrics(pred_depth, true_depth)
    
    print("Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")

def demo_noise_schedules():
    """Demonstrate different noise schedules"""
    print("\n=== Noise Schedule Demo ===")
    
    # Generate different noise schedules
    linear_schedule = generate_noise_schedule(schedule_type='linear')
    cosine_schedule = generate_noise_schedule(schedule_type='cosine')
    
    print(f"Linear schedule shape: {linear_schedule.shape}")
    print(f"Linear schedule range: [{linear_schedule.min():.6f}, {linear_schedule.max():.6f}]")
    print(f"Cosine schedule shape: {cosine_schedule.shape}")
    print(f"Cosine schedule range: [{cosine_schedule.min():.6f}, {cosine_schedule.max():.6f}]")

def main():
    """Run all demos"""
    print("Diffusion-Based Depth Estimation - Example Usage")
    print("=" * 50)
    
    # Run demos
    demo_training()
    demo_inference()
    demo_metrics()
    demo_noise_schedules()
    
    print("\n" + "=" * 50)
    print("Example usage completed successfully!")
    print("\nTo use this system in practice:")
    print("1. Replace dummy data with actual NYUv2 or KITTI dataset")
    print("2. Adjust model architecture for your specific needs")
    print("3. Train on your dataset with appropriate hyperparameters")
    print("4. Evaluate using the provided metrics")

if __name__ == "__main__":
    main()