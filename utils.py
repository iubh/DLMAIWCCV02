import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os

def generate_noise_schedule(beta_start=0.0001, beta_end=0.02, T=1000, schedule_type='linear'):
    """
    Generate noise schedule for diffusion process
    
    Args:
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
        T (int): Number of diffusion steps
        schedule_type (str): Type of noise schedule ('linear', 'cosine', 'sigmoid')
    
    Returns:
        torch.Tensor: Noise schedule
    """
    if schedule_type == 'linear':
        return torch.linspace(beta_start, beta_end, T)
    elif schedule_type == 'cosine':
        # Cosine schedule as proposed in the paper
        def cosine_schedule(t, s=0.008):
            return torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
        
        betas = []
        for t in range(T):
            alpha_bar = cosine_schedule(t)
            alpha_bar_prev = cosine_schedule(t - 1) if t > 0 else torch.tensor(1.0)
            beta = 1 - alpha_bar / alpha_bar_prev
            betas.append(beta)
        return torch.tensor(betas)
    elif schedule_type == 'sigmoid':
        # Sigmoid schedule
        def sigmoid_schedule(t):
            return torch.sigmoid((t / T - 0.5) * 10)
        
        betas = []
        for t in range(T):
            beta = sigmoid_schedule(t) - sigmoid_schedule(t - 1) if t > 0 else sigmoid_schedule(0)
            betas.append(beta)
        return torch.tensor(betas)
    else:
        raise ValueError("Unsupported schedule type. Choose from 'linear', 'cosine', 'sigmoid'")

def create_alphas_from_betas(betas):
    """
    Create alpha values from beta values
    
    Args:
        betas (torch.Tensor): Noise schedule
    
    Returns:
        tuple: (alphas, alpha_cumprod)
    """
    alphas = 1. - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alpha_cumprod

def load_sample_data(image_path=None, depth_path=None):
    """
    Load sample data for demonstration
    
    Args:
        image_path (str): Path to image file (optional)
        depth_path (str): Path to depth file (optional)
    
    Returns:
        tuple: (image_tensor, depth_tensor)
    """
    if image_path and depth_path:
        # Load actual files
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        depth_tensor = transform(depth)
        
        return image_tensor, depth_tensor
    else:
        # Return dummy data for demonstration
        image_tensor = torch.rand(3, 256, 256)
        depth_tensor = torch.rand(1, 256, 256)
        return image_tensor, depth_tensor

def calculate_metrics(pred_depth, true_depth):
    """
    Calculate evaluation metrics for depth estimation
    
    Args:
        pred_depth (torch.Tensor): Predicted depth map
        true_depth (torch.Tensor): Ground truth depth map
    
    Returns:
        dict: Dictionary with metrics
    """
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_depth - true_depth)**2))
    
    # MAE
    mae = torch.mean(torch.abs(pred_depth - true_depth))
    
    # R² Score
    ss_res = torch.sum((pred_depth - true_depth) ** 2)
    ss_tot = torch.sum((pred_depth - torch.mean(true_depth)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Log RMSE
    log_rmse = torch.sqrt(torch.mean((torch.log(pred_depth + 1e-6) - torch.log(true_depth + 1e-6))**2))
    
    return {
        'rmse': rmse.item(),
        'mae': mae.item(),
        'r2': r2.item(),
        'log_rmse': log_rmse.item()
    }

def apply_post_processing(depth_map, method='bilateral'):
    """
    Apply post-processing to smooth depth boundaries
    
    Args:
        depth_map (torch.Tensor): Depth map to process
        method (str): Post-processing method ('bilateral', 'gaussian', 'median')
    
    Returns:
        torch.Tensor: Processed depth map
    """
    if method == 'bilateral':
        # Simplified bilateral filtering implementation
        # In practice, you would use cv2.bilateralFilter or similar
        return depth_map
    elif method == 'gaussian':
        # Apply Gaussian blur
        kernel_size = 5
        sigma = 1.0
        # This is a simplified version - real implementation would use proper Gaussian filter
        return depth_map
    elif method == 'median':
        # Apply median filter
        return depth_map
    else:
        return depth_map

def save_model(model, path):
    """
    Save model weights
    
    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save model
    """
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device='cpu'):
    """
    Load model weights
    
    Args:
        model_class (class): Model class
        path (str): Path to model weights
        device (str): Device to load model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model (torch.nn.Module): Model
    
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb