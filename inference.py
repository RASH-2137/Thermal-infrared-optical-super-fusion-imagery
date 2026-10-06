"""
Thermal Infrared Super-Resolution Inference Script

This script provides:
- Command-line inference on thermal/optical image pairs
- Demo mode with auto-generated synthetic data
- Reusable run_inference() function for API integration
"""

import torch
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from model import ThermalSuperResolutionNet
from preprocess import ThermalDataProcessor

# ---------------- CONFIG ----------------
MODEL_PATH = "checkpoints/best_model.pth"
DEMO_DATA_DIR = "demo_data"
OUTPUT_DIR = "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------


def load_model(model_path=MODEL_PATH, device=DEVICE):
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Torch device (CPU/GPU)
    
    Returns:
        model: Loaded model in eval mode
        error: Error message if loading failed, None otherwise
    """
    if not os.path.exists(model_path):
        return None, f"Model not found at {model_path}"
    
    try:
        model = ThermalSuperResolutionNet(
            thermal_channels=1,
            optical_channels=3,
            base_channels=64,
            scale_factor=4
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def generate_demo_data(output_dir=DEMO_DATA_DIR):
    """
    Generate synthetic thermal and optical images for demo purposes.
    
    Args:
        output_dir: Directory to save demo images
    
    Returns:
        thermal_path, optical_path: Paths to generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    thermal_path = os.path.join(output_dir, "sample_thermal.png")
    optical_path = os.path.join(output_dir, "sample_optical.png")
    
    # Generate synthetic thermal image (temperature-like patterns)
    size = 256
    thermal = np.zeros((size, size), dtype=np.float32)
    
    # Create temperature gradient and patterns
    y, x = np.ogrid[:size, :size]
    center_x, center_y = size // 2, size // 2
    
    # Radial gradient (hot center)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    thermal += 200 * np.exp(-r / 80)
    
    # Add some structure (hot spots)
    thermal += 50 * np.sin(x * 0.1) * np.cos(y * 0.1)
    thermal += 30 * np.sin(x * 0.05) * np.cos(y * 0.05)
    
    # Add noise
    thermal += np.random.randn(size, size) * 10
    
    # Normalize to 0-255
    thermal = np.clip(thermal, 0, 255).astype(np.uint8)
    
    # Generate synthetic optical image (RGB)
    optical = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create colorful patterns
    optical[:, :, 0] = 100 + 50 * np.sin(x * 0.05) * np.cos(y * 0.05)  # Red
    optical[:, :, 1] = 120 + 40 * np.sin(x * 0.07) * np.cos(y * 0.07)  # Green
    optical[:, :, 2] = 80 + 60 * np.sin(x * 0.09) * np.cos(y * 0.09)  # Blue
    
    # Add some structure matching thermal
    mask = thermal > 150
    optical[mask, 0] = np.clip(optical[mask, 0] + 50, 0, 255)
    optical[mask, 1] = np.clip(optical[mask, 1] + 30, 0, 255)
    
    # Apply blur for realism
    thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
    optical = cv2.GaussianBlur(optical, (3, 3), 0)
    
    # Save images
    cv2.imwrite(thermal_path, thermal)
    cv2.imwrite(optical_path, optical)
    
    print(f"✅ Generated demo data:")
    print(f"   Thermal: {thermal_path}")
    print(f"   Optical: {optical_path}")
    
    return thermal_path, optical_path


def run_inference(thermal_path, optical_path, output_path=None, model_path=MODEL_PATH, 
                 device=DEVICE, align=True, visualize=False):
    """
    Run super-resolution inference on thermal/optical image pair.
    
    Args:
        thermal_path: Path to thermal image
        optical_path: Path to optical image
        output_path: Path to save output (default: outputs/sr_thermal.png)
        model_path: Path to model checkpoint
        device: Torch device
        align: Whether to align images
        visualize: Whether to create side-by-side visualization
    
    Returns:
        output_path: Path to saved output image
        error: Error message if failed, None otherwise
    """
    # Default output path
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "sr_thermal.png")
    
    # Load model
    model, error = load_model(model_path, device)
    if error:
        return None, error
    
    # Load images
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    optical = cv2.imread(optical_path, cv2.IMREAD_COLOR)
    
    if thermal is None:
        return None, f"Could not load thermal image from {thermal_path}"
    if optical is None:
        return None, f"Could not load optical image from {optical_path}"
    
    # Process images
    processor = ThermalDataProcessor(target_size=(256, 256))
    lr_thermal, hr_optical, hr_thermal = processor.create_lr_hr_pair(
        thermal, optical, scale_factor=4, align=align
    )
    
    if lr_thermal is None or hr_optical is None:
        return None, "Failed to process images. Check if images are valid."
    
    # Convert to tensors
    lr_thermal_tensor = torch.tensor(lr_thermal).unsqueeze(0).unsqueeze(0).float().to(device)
    hr_optical_tensor = torch.tensor(hr_optical).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Run inference
    with torch.no_grad():
        sr = model(lr_thermal_tensor, hr_optical_tensor)
        sr = torch.clamp(sr, 0, 1)
    
    # Save output
    sr_img = (sr.squeeze().cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(output_path, sr_img)
    
    # Create visualization if requested
    if visualize:
        vis_path = output_path.replace('.png', '_comparison.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Low-res thermal
        axes[0].imshow(lr_thermal, cmap='hot')
        axes[0].set_title('Low-Resolution Thermal', fontsize=12)
        axes[0].axis('off')
        
        # Super-resolved thermal
        axes[1].imshow(sr_img, cmap='hot')
        axes[1].set_title('Super-Resolved Thermal (4x)', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Comparison saved to {vis_path}")
    
    return output_path, None


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Thermal Super-Resolution Inference')
    parser.add_argument('--thermal', type=str, help='Path to thermal image')
    parser.add_argument('--optical', type=str, help='Path to optical image')
    parser.add_argument('--output', type=str, help='Output path (default: outputs/sr_thermal.png)')
    parser.add_argument('--demo', action='store_true', help='Run with auto-generated demo data')
    parser.add_argument('--no-align', action='store_true', help='Disable image alignment')
    parser.add_argument('--visualize', action='store_true', help='Create side-by-side comparison')
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo:
        print("🎮 Running in DEMO mode...")
        print("📦 Generating synthetic demo data...")
        thermal_path, optical_path = generate_demo_data()
    else:
        # Use provided paths or defaults
        thermal_path = args.thermal or "data/thermal/sample_thermal.png"
        optical_path = args.optical or "data/optical/sample_optical.png"
        
        # Check if files exist
        if not os.path.exists(thermal_path):
            print(f"❌ Thermal image not found: {thermal_path}")
            print("💡 Tip: Use --demo to generate synthetic data")
            return
        
        if not os.path.exists(optical_path):
            print(f"❌ Optical image not found: {optical_path}")
            print("💡 Tip: Use --demo to generate synthetic data")
            return
    
    print(f"📦 Loading model from {MODEL_PATH}...")
    print(f"🖼️  Processing images...")
    print(f"   Thermal: {thermal_path}")
    print(f"   Optical: {optical_path}")
    
    # Run inference
    output_path, error = run_inference(
        thermal_path=thermal_path,
        optical_path=optical_path,
        output_path=args.output,
        align=not args.no_align,
        visualize=args.visualize
    )
    
    if error:
        print(f"❌ Error: {error}")
        return
    
    print(f"✅ Inference complete!")
    print(f"💾 Output saved to {output_path}")


if __name__ == "__main__":
    main()
