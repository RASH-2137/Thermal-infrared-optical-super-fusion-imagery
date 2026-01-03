"""
Thermal Infrared Super-Resolution Data Preprocessing Module

This module handles:
- Geometric alignment between thermal and optical images
- Image normalization and resizing
- Data augmentation for training
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform, exposure
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import logging
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. SSL4EO preprocessing will not work.")
from tqdm import tqdm


class ImageAligner:
    """Handles geometric alignment between thermal and optical images"""
    
    def __init__(self, method: str = 'SIFT'):
        """
        Initialize aligner with specified method
        
        Args:
            method: 'SIFT', 'ORB', or 'template_matching'
        """
        self.method = method
        if method == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher()
        elif method == 'ORB':
            self.detector = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.detector = None
            self.matcher = None
    
    def align_images(self, thermal: np.ndarray, optical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align thermal and optical images
        
        Args:
            thermal: Thermal image (H, W) or (H, W, C)
            optical: Optical image (H, W) or (H, W, C)
            
        Returns:
            aligned_thermal, aligned_optical, transformation_matrix
        """
        if len(thermal.shape) == 3:
            thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY)
        else:
            thermal_gray = thermal.copy()
            
        if len(optical.shape) == 3:
            optical_gray = cv2.cvtColor(optical, cv2.COLOR_RGB2GRAY)
        else:
            optical_gray = optical.copy()
        
        # Ensure images are uint8
        thermal_gray = (thermal_gray * 255).astype(np.uint8) if thermal_gray.max() <= 1 else thermal_gray.astype(np.uint8)
        optical_gray = (optical_gray * 255).astype(np.uint8) if optical_gray.max() <= 1 else optical_gray.astype(np.uint8)
        
        if self.method in ['SIFT', 'ORB']:
            return self._feature_based_alignment(thermal_gray, optical_gray, thermal, optical)
        else:
            return self._template_matching_alignment(thermal_gray, optical_gray, thermal, optical)
    
    def _feature_based_alignment(self, thermal_gray: np.ndarray, optical_gray: np.ndarray, 
                                thermal: np.ndarray, optical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Feature-based alignment using SIFT/ORB"""
        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(thermal_gray, None)
        kp2, des2 = self.detector.detectAndCompute(optical_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("Warning: Not enough features found, returning original images")
            return thermal, optical, np.eye(3)
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep only good matches
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 4:
            print("Warning: Not enough good matches, returning original images")
            return thermal, optical, np.eye(3)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Warning: Could not find homography, returning original images")
            return thermal, optical, np.eye(3)
        
        # Apply transformation
        h, w = optical.shape[:2]
        aligned_thermal = cv2.warpPerspective(thermal, H, (w, h))
        
        return aligned_thermal, optical, H
    
    def _template_matching_alignment(self, thermal_gray: np.ndarray, optical_gray: np.ndarray,
                                   thermal: np.ndarray, optical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Template matching alignment"""
        # Resize thermal to match optical size
        h, w = optical_gray.shape
        thermal_resized = cv2.resize(thermal_gray, (w, h))
        
        # Simple template matching
        result = cv2.matchTemplate(optical_gray, thermal_resized, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        # Create transformation matrix
        H = np.eye(3)
        H[0, 2] = max_loc[0] - w//2
        H[1, 2] = max_loc[1] - h//2
        
        # Apply transformation
        aligned_thermal = cv2.warpAffine(thermal, H[:2], (w, h))
        
        return aligned_thermal, optical, H


class ThermalDataProcessor:
    """Processes thermal and optical images for training/inference"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256), 
                 thermal_range: Tuple[float, float] = (0.0, 1.0),
                 optical_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize processor
        
        Args:
            target_size: Target image size (height, width)
            thermal_range: Normalization range for thermal images
            optical_range: Normalization range for optical images
        """
        self.target_size = target_size
        self.thermal_range = thermal_range
        self.optical_range = optical_range
        self.aligner = ImageAligner()
    
    def normalize_thermal(self, thermal: np.ndarray) -> Optional[np.ndarray]:
        """Normalize thermal image to specified range with safety checks"""
        if thermal is None:
            return None
        
        # Check for NaN or Inf
        if np.any(np.isnan(thermal)) or np.any(np.isinf(thermal)):
            return None
        
        thermal_min = thermal.min()
        thermal_max = thermal.max()
        thermal_range = thermal_max - thermal_min
        
        # Check if image is flat or invalid
        if thermal_range < 1e-6:
            return None
        
        # Safe normalization
        thermal_norm = (thermal - thermal_min) / thermal_range
        thermal_norm = thermal_norm * (self.thermal_range[1] - self.thermal_range[0]) + self.thermal_range[0]
        thermal_norm = thermal_norm.astype(np.float32)
        
        # Final check for validity
        if np.any(np.isnan(thermal_norm)) or np.any(np.isinf(thermal_norm)):
            return None
        
        return thermal_norm
    
    def normalize_optical(self, optical: np.ndarray) -> Optional[np.ndarray]:
        """Normalize optical image to specified range with safety checks"""
        if optical is None:
            return None
        
        # Check for NaN or Inf
        if np.any(np.isnan(optical)) or np.any(np.isinf(optical)):
            return None
        
        optical_min = optical.min()
        optical_max = optical.max()
        optical_range = optical_max - optical_min
        
        # Check if image is flat or invalid
        if optical_range < 1e-6:
            return None
        
        # Safe normalization
        optical_norm = (optical - optical_min) / optical_range
        optical_norm = optical_norm * (self.optical_range[1] - self.optical_range[0]) + self.optical_range[0]
        optical_norm = optical_norm.astype(np.float32)
        
        # Final check for validity
        if np.any(np.isnan(optical_norm)) or np.any(np.isinf(optical_norm)):
            return None
        
        return optical_norm
    
    def process_pair(self, thermal: np.ndarray, optical: np.ndarray, 
                    align: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process thermal-optical image pair with safety checks
        
        Args:
            thermal: Thermal image
            optical: Optical image
            align: Whether to perform alignment
            
        Returns:
            processed_thermal, processed_optical (may be None if invalid)
        """
        if thermal is None or optical is None:
            return None, None
        
        # Align images if requested
        if align:
            thermal, optical, _ = self.aligner.align_images(thermal, optical)
        
        # Resize to target size
        thermal = cv2.resize(thermal, (self.target_size[1], self.target_size[0]))
        optical = cv2.resize(optical, (self.target_size[1], self.target_size[0]))
        
        # Normalize (may return None if invalid)
        thermal = self.normalize_thermal(thermal)
        optical = self.normalize_optical(optical)
        
        return thermal, optical
    
    def create_lr_hr_pair(self, thermal: np.ndarray, optical: np.ndarray, 
                         scale_factor: int = 4, align: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create low-resolution thermal, high-resolution optical, and high-resolution thermal pair
        
        Args:
            thermal: High-resolution thermal image
            optical: High-resolution optical image
            scale_factor: Downsampling factor for thermal
            align: Whether to perform geometric alignment (default: True)
            
        Returns:
            lr_thermal, hr_optical, hr_thermal (may be None if invalid)
        """
        # Process the pair
        hr_thermal, hr_optical = self.process_pair(thermal, optical, align=align)
        
        # Check if processing failed
        if hr_thermal is None or hr_optical is None:
            return None, None, None
        
        # Create low-resolution thermal
        h, w = hr_thermal.shape[:2]
        lr_h, lr_w = h // scale_factor, w // scale_factor
        lr_thermal = cv2.resize(hr_thermal, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # Final validity check
        if np.any(np.isnan(lr_thermal)) or np.any(np.isinf(lr_thermal)):
            return None, None, None
        
        return lr_thermal, hr_optical, hr_thermal


class ThermalDataset(Dataset):
    """PyTorch dataset for thermal super-resolution"""
    
    def __init__(self, thermal_paths: list, optical_paths: list, 
                 processor: ThermalDataProcessor, scale_factor: int = 4,
                 augment: bool = True, align: bool = False):
        """
        Initialize dataset
        
        Args:
            thermal_paths: List of thermal image paths
            optical_paths: List of optical image paths
            processor: Data processor
            scale_factor: Super-resolution scale factor
            augment: Whether to apply data augmentation
            align: Whether to perform geometric alignment (default: False for training speed)
        """
        self.thermal_paths = thermal_paths
        self.optical_paths = optical_paths
        self.processor = processor
        self.scale_factor = scale_factor
        self.augment = augment
        self.align = align
        self.skipped_samples = 0

    
    def __len__(self):
        return len(self.thermal_paths)
    
    def __getitem__(self, idx):
        # Load images
        thermal = cv2.imread(self.thermal_paths[idx], cv2.IMREAD_GRAYSCALE)
        optical = cv2.imread(self.optical_paths[idx], cv2.IMREAD_COLOR)
        
        if thermal is None or optical is None:
            self.skipped_samples += 1
            return None
        
        # Process pair
        lr_thermal, hr_optical, hr_thermal = self.processor.create_lr_hr_pair(
            thermal, optical, self.scale_factor, align=self.align
        )
        
        # Check if processing failed
        if lr_thermal is None or hr_optical is None or hr_thermal is None:
            self.skipped_samples += 1
            return None
        
        # Convert to tensors
        lr_thermal = torch.from_numpy(lr_thermal).unsqueeze(0).float()  # Add channel dimension, ensure float32
        hr_optical = torch.from_numpy(hr_optical).permute(2, 0, 1).float()  # HWC to CHW, ensure float32
        hr_thermal = torch.from_numpy(hr_thermal).unsqueeze(0).float()  # Ensure float32
        
        # Check for NaN or Inf in tensors
        if (torch.any(torch.isnan(lr_thermal)) or torch.any(torch.isinf(lr_thermal)) or
            torch.any(torch.isnan(hr_optical)) or torch.any(torch.isinf(hr_optical)) or
            torch.any(torch.isnan(hr_thermal)) or torch.any(torch.isinf(hr_thermal))):
            self.skipped_samples += 1
            return None
        
        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            lr_thermal, hr_optical, hr_thermal = self._augment(lr_thermal, hr_optical, hr_thermal)
            
            # Check again after augmentation
            if (torch.any(torch.isnan(lr_thermal)) or torch.any(torch.isinf(lr_thermal)) or
                torch.any(torch.isnan(hr_optical)) or torch.any(torch.isinf(hr_optical)) or
                torch.any(torch.isnan(hr_thermal)) or torch.any(torch.isinf(hr_thermal))):
                self.skipped_samples += 1
                return None
        
        return {
            'lr_thermal': lr_thermal,
            'hr_optical': hr_optical,
            'hr_thermal': hr_thermal
        }
    
    def _augment(self, lr_thermal: torch.Tensor, hr_optical: torch.Tensor, 
                hr_thermal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            lr_thermal = torch.flip(lr_thermal, dims=[2])
            hr_optical = torch.flip(hr_optical, dims=[2])
            hr_thermal = torch.flip(hr_thermal, dims=[2])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            lr_thermal = torch.flip(lr_thermal, dims=[1])
            hr_optical = torch.flip(hr_optical, dims=[1])
            hr_thermal = torch.flip(hr_thermal, dims=[1])
        
        return lr_thermal, hr_optical, hr_thermal


def create_sample_data(output_dir: str = "sample_data", num_samples: int = 10):
    """
    Create sample thermal and optical images for testing
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of sample pairs to create
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "thermal"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "optical"), exist_ok=True)
    
    for i in range(num_samples):
        # Create synthetic thermal image (temperature-like)
        thermal = np.random.rand(256, 256) * 50 + 273.15  # Kelvin range
        thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
        
        # Create synthetic optical image
        optical = np.random.rand(256, 256, 3) * 255
        optical = cv2.GaussianBlur(optical, (3, 3), 0)
        
        # Add some structure
        y, x = np.ogrid[:256, :256]
        thermal += 10 * np.sin(x * 0.1) * np.cos(y * 0.1)
        optical[:, :, 0] += 20 * np.sin(x * 0.05) * np.cos(y * 0.05)
        optical[:, :, 1] += 20 * np.sin(x * 0.07) * np.cos(y * 0.07)
        optical[:, :, 2] += 20 * np.sin(x * 0.09) * np.cos(y * 0.09)
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, "thermal", f"thermal_{i:03d}.png"), 
                   thermal.astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "optical", f"optical_{i:03d}.png"), 
                   optical.astype(np.uint8))
    
    print(f"Created {num_samples} sample image pairs in {output_dir}")


def process_ssl4eo_landsat_dataset(
    dataset_root: str,
    output_dir: str = "training_data",
    target_size: Tuple[int, int] = (256, 256),
    log_file: str = "preprocessing_log.txt"
) -> dict:
    """
    Fully automated preprocessing pipeline for SSL4EO Landsat dataset.
    
    Processes the SSL4EO Landsat dataset structure:
    - Root folder contains numeric subfolders (0000000, 0000001, ...)
    - Each contains LC08_XXXXXX_YYYYMMDD folders
    - Each contains a multi-band GeoTIFF
    
    Band extraction:
    - Optical RGB: Bands [4, 3, 2] (Landsat band numbers, 1-indexed)
    - Thermal: Band [10] only
    - Band 11 is ignored
    
    Args:
        dataset_root: Root directory containing numeric subfolders
        output_dir: Output directory for processed images (default: training_data)
        target_size: Target image size (height, width)
        log_file: Path to log file for skipped scenes
        
    Returns:
        Dictionary with processing statistics
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for SSL4EO preprocessing. Install with: pip install rasterio")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create output directories
    optical_dir = os.path.join(output_dir, 'optical')
    thermal_dir = os.path.join(output_dir, 'thermal')
    os.makedirs(optical_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total_scenes_found': 0,
        'successfully_processed': 0,
        'skipped_corrupted': 0,
        'skipped_missing_bands': 0,
        'skipped_other_errors': 0
    }
    
    skipped_scenes = []
    
    logger.info(f"Starting SSL4EO Landsat dataset preprocessing")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target size: {target_size}")
    
    # Find all numeric subfolders
    numeric_folders = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path) and item.isdigit():
            numeric_folders.append(item_path)
    
    numeric_folders.sort()
    logger.info(f"Found {len(numeric_folders)} numeric subfolders")
    
    # Process each numeric folder
    sample_index = 0
    
    # Progress bar for numeric folders
    pbar_numeric = tqdm(numeric_folders, desc="Processing numeric folders", unit="folder")
    
    for numeric_folder in pbar_numeric:
        # Find LC08 scene folders
        scene_folders = glob.glob(os.path.join(numeric_folder, "LC08_*"))
        
        for scene_folder in scene_folders:
            stats['total_scenes_found'] += 1
            
            # Find GeoTIFF files in scene folder
            tif_files = glob.glob(os.path.join(scene_folder, "*.tif")) + \
                       glob.glob(os.path.join(scene_folder, "*.TIF")) + \
                       glob.glob(os.path.join(scene_folder, "*.tiff")) + \
                       glob.glob(os.path.join(scene_folder, "*.TIFF"))
            
            if len(tif_files) == 0:
                logger.warning(f"No GeoTIFF found in {scene_folder}")
                stats['skipped_missing_bands'] += 1
                skipped_scenes.append((scene_folder, "No GeoTIFF file found"))
                continue
            
            # Use the first GeoTIFF file found
            tif_path = tif_files[0]
            
            try:
                # Open GeoTIFF with rasterio
                with rasterio.open(tif_path) as src:
                    # Landsat 8 band indices (0-indexed in rasterio)
                    # Band 4 (Red) = index 3, Band 3 (Green) = index 2, Band 2 (Blue) = index 1
                    # Band 10 (Thermal) = index 9
                    # Band 11 is index 10 (we ignore it)
                    
                    # Check if we have enough bands
                    num_bands = src.count
                    if num_bands < 10:
                        logger.warning(f"Insufficient bands ({num_bands}) in {tif_path}. Need at least 10 bands.")
                        stats['skipped_missing_bands'] += 1
                        skipped_scenes.append((tif_path, f"Insufficient bands: {num_bands}"))
                        continue
                    
                    # Read optical bands [4, 3, 2] -> indices [3, 2, 1]
                    # Note: Landsat bands are 1-indexed, rasterio is 1-indexed for read()
                    optical_bands = []
                    band_indices = [4, 3, 2]  # Landsat band numbers (1-indexed)
                    
                    for band_num in band_indices:
                        if band_num > src.count:
                            raise ValueError(f"Band {band_num} not available (only {src.count} bands)")
                        band_data = src.read(band_num)  # rasterio.read() is 1-indexed
                        optical_bands.append(band_data)
                    
                    # Stack to create RGB image: (3, H, W) -> (H, W, 3)
                    optical = np.stack(optical_bands, axis=0)  # (3, H, W)
                    optical = np.transpose(optical, (1, 2, 0))  # (H, W, 3)
                    
                    # Read thermal band [10]
                    if 10 > src.count:
                        raise ValueError(f"Band 10 not available (only {src.count} bands)")
                    thermal = src.read(10)  # Band 10 (rasterio 1-indexed)
                    thermal = thermal.squeeze()  # Remove channel dimension if present
                    
                    # Validate data
                    if optical.shape[0] == 0 or optical.shape[1] == 0:
                        raise ValueError("Empty optical image")
                    if thermal.shape[0] == 0 or thermal.shape[1] == 0:
                        raise ValueError("Empty thermal image")
                    
                    # Check for NaN or Inf values
                    if np.any(np.isnan(optical)) or np.any(np.isinf(optical)):
                        raise ValueError("Optical image contains NaN or Inf")
                    if np.any(np.isnan(thermal)) or np.any(np.isinf(thermal)):
                        raise ValueError("Thermal image contains NaN or Inf")
                    
            except Exception as e:
                logger.error(f"Error reading {tif_path}: {str(e)}")
                stats['skipped_corrupted'] += 1
                skipped_scenes.append((tif_path, f"Read error: {str(e)}"))
                continue
            
            try:
                # Normalize optical to [0, 255]
                # Clip to valid range first (Landsat values typically 0-65535 or 0-10000)
                optical_clipped = np.clip(optical, 0, 10000)  # Typical Landsat range
                if optical_clipped.max() > 0:
                    optical_normalized = (optical_clipped / optical_clipped.max() * 255).astype(np.uint8)
                else:
                    optical_normalized = np.zeros_like(optical_clipped, dtype=np.uint8)
                
                # Ensure 3 channels
                if len(optical_normalized.shape) == 2:
                    optical_normalized = np.stack([optical_normalized] * 3, axis=-1)
                elif optical_normalized.shape[2] != 3:
                    # If more channels, take first 3
                    optical_normalized = optical_normalized[:, :, :3]
                
                # Normalize thermal preserving relative temperature
                # Clip to valid thermal range (Landsat TIRS Band 10: ~-50 to 50°C, typically 0-65535 DN)
                thermal_clipped = np.clip(thermal, 0, 65535)
                # Preserve relative temperature by min-max normalization
                thermal_min = thermal_clipped.min()
                thermal_max = thermal_clipped.max()
                if thermal_max > thermal_min:
                    thermal_normalized = ((thermal_clipped - thermal_min) / (thermal_max - thermal_min) * 255).astype(np.uint8)
                else:
                    # Constant thermal image, set to middle value
                    thermal_normalized = np.full_like(thermal_clipped, 128, dtype=np.uint8)
                
                # Resize to target size
                optical_resized = cv2.resize(optical_normalized, (target_size[1], target_size[0]), 
                                            interpolation=cv2.INTER_CUBIC)
                thermal_resized = cv2.resize(thermal_normalized, (target_size[1], target_size[0]), 
                                           interpolation=cv2.INTER_CUBIC)
                
                # Save with indexed naming
                optical_filename = f"optical_{sample_index:05d}.png"
                thermal_filename = f"thermal_{sample_index:05d}.png"
                
                optical_path = os.path.join(optical_dir, optical_filename)
                thermal_path = os.path.join(thermal_dir, thermal_filename)
                
                # Save images (OpenCV uses BGR, so convert RGB to BGR for optical)
                cv2.imwrite(optical_path, cv2.cvtColor(optical_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(thermal_path, thermal_resized)
                
                stats['successfully_processed'] += 1
                sample_index += 1
                
                # Update progress bar
                pbar_numeric.set_postfix({
                    'Processed': stats['successfully_processed'],
                    'Skipped': stats['skipped_corrupted'] + stats['skipped_missing_bands']
                })
                
            except Exception as e:
                logger.error(f"Error processing {tif_path}: {str(e)}")
                stats['skipped_other_errors'] += 1
                skipped_scenes.append((tif_path, f"Processing error: {str(e)}"))
                continue
    
    # Write skipped scenes to log file
    with open(log_file, 'w') as f:
        f.write("Skipped Scenes Log\n")
        f.write("=" * 80 + "\n\n")
        for scene_path, reason in skipped_scenes:
            f.write(f"Scene: {scene_path}\n")
            f.write(f"Reason: {reason}\n")
            f.write("-" * 80 + "\n")
    
    # Final statistics
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Complete!")
    logger.info(f"Total scenes found: {stats['total_scenes_found']}")
    logger.info(f"Successfully processed: {stats['successfully_processed']}")
    logger.info(f"Skipped (corrupted): {stats['skipped_corrupted']}")
    logger.info(f"Skipped (missing bands): {stats['skipped_missing_bands']}")
    logger.info(f"Skipped (other errors): {stats['skipped_other_errors']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    return stats


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    # Create sample data
    create_sample_data()
    
    # Test processor
    processor = ThermalDataProcessor()
    
    # Load sample images
    thermal = cv2.imread("sample_data/thermal/thermal_000.png", cv2.IMREAD_GRAYSCALE)
    optical = cv2.imread("sample_data/optical/optical_000.png", cv2.IMREAD_COLOR)
    
    # Process pair
    lr_thermal, hr_optical, hr_thermal = processor.create_lr_hr_pair(thermal, optical)
    
    print(f"Low-res thermal shape: {lr_thermal.shape}")
    print(f"High-res optical shape: {hr_optical.shape}")
    print(f"High-res thermal shape: {hr_thermal.shape}")
    print("Preprocessing pipeline test completed!")
