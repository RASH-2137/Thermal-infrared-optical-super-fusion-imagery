"""
Training Pipeline for Thermal Infrared Super-Resolution

This module handles:
- Model training with physics-aware loss functions
- Evaluation metrics (PSNR, SSIM, RMSE)
- Checkpointing and model saving
- Training visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

from model import ThermalSuperResolutionNet, ThermalLoss, count_parameters
from preprocess import ThermalDataset, ThermalDataProcessor, create_sample_data


class MetricsCalculator:
    """Calculate evaluation metrics for thermal super-resolution"""
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
    
    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Calculate Structural Similarity Index"""
        def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32)
            coords = coords - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)
        
        window = gaussian_window(window_size).to(pred.device)
        
        mu1 = torch.nn.functional.conv2d(pred, window, padding=window_size//2, groups=1)
        mu2 = torch.nn.functional.conv2d(target, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=window_size//2, groups=1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    @staticmethod
    def rmse_kelvin(pred: torch.Tensor, target: torch.Tensor, 
                   temp_range: Tuple[float, float] = (273.15, 323.15)) -> float:
        """Calculate RMSE in Kelvin"""
        # Convert normalized values back to Kelvin
        pred_k = pred * (temp_range[1] - temp_range[0]) + temp_range[0]
        target_k = target * (temp_range[1] - temp_range[0]) + temp_range[0]
        
        mse = torch.mean((pred_k - target_k) ** 2)
        return torch.sqrt(mse).item()
    
    @classmethod
    def calculate_all(cls, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics"""
        return {
            'psnr': cls.psnr(pred, target),
            'ssim': cls.ssim(pred, target),
            'rmse_k': cls.rmse_kelvin(pred, target)
        }


class ThermalTrainer:
    """Main trainer class for thermal super-resolution"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: torch.device, config: Dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = ThermalLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.1),
            gamma=config.get('gamma', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(config.get('log_dir', 'runs/thermal_sr'))
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Create checkpoint directory
        os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with NaN/Inf protection"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0, 'edge': 0.0}
        skipped_batches = 0
        valid_batches = 0
        first_nan_warning = True
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Filter out None samples from batch
            if batch is None:
                skipped_batches += 1
                continue
            
            # Check if batch contains None values
            if (batch.get('lr_thermal') is None or batch.get('hr_optical') is None or 
                batch.get('hr_thermal') is None):
                skipped_batches += 1
                continue
            
            lr_thermal = batch['lr_thermal'].to(self.device)
            hr_optical = batch['hr_optical'].to(self.device)
            hr_thermal = batch['hr_thermal'].to(self.device)
            
            # Check for NaN/Inf in inputs
            if (torch.any(torch.isnan(lr_thermal)) or torch.any(torch.isinf(lr_thermal)) or
                torch.any(torch.isnan(hr_optical)) or torch.any(torch.isinf(hr_optical)) or
                torch.any(torch.isnan(hr_thermal)) or torch.any(torch.isinf(hr_thermal))):
                if first_nan_warning:
                    print(f"Warning: NaN/Inf detected in batch {batch_idx}, skipping")
                    first_nan_warning = False
                skipped_batches += 1
                continue
            
            # Forward pass
            self.optimizer.zero_grad()
            sr_thermal = self.model(lr_thermal, hr_optical)
            
            # Check for NaN/Inf in model output
            if torch.any(torch.isnan(sr_thermal)) or torch.any(torch.isinf(sr_thermal)):
                if first_nan_warning:
                    print(f"Warning: NaN/Inf in model output at batch {batch_idx}, skipping")
                    first_nan_warning = False
                skipped_batches += 1
                continue
            
            # Clamp output to [0, 1] for numerical stability
            sr_thermal = torch.clamp(sr_thermal, 0.0, 1.0)
            
            # Align spatial resolution before loss computation
            if sr_thermal.shape[2:] != hr_thermal.shape[2:]:
                sr_thermal = F.interpolate(
                    sr_thermal,
                    size=hr_thermal.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Calculate loss
            losses = self.criterion(sr_thermal, hr_thermal)
            loss = losses['total']
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                if first_nan_warning:
                    print(f"Warning: NaN/Inf in loss at batch {batch_idx}, skipping")
                    first_nan_warning = False
                skipped_batches += 1
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                if first_nan_warning:
                    print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping")
                    first_nan_warning = False
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            valid_batches += 1
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L1': f'{losses["l1"].item():.4f}',
                'SSIM': f'{losses["ssim"].item():.4f}',
                'Edge': f'{losses["edge"].item():.4f}',
                'Skipped': skipped_batches
            })
        
        # Average losses
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            for key in loss_components:
                loss_components[key] /= valid_batches
        else:
            avg_loss = 0.0
        
        if skipped_batches > 0:
            print(f"Epoch {self.epoch}: Skipped {skipped_batches} batches, {valid_batches} valid batches")
        
        return {'total': avg_loss, **loss_components}
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0, 'edge': 0.0}
        all_metrics = {'psnr': [], 'ssim': [], 'rmse_k': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                lr_thermal = batch['lr_thermal'].to(self.device)
                hr_optical = batch['hr_optical'].to(self.device)
                hr_thermal = batch['hr_thermal'].to(self.device)
                
                # Forward pass
                sr_thermal = self.model(lr_thermal, hr_optical)
                
                # Clamp output to [0, 1] for numerical stability
                sr_thermal = torch.clamp(sr_thermal, 0.0, 1.0)
                
                # Align spatial resolution before loss computation
                if sr_thermal.shape[2:] != hr_thermal.shape[2:]:
                    sr_thermal = F.interpolate(
                        sr_thermal,
                        size=hr_thermal.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Calculate loss
                losses = self.criterion(sr_thermal, hr_thermal)
                loss = losses['total']
                
                # Update loss statistics
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += losses[key].item()
                
                # Calculate metrics (sr_thermal already aligned with hr_thermal)
                metrics = MetricsCalculator.calculate_all(sr_thermal, hr_thermal)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
        
        # Average losses and metrics
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        return {'total': avg_loss, **loss_components}, avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 
                                     f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, save_freq: int = 10):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses, val_metrics = self.validate()
            self.val_losses.append(val_losses)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_losses['total'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val', val_losses['total'], epoch)
            self.writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
            self.writer.add_scalar('Metrics/SSIM', val_metrics['ssim'], epoch)
            self.writer.add_scalar('Metrics/RMSE_K', val_metrics['rmse_k'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"Val Loss: {val_losses['total']:.4f}")
            print(f"Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"Val RMSE: {val_metrics['rmse_k']:.2f} K")
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(is_best)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint()
        
        # Close tensorboard writer
        self.writer.close()
    
    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(len(self.train_losses))
        axes[0, 0].plot(epochs, [l['total'] for l in self.train_losses], 'b-', label='Train')
        axes[0, 0].plot(epochs, [l['total'] for l in self.val_losses], 'r-', label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PSNR
        axes[0, 1].plot(epochs, [m['psnr'] for m in self.val_metrics], 'g-')
        axes[0, 1].set_title('Validation PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True)
        
        # SSIM
        axes[1, 0].plot(epochs, [m['ssim'] for m in self.val_metrics], 'm-')
        axes[1, 0].set_title('Validation SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].grid(True)
        
        # RMSE
        axes[1, 1].plot(epochs, [m['rmse_k'] for m in self.val_metrics], 'c-')
        axes[1, 1].set_title('Validation RMSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE (K)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_data_loaders(data_dir: str, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    # Create sample data if directory doesn't exist
    if not os.path.exists(data_dir):
        print(f"Creating sample data in {data_dir}")
        create_sample_data(data_dir, num_samples=config.get('num_samples', 100))
    
    # Get image paths
    thermal_dir = os.path.join(data_dir, 'thermal')
    optical_dir = os.path.join(data_dir, 'optical')
    
    thermal_paths = [os.path.join(thermal_dir, f) for f in sorted(os.listdir(thermal_dir)) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
    optical_paths = [os.path.join(optical_dir, f) for f in sorted(os.listdir(optical_dir)) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create processor
    processor = ThermalDataProcessor(
        target_size=config.get('target_size', (256, 256)),
        thermal_range=config.get('thermal_range', (0.0, 1.0)),
        optical_range=config.get('optical_range', (0.0, 1.0))
    )
    
    # Create datasets
    train_size = int(0.8 * len(thermal_paths))
    train_thermal = thermal_paths[:train_size]
    train_optical = optical_paths[:train_size]
    val_thermal = thermal_paths[train_size:]
    val_optical = optical_paths[train_size:]
    
    train_dataset = ThermalDataset(
        train_thermal, train_optical, processor,
        scale_factor=config.get('scale_factor', 4),
        augment=config.get('augment', True),
        align=False  # Disable alignment for training speed
    )
    
    val_dataset = ThermalDataset(
        val_thermal, val_optical, processor,
        scale_factor=config.get('scale_factor', 4),
        augment=False,
        align=False  # Disable alignment for validation speed
    )
    
    # Create data loaders with collate function to filter None samples
    def collate_fn(batch):
        """Filter out None samples from batch"""
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        # Stack tensors
        return {
            'lr_thermal': torch.stack([b['lr_thermal'] for b in batch]),
            'hr_optical': torch.stack([b['hr_optical'] for b in batch]),
            'hr_thermal': torch.stack([b['hr_thermal'] for b in batch])
        }
    
    # Windows-safe: num_workers=0 to avoid multiprocessing pickle issues with cv2.SIFT
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=0,  # Windows-safe: single-process mode
        pin_memory=False,  # Windows-safe: pin_memory not needed with num_workers=0
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=0,  # Windows-safe: single-process mode
        pin_memory=False,  # Windows-safe: pin_memory not needed with num_workers=0
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    # Configuration
    # For large datasets (10GB+), use 'training_data' directory
    # Make sure your data is organized as: training_data/thermal/ and training_data/optical/
    config = {
        'data_dir': 'training_data',  # Change this to your dataset directory
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs/thermal_sr',
        'num_samples': 50,  # Only used if creating sample data
        'batch_size': 4,    # Batch size for training
        'num_workers': 4,   # Increased for faster data loading (reduce if memory issues)
        'target_size': (256, 256),
        'scale_factor': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'alpha': 1.0,  # L1 loss weight
        'beta': 0.0,   # SSIM loss weight (disabled temporarily for stability)
        'gamma': 0.0,  # Edge loss weight (disabled temporarily for stability)
        'augment': True
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if data directory exists
    if not os.path.exists(config['data_dir']):
        print(f"\n⚠️  WARNING: Data directory '{config['data_dir']}' not found!")
        print(f"Please create the directory structure:")
        print(f"  {config['data_dir']}/thermal/  (put thermal images here)")
        print(f"  {config['data_dir']}/optical/  (put optical images here)")
        print("\nImages must have matching names (e.g., thermal_000.png and optical_000.png)")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    print(f"Loading data from: {config['data_dir']}")
    train_loader, val_loader = create_data_loaders(config['data_dir'], config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    if len(train_loader.dataset) == 0:
        print("\n⚠️  ERROR: No training images found!")
        print(f"Please check that {config['data_dir']}/thermal/ and {config['data_dir']}/optical/ contain images")
        return
    
    # Create model
    print("Creating model...")
    model = ThermalSuperResolutionNet(
        thermal_channels=1,
        optical_channels=3,
        base_channels=64,
        scale_factor=config['scale_factor']
    )
    
    # Create trainer
    trainer = ThermalTrainer(model, train_loader, val_loader, device, config)
    
    # Train
    num_epochs = 3  # Number of training epochs
    trainer.train(num_epochs=num_epochs, save_freq=1)  # Save every epoch
    
    # Plot training curves
    trainer.plot_training_curves()


if __name__ == "__main__":
    main()
