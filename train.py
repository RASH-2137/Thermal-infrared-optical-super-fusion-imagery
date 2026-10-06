"""
Training Pipeline for Thermal Infrared Super-Resolution
========================================================

This module handles:
- Model training with combined L1 + SSIM + Edge loss
- Evaluation metrics: PSNR, SSIM, RMSE (in Kelvin)
- Checkpoint saving (best model + per-epoch)
- Training curve visualization

Dataset:    SSL4EO-L (Landsat 8 OLI/TIRS)
            Optical  : Bands 4 (Red), 3 (Green), 2 (Blue)
            Thermal  : Band 10 (TIRS-1)
Task:       4× super-resolution of Band 10 guided by RGB optical
"""

import os
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ThermalSuperResolutionNet, ThermalLoss, count_parameters
from preprocess import ThermalDataset, ThermalDataProcessor, create_sample_data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """Standard image-quality metrics for thermal super-resolution evaluation."""

    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """
        Peak Signal-to-Noise Ratio (higher is better, typical range: 25–40 dB).

        Formula:  PSNR = 20 · log10(MAX / sqrt(MSE))
        We work in [0, 1] space, so MAX = 1.0.
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()

    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """
        Structural Similarity Index (higher is better, range: -1 to 1).

        Measures luminance, contrast, and structural similarity.
        Uses an 11×11 Gaussian window (sigma=1.5) as in the original paper
        (Wang et al., 2004).
        """
        def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)

        window = gaussian_window(window_size).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=1)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred,     window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2) - mu2_sq
        sigma12   = F.conv2d(pred * target,   window, padding=window_size // 2) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        return ssim_map.mean().item()

    @staticmethod
    def rmse_kelvin(pred: torch.Tensor, target: torch.Tensor,
                    temp_range: Tuple[float, float] = (273.15, 323.15)) -> float:
        """
        RMSE in Kelvin (lower is better).

        Converts normalized [0, 1] predictions back to Kelvin using the
        known thermal range of Landsat Band 10 (~0°C to 50°C → 273–323 K).
        """
        lo, hi = temp_range
        pred_k   = pred   * (hi - lo) + lo
        target_k = target * (hi - lo) + lo
        return torch.sqrt(torch.mean((pred_k - target_k) ** 2)).item()

    @classmethod
    def calculate_all(cls, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        return {
            'psnr':   cls.psnr(pred, target),
            'ssim':   cls.ssim(pred, target),
            'rmse_k': cls.rmse_kelvin(pred, target),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ThermalTrainer:
    """
    Main training controller.

    Handles the full training loop including:
    - NaN / Inf protection at every stage (inputs → output → loss → gradients)
    - Gradient clipping (max_norm=1.0)
    - ReduceLROnPlateau scheduler
    - Best-model checkpoint saving
    - TensorBoard logging
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device, config: Dict):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.config       = config

        # Loss: L1 (pixel fidelity) + SSIM (structure) + Sobel edge (sharpness)
        self.criterion = ThermalLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.1),
            gamma=config.get('gamma', 0.1),
        )

        # Adam with mild weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
        )

        # Halve LR when val loss plateaus for `patience` epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
        )

        self.writer = SummaryWriter(config.get('log_dir', 'runs/thermal_sr'))

        # Training state
        self.epoch          = 0
        self.best_val_loss  = float('inf')
        self.train_losses   = []
        self.val_losses     = []
        self.val_metrics    = []

        os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with full NaN/Inf guard at every stage."""
        self.model.train()
        total_loss     = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0, 'edge': 0.0}
        skipped_batches = 0
        valid_batches   = 0
        warned          = False  # Print NaN warning only once per epoch

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):

            # Skip None batches (all samples in batch were invalid)
            if batch is None:
                skipped_batches += 1
                continue

            lr_thermal = batch['lr_thermal'].to(self.device)
            hr_optical = batch['hr_optical'].to(self.device)
            hr_thermal = batch['hr_thermal'].to(self.device)

            # Guard: NaN/Inf in inputs
            if self._has_bad_values(lr_thermal, hr_optical, hr_thermal):
                if not warned:
                    print(f"\nWARN batch {batch_idx}: NaN/Inf in inputs — skipping")
                    warned = True
                skipped_batches += 1
                continue

            # --- Forward pass ---
            self.optimizer.zero_grad()
            sr_thermal = self.model(lr_thermal, hr_optical)

            if self._has_bad_values(sr_thermal):
                if not warned:
                    print(f"\nWARN batch {batch_idx}: NaN/Inf in model output — skipping")
                    warned = True
                skipped_batches += 1
                continue

            sr_thermal = torch.clamp(sr_thermal, 0.0, 1.0)

            # Align spatial dims if decoder output size drifted
            if sr_thermal.shape[2:] != hr_thermal.shape[2:]:
                sr_thermal = F.interpolate(
                    sr_thermal, size=hr_thermal.shape[2:],
                    mode='bilinear', align_corners=False,
                )

            # --- Loss ---
            losses = self.criterion(sr_thermal, hr_thermal)
            loss   = losses['total']

            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                continue

            # --- Backward pass ---
            loss.backward()

            # Guard: NaN gradients
            if self._has_nan_grads():
                if not warned:
                    print(f"\nWARN batch {batch_idx}: NaN gradients — skipping update")
                    warned = True
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # --- Stats ---
            valid_batches += 1
            total_loss    += loss.item()
            for k in loss_components:
                loss_components[k] += losses[k].item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L1':   f'{losses["l1"].item():.4f}',
                'Skip': skipped_batches,
            })

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            for k in loss_components:
                loss_components[k] /= valid_batches
        else:
            avg_loss = 0.0

        if skipped_batches > 0:
            print(f"  → Skipped {skipped_batches} batches ({valid_batches} valid)")

        return {'total': avg_loss, **loss_components}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate on validation set. Returns (loss_dict, metrics_dict)."""
        self.model.eval()
        total_loss      = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0, 'edge': 0.0}
        all_metrics     = {'psnr': [], 'ssim': [], 'rmse_k': []}
        valid_batches   = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Guard: None batch
                if batch is None:
                    continue

                lr_thermal = batch['lr_thermal'].to(self.device)
                hr_optical = batch['hr_optical'].to(self.device)
                hr_thermal = batch['hr_thermal'].to(self.device)

                sr_thermal = self.model(lr_thermal, hr_optical)
                sr_thermal = torch.clamp(sr_thermal, 0.0, 1.0)

                if sr_thermal.shape[2:] != hr_thermal.shape[2:]:
                    sr_thermal = F.interpolate(
                        sr_thermal, size=hr_thermal.shape[2:],
                        mode='bilinear', align_corners=False,
                    )

                losses = self.criterion(sr_thermal, hr_thermal)
                total_loss += losses['total'].item()
                for k in loss_components:
                    loss_components[k] += losses[k].item()

                metrics = MetricsCalculator.calculate_all(sr_thermal, hr_thermal)
                for k in all_metrics:
                    all_metrics[k].append(metrics[k])

                valid_batches += 1

        if valid_batches == 0:
            return {'total': 0.0, **loss_components}, {k: 0.0 for k in all_metrics}

        avg_loss = total_loss / valid_batches
        for k in loss_components:
            loss_components[k] /= valid_batches
        avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

        return {'total': avg_loss, **loss_components}, avg_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False):
        """Save model + optimizer + scheduler state to disk."""
        ckpt = {
            'epoch':               self.epoch,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss':       self.best_val_loss,
            'config':              self.config,
        }
        epoch_path = os.path.join(
            self.config.get('checkpoint_dir', 'checkpoints'),
            f'checkpoint_epoch_{self.epoch:03d}.pth',
        )
        torch.save(ckpt, epoch_path)

        if is_best:
            best_path = os.path.join(
                self.config.get('checkpoint_dir', 'checkpoints'), 'best_model.pth',
            )
            torch.save(ckpt, best_path)
            print(f"  ★ New best model saved (epoch {self.epoch})")

    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch         = ckpt['epoch']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from checkpoint: epoch {self.epoch}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, save_freq: int = 1):
        """Run the full training loop for `num_epochs` epochs."""
        print(f"\n{'='*60}")
        print(f"  Training for {num_epochs} epochs")
        print(f"  Parameters : {count_parameters(self.model):,}")
        print(f"  Device     : {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)

            val_losses, val_metrics = self.validate()
            self.val_losses.append(val_losses)
            self.val_metrics.append(val_metrics)

            self.scheduler.step(val_losses['total'])

            # TensorBoard
            self.writer.add_scalar('Loss/Train',      train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val',        val_losses['total'],   epoch)
            self.writer.add_scalar('Metrics/PSNR',    val_metrics['psnr'],   epoch)
            self.writer.add_scalar('Metrics/SSIM',    val_metrics['ssim'],   epoch)
            self.writer.add_scalar('Metrics/RMSE_K',  val_metrics['rmse_k'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Console summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss : {train_losses['total']:.4f}")
            print(f"  Val   Loss : {val_losses['total']:.4f}  |  "
                  f"PSNR {val_metrics['psnr']:.2f} dB  |  "
                  f"SSIM {val_metrics['ssim']:.4f}  |  "
                  f"RMSE {val_metrics['rmse_k']:.2f} K")

            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(is_best)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        self.save_checkpoint()
        self.writer.close()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot and save 2×2 grid of training curves."""
        epochs = range(len(self.train_losses))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Curves — Thermal Super-Resolution', fontsize=14)

        # Loss
        axes[0, 0].plot(epochs, [l['total'] for l in self.train_losses], 'b-', label='Train')
        axes[0, 0].plot(epochs, [l['total'] for l in self.val_losses],   'r-', label='Val')
        axes[0, 0].set(title='Total Loss', xlabel='Epoch', ylabel='Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True)

        # PSNR
        axes[0, 1].plot(epochs, [m['psnr'] for m in self.val_metrics], 'g-')
        axes[0, 1].set(title='Val PSNR', xlabel='Epoch', ylabel='dB')
        axes[0, 1].grid(True)

        # SSIM
        axes[1, 0].plot(epochs, [m['ssim'] for m in self.val_metrics], 'm-')
        axes[1, 0].set(title='Val SSIM', xlabel='Epoch', ylabel='SSIM')
        axes[1, 0].grid(True)

        # RMSE
        axes[1, 1].plot(epochs, [m['rmse_k'] for m in self.val_metrics], 'c-')
        axes[1, 1].set(title='Val RMSE', xlabel='Epoch', ylabel='Kelvin')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_bad_values(*tensors: torch.Tensor) -> bool:
        """Return True if any tensor contains NaN or Inf."""
        return any(
            torch.any(torch.isnan(t)) or torch.any(torch.isinf(t))
            for t in tensors
        )

    def _has_nan_grads(self) -> bool:
        """Return True if any model parameter gradient contains NaN or Inf."""
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.any(torch.isnan(p.grad)) or torch.any(torch.isinf(p.grad)):
                    return True
        return False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def create_data_loaders(data_dir: str, config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train + validation DataLoaders from a preprocessed dataset directory.

    Expected directory layout::

        data_dir/
          thermal/   thermal_00000.png  thermal_00001.png  ...
          optical/   optical_00000.png  optical_00001.png  ...

    Images are matched by sorted filename order, so names must align.
    An 80/20 train/val split is applied.
    """
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Creating synthetic sample data for testing …")
        create_sample_data(data_dir, num_samples=config.get('num_samples', 100))

    thermal_dir = os.path.join(data_dir, 'thermal')
    optical_dir = os.path.join(data_dir, 'optical')

    extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    thermal_paths = sorted([
        os.path.join(thermal_dir, f)
        for f in os.listdir(thermal_dir) if f.lower().endswith(extensions)
    ])
    optical_paths = sorted([
        os.path.join(optical_dir, f)
        for f in os.listdir(optical_dir) if f.lower().endswith(extensions)
    ])

    assert len(thermal_paths) == len(optical_paths), (
        f"Mismatch: {len(thermal_paths)} thermal vs {len(optical_paths)} optical images"
    )
    print(f"Found {len(thermal_paths)} image pairs")

    processor = ThermalDataProcessor(
        target_size=config.get('target_size', (256, 256)),
        thermal_range=config.get('thermal_range', (0.0, 1.0)),
        optical_range=config.get('optical_range', (0.0, 1.0)),
    )

    split = int(0.8 * len(thermal_paths))
    train_dataset = ThermalDataset(
        thermal_paths[:split], optical_paths[:split], processor,
        scale_factor=config.get('scale_factor', 4),
        augment=config.get('augment', True),
        align=False,  # Alignment disabled at train time for speed
    )
    val_dataset = ThermalDataset(
        thermal_paths[split:], optical_paths[split:], processor,
        scale_factor=config.get('scale_factor', 4),
        augment=False,
        align=False,
    )

    def collate_fn(batch):
        """Drop None samples (invalid images) without crashing the loader."""
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            'lr_thermal': torch.stack([b['lr_thermal'] for b in batch]),
            'hr_optical': torch.stack([b['hr_optical'] for b in batch]),
            'hr_thermal': torch.stack([b['hr_thermal'] for b in batch]),
        }

    num_workers = config.get('num_workers', 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        collate_fn=collate_fn,
    )

    print(f"  Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main training entry point.

    Dataset layout expected at `data_dir`::

        training_data/
          thermal/   thermal_00000.png ...   (grayscale, 256×256)
          optical/   optical_00000.png ...   (RGB,       256×256)

    On Kaggle, set data_dir to the mounted input path, e.g.:
        '/kaggle/input/landsat-dataset/training_data'
    """
    config = {
        # ── Data ──────────────────────────────────────────────────────
        'data_dir':        'training_data',   # Override for Kaggle
        'target_size':     (256, 256),
        'scale_factor':    4,
        'thermal_range':   (0.0, 1.0),
        'optical_range':   (0.0, 1.0),

        # ── Training ──────────────────────────────────────────────────
        'num_epochs':      30,
        'batch_size':      8,
        'num_workers':     4,         # Set to 0 on Windows; 4 on Linux/Kaggle
        'learning_rate':   1e-4,
        'weight_decay':    1e-5,
        'augment':         True,

        # ── Loss weights ──────────────────────────────────────────────
        'alpha':           1.0,       # L1  — pixel-level fidelity
        'beta':            0.1,       # SSIM — structural similarity
        'gamma':           0.1,       # Edge (Sobel) — edge sharpness

        # ── Checkpointing ─────────────────────────────────────────────
        'checkpoint_dir':  'checkpoints',
        'log_dir':         'runs/thermal_sr',
        'num_samples':     100,       # Only used for synthetic demo data
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = create_data_loaders(config['data_dir'], config)

    if len(train_loader.dataset) == 0:
        print("ERROR: No training images found. Exiting.")
        return

    # Model
    model = ThermalSuperResolutionNet(
        thermal_channels=1,
        optical_channels=3,
        base_channels=64,
        scale_factor=config['scale_factor'],
    )

    trainer = ThermalTrainer(model, train_loader, val_loader, device, config)
    trainer.train(num_epochs=config['num_epochs'], save_freq=1)
    trainer.plot_training_curves()


if __name__ == '__main__':
    main()
