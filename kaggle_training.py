# ============================================================
# Thermal Infrared Super-Resolution — Kaggle Training Notebook
# ============================================================
# Dataset  : SSL4EO-L (Landsat 8 OLI/TIRS TOA benchmark)
# Task     : 4× guided thermal super-resolution
# Model    : Dual-encoder (thermal + optical) + CBAM fusion + PixelShuffle decoder
# GPU      : T4 / P100 on Kaggle (enable in Settings → Accelerator)
# ============================================================

# ----------------------------------------------------------
# CELL 1 — Install dependencies
# ----------------------------------------------------------
!pip install -q rasterio tqdm

# ----------------------------------------------------------
# CELL 2 — Upload / copy source files from your dataset
# ----------------------------------------------------------
# If you have added the code files as a Kaggle dataset, copy them here.
# Otherwise paste model.py / preprocess.py / train.py content directly.

import shutil, os

# If your code is in a Kaggle dataset called "thermal-sr-code":
CODE_DATASET = '/kaggle/input/thermal-sr-code'
if os.path.exists(CODE_DATASET):
    for fname in ['model.py', 'preprocess.py', 'train.py']:
        src = os.path.join(CODE_DATASET, fname)
        if os.path.exists(src):
            shutil.copy(src, f'/kaggle/working/{fname}')
            print(f'Copied {fname}')
else:
    print("Code dataset not found — paste model.py / preprocess.py manually.")

os.chdir('/kaggle/working')

# ----------------------------------------------------------
# CELL 3 — Verify GPU
# ----------------------------------------------------------
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
print(f"Device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ----------------------------------------------------------
# CELL 4 — Inspect the raw SSL4EO-L dataset structure
# ----------------------------------------------------------
RAW_DATASET = '/kaggle/input/ssl4eo-landsat-benchmark'  # ← change to your dataset slug

import os, glob
sample_folders = sorted(os.listdir(RAW_DATASET))[:3]
for folder in sample_folders:
    scenes = os.listdir(os.path.join(RAW_DATASET, folder))
    for scene in scenes:
        tifs = glob.glob(os.path.join(RAW_DATASET, folder, scene, '*.tif'))
        print(f"{folder}/{scene} → {[os.path.basename(t) for t in tifs]}")

# ----------------------------------------------------------
# CELL 5 — Preprocess: extract optical (RGB) + thermal bands
# ----------------------------------------------------------
# This converts raw GeoTIFFs → training_data/thermal/ + training_data/optical/
# Bands used:
#   Optical RGB : Band 4 (Red), Band 3 (Green), Band 2 (Blue)
#   Thermal     : Band 10 (TIRS-1, 100m resolution)

from preprocess import process_ssl4eo_landsat_dataset

stats = process_ssl4eo_landsat_dataset(
    dataset_root=RAW_DATASET,
    output_dir='/kaggle/working/training_data',
    target_size=(256, 256),
    log_file='/kaggle/working/preprocessing_log.txt',
)

print("\n=== Preprocessing Statistics ===")
for key, val in stats.items():
    print(f"  {key:35s}: {val}")

# Verify output
import os
n_thermal = len(os.listdir('/kaggle/working/training_data/thermal'))
n_optical = len(os.listdir('/kaggle/working/training_data/optical'))
print(f"\nReady: {n_thermal} thermal, {n_optical} optical images")

# ----------------------------------------------------------
# CELL 6 — (Optional) Visualise a few preprocessed samples
# ----------------------------------------------------------
import cv2, matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Preprocessed Sample Pairs (Optical RGB  |  Thermal Grayscale)', fontsize=13)

thermal_dir = '/kaggle/working/training_data/thermal'
optical_dir = '/kaggle/working/training_data/optical'
sample_names = sorted(os.listdir(thermal_dir))[:4]

for col, name in enumerate(sample_names):
    idx = name.replace('thermal_', '').replace('.png', '')
    opt_path = os.path.join(optical_dir, f'optical_{idx}.png')
    thm_path = os.path.join(thermal_dir, name)

    opt = cv2.cvtColor(cv2.imread(opt_path), cv2.COLOR_BGR2RGB)
    thm = cv2.imread(thm_path, cv2.IMREAD_GRAYSCALE)

    axes[0, col].imshow(opt)
    axes[0, col].set_title(f'Optical {idx}', fontsize=9)
    axes[0, col].axis('off')

    axes[1, col].imshow(thm, cmap='hot')
    axes[1, col].set_title(f'Thermal {idx}', fontsize=9)
    axes[1, col].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/sample_pairs.png', dpi=120)
plt.show()

# ----------------------------------------------------------
# CELL 7 — Verify one sample goes through the data pipeline
# ----------------------------------------------------------
import sys
sys.path.insert(0, '/kaggle/working')

from preprocess import ThermalDataProcessor

processor = ThermalDataProcessor(target_size=(256, 256))
t_path = os.path.join(thermal_dir, sorted(os.listdir(thermal_dir))[0])
o_path = os.path.join(optical_dir, sorted(os.listdir(optical_dir))[0])

thermal_img = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
optical_img = cv2.imread(o_path, cv2.IMREAD_COLOR)

lr_t, hr_o, hr_t = processor.create_lr_hr_pair(thermal_img, optical_img, scale_factor=4)

print(f"LR Thermal  : {lr_t.shape}  range [{lr_t.min():.3f}, {lr_t.max():.3f}]")
print(f"HR Optical  : {hr_o.shape}  range [{hr_o.min():.3f}, {hr_o.max():.3f}]")
print(f"HR Thermal  : {hr_t.shape}  range [{hr_t.min():.3f}, {hr_t.max():.3f}]")
# Expected:
#   LR Thermal : (64, 64)    range [0.0, 1.0]
#   HR Optical : (256, 256, 3) range [0.0, 1.0]
#   HR Thermal : (256, 256)  range [0.0, 1.0]

# ----------------------------------------------------------
# CELL 8 — Verify model forward pass
# ----------------------------------------------------------
from model import ThermalSuperResolutionNet, count_parameters

model = ThermalSuperResolutionNet(
    thermal_channels=1,
    optical_channels=3,
    base_channels=64,
    scale_factor=4,
)
print(f"Model parameters: {count_parameters(model):,}")

# Dummy forward pass (batch=2, lr thermal 64×64, hr optical 256×256)
with torch.no_grad():
    dummy_thermal = torch.randn(2, 1, 64, 64)
    dummy_optical = torch.randn(2, 3, 256, 256)
    out = model(dummy_thermal, dummy_optical)
    print(f"Input LR thermal : {dummy_thermal.shape}")
    print(f"Input HR optical : {dummy_optical.shape}")
    print(f"Output SR thermal: {out.shape}")
    # Expected: torch.Size([2, 1, 256, 256])
    assert out.shape == (2, 1, 256, 256), f"Shape mismatch: {out.shape}"
    print("✓ Forward pass OK")

# ----------------------------------------------------------
# CELL 9 — Train the model
# ----------------------------------------------------------
# This is the main training cell. Estimated time on Kaggle T4:
#   ~15–25 min per epoch with 25,000 samples, batch_size=8
#   30 epochs ≈ 8–12 hours (use Kaggle's 12h GPU session)

from train import ThermalTrainer, create_data_loaders
import json

config = {
    # Data
    'data_dir':       '/kaggle/working/training_data',
    'target_size':    (256, 256),
    'scale_factor':   4,
    'thermal_range':  (0.0, 1.0),
    'optical_range':  (0.0, 1.0),

    # Training
    'num_epochs':     30,           # Reduce to 5–10 for a quick test run
    'batch_size':     8,            # T4: try 8–16; P100: try 16–32
    'num_workers':    4,            # Linux Kaggle supports multiprocessing
    'learning_rate':  1e-4,
    'weight_decay':   1e-5,
    'augment':        True,

    # Loss weights (L1 + SSIM + Edge)
    'alpha':          1.0,
    'beta':           0.1,
    'gamma':          0.1,

    # Checkpointing
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'log_dir':        '/kaggle/working/runs/thermal_sr',
}

# Save config for reproducibility
with open('/kaggle/working/training_config.json', 'w') as f:
    json.dump(config, f, indent=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

train_loader, val_loader = create_data_loaders(config['data_dir'], config)

model = ThermalSuperResolutionNet(
    thermal_channels=1,
    optical_channels=3,
    base_channels=64,
    scale_factor=config['scale_factor'],
).to(device)

trainer = ThermalTrainer(model, train_loader, val_loader, device, config)
trainer.train(num_epochs=config['num_epochs'], save_freq=1)

# ----------------------------------------------------------
# CELL 10 — Plot training curves
# ----------------------------------------------------------
trainer.plot_training_curves('/kaggle/working/training_curves.png')

from IPython.display import Image
Image('/kaggle/working/training_curves.png')

# ----------------------------------------------------------
# CELL 11 — Quick qualitative check (inference on a val sample)
# ----------------------------------------------------------
model.eval()

# Load a validation sample
thermal_val = cv2.imread(
    os.path.join(thermal_dir, sorted(os.listdir(thermal_dir))[-1]),
    cv2.IMREAD_GRAYSCALE,
)
optical_val = cv2.imread(
    os.path.join(optical_dir, sorted(os.listdir(optical_dir))[-1]),
    cv2.IMREAD_COLOR,
)

lr_t, hr_o, hr_t = processor.create_lr_hr_pair(thermal_val, optical_val, scale_factor=4)

lr_tensor = torch.tensor(lr_t).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,64,64)
hr_tensor = torch.tensor(hr_o).permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1,3,256,256)

with torch.no_grad():
    sr = model(lr_tensor, hr_tensor)
    sr = torch.clamp(sr, 0, 1)

sr_np = (sr.squeeze().cpu().numpy() * 255).astype(np.uint8)

# Side-by-side visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Super-Resolution Result', fontsize=13)

# Bicubic upscale of LR thermal as baseline
lr_bicubic = cv2.resize(lr_t, (256, 256), interpolation=cv2.INTER_CUBIC)

axes[0].imshow(lr_bicubic, cmap='hot')
axes[0].set_title('Bicubic (baseline)', fontsize=10)
axes[0].axis('off')

axes[1].imshow(sr_np, cmap='hot')
axes[1].set_title('Our Model (SR)', fontsize=10)
axes[1].axis('off')

axes[2].imshow(hr_t, cmap='hot')
axes[2].set_title('Ground Truth HR', fontsize=10)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/sr_comparison.png', dpi=150)
plt.show()

# Print final metrics vs. bicubic baseline
from train import MetricsCalculator

hr_tensor_target = torch.tensor(hr_t).unsqueeze(0).unsqueeze(0).float().to(device)

lr_bicubic_tensor = torch.tensor(lr_bicubic).unsqueeze(0).unsqueeze(0).float().to(device)

print("=== Final Metrics (single sample) ===")
print("  Method          PSNR (dB)   SSIM    RMSE (K)")
bic_metrics = MetricsCalculator.calculate_all(lr_bicubic_tensor, hr_tensor_target)
sr_metrics  = MetricsCalculator.calculate_all(sr, hr_tensor_target)
print(f"  Bicubic         {bic_metrics['psnr']:6.2f}      {bic_metrics['ssim']:.4f}  {bic_metrics['rmse_k']:.2f}")
print(f"  Our Model       {sr_metrics['psnr']:6.2f}      {sr_metrics['ssim']:.4f}  {sr_metrics['rmse_k']:.2f}")

# ----------------------------------------------------------
# CELL 12 — Save final artifacts for download
# ----------------------------------------------------------
import zipfile

artifacts = [
    '/kaggle/working/checkpoints/best_model.pth',
    '/kaggle/working/training_curves.png',
    '/kaggle/working/sr_comparison.png',
    '/kaggle/working/training_config.json',
    '/kaggle/working/preprocessing_log.txt',
]

with zipfile.ZipFile('/kaggle/working/thermal_sr_results.zip', 'w') as zf:
    for path in artifacts:
        if os.path.exists(path):
            zf.write(path, os.path.basename(path))
            print(f"  Added: {os.path.basename(path)}")

print("\nDownload thermal_sr_results.zip from the Output tab.")
print("Copy best_model.pth into your local checkpoints/ folder to run inference.")
