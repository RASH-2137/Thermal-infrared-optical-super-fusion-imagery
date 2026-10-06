# 🔥 Thermal Infrared Super-Resolution (Optical-Guided)

A research prototype demonstrating an **end-to-end deep learning pipeline** for
4× super-resolution of thermal infrared (TIRS) imagery, guided by co-registered
high-resolution optical (RGB) imagery.

Built on real-world satellite data from the
[SSL4EO-L Landsat 8 benchmark dataset](https://github.com/zhu-xlab/SSL4EO-L).

---

## 📌 Motivation

Thermal sensors (e.g., Landsat 8 TIRS Band 10) operate at **100 m native resolution**,
while optical sensors (OLI) capture at **30 m**. This spatial gap limits thermal
imagery in applications such as:

- Urban heat island mapping
- Agricultural drought monitoring
- Wildfire detection

This project uses a **learning-based guided super-resolution** approach:
- Optical bands carry high-frequency spatial structure (edges, textures)
- Thermal bands carry radiometric / temperature information
- A fusion network learns to transfer spatial detail from optical to thermal

---

## ⚠️ Disclaimer

This is a **research prototype** built for learning and demonstration.

- The model is trained on 25,000 Landsat 8 scenes (SSL4EO-L TOA benchmark)
- Output is suitable for demonstration, not operational science use
- No atmospheric correction is applied to the raw DN values

---

## 🏗️ Architecture

```
Low-Res Thermal  (B, 1, 64, 64)       High-Res Optical  (B, 3, 256, 256)
        │                                       │
  ThermalEncoder                         OpticalEncoder
  3× stride-2 conv + ResBlocks          3× stride-2 conv + ResBlocks
        │                                       │
        └─────────────┬─────────────────────────┘
                      │
                FusionModule
          Channel Attention (CBAM)
          Spatial  Attention (CBAM)
          Fusion Convolution
                      │
           SuperResolutionDecoder
           2× PixelShuffle(2) = 4× total
                      │
         SR Thermal Output  (B, 1, 256, 256)
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Dual encoder | Separate branches for thermal & optical | Preserves modality-specific features before fusion |
| CBAM attention | Channel + Spatial | Focuses on thermally-relevant regions in optical guidance |
| PixelShuffle | Sub-pixel convolution (Shi et al., 2016) | Avoids checkerboard artifacts vs. transposed conv |
| Loss | L1 + SSIM + Sobel edge | Balances pixel fidelity, structural quality, and sharpness |

---

## 📊 Dataset

**SSL4EO-L — Landsat 8 OLI/TIRS TOA Benchmark**

| Property | Value |
|----------|-------|
| Source | USGS Landsat 8 (OLI + TIRS) |
| Scenes | 25,000 global samples |
| Optical bands used | Band 4 (Red), Band 3 (Green), Band 2 (Blue) |
| Thermal band used | Band 10 (TIRS-1, ~100 m resolution) |
| Preprocessing | Min-max normalized to \[0, 1\], resized to 256×256 |
| Train / Val split | 80 / 20 |

---

## 📁 Project Structure

```
thermal-super-resolution/
├── model.py              # Neural network architecture
├── preprocess.py         # Data preprocessing + SSL4EO-L pipeline
├── train.py              # Training loop, metrics, checkpointing
├── inference.py          # CLI inference + demo mode
├── api.py                # FastAPI REST backend
├── static/
│   └── index.html        # Web frontend (drag-and-drop demo)
├── kaggle_training.py    # Step-by-step Kaggle GPU training notebook
├── checkpoints/
│   └── best_model.pth    # Best trained model (after training)
├── requirements.txt
├── Procfile              # Deployment config (Render / Railway)
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone <your-repo-url>
cd thermal-super-resolution
pip install -r requirements.txt
```

### Run Demo (requires trained model)

```bash
# Generate synthetic images + run inference
python inference.py --demo --visualize
```

### Run on Your Own Images

```bash
python inference.py \
  --thermal path/to/thermal.png \
  --optical path/to/optical.png \
  --output   sr_output.png \
  --visualize
```

### Web Interface

```bash
# Windows
start_api.bat

# macOS / Linux
./start_api.sh
```

Open http://localhost:8000 — upload images or click **Try Demo**.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Web UI |
| `GET`  | `/health`| Model + server health check |
| `GET`  | `/demo`  | Run with synthetic data |
| `POST` | `/infer` | Upload thermal + optical → download SR result |
| `GET`  | `/docs`  | Auto-generated Swagger UI |

```bash
# Health check
curl http://localhost:8000/health

# Demo
curl http://localhost:8000/demo -o demo_output.png

# Inference
curl -X POST http://localhost:8000/infer \
  -F "thermal=@thermal.png" \
  -F "optical=@optical.png" \
  -o sr_output.png
```

---

## 🧪 Training

### 1. Preprocess the SSL4EO-L dataset

```python
from preprocess import process_ssl4eo_landsat_dataset

stats = process_ssl4eo_landsat_dataset(
    dataset_root='/path/to/ssl4eo_l_oli_tirs_toa_benchmark',
    output_dir='training_data',
    target_size=(256, 256),
)
```

This extracts:
- `training_data/optical/` — RGB images (Bands 4, 3, 2)
- `training_data/thermal/` — Grayscale images (Band 10)

### 2. Train (Kaggle GPU recommended)

See [`kaggle_training.py`](kaggle_training.py) for the full step-by-step notebook.

```bash
# Local (CPU — slow, for testing only)
python train.py
```

Training config (in `train.py → main()`):

```python
config = {
    'num_epochs':    30,
    'batch_size':    8,
    'learning_rate': 1e-4,
    'alpha': 1.0,   # L1 loss weight
    'beta':  0.1,   # SSIM loss weight
    'gamma': 0.1,   # Edge loss weight
}
```

### 3. After training

Copy `checkpoints/best_model.pth` from Kaggle output back to your local project.
The demo and API will then work immediately.

---

## 📈 Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher (dB) |
| **SSIM** | Structural Similarity Index | Higher (0–1) |
| **RMSE** | Root Mean Squared Error in Kelvin | Lower (K) |

Baseline comparison: bicubic upsampling of the LR thermal image.

---

## 🚢 Deployment

Designed for **CPU-only inference** — no GPU required in production.

```bash
# Start server
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Compatible with: **Render**, **Railway**, **Fly.io**, or any Linux VPS.

---

## 🎓 What This Project Demonstrates

- ✅ End-to-end ML pipeline (data → preprocess → train → evaluate → serve)
- ✅ Multi-modal deep learning (fusing heterogeneous sensor data)
- ✅ Attention mechanisms (CBAM: channel + spatial)
- ✅ Sub-pixel convolution upsampling (PixelShuffle)
- ✅ Production-grade inference with error handling
- ✅ REST API + web frontend deployment
- ✅ Real-world satellite dataset (SSL4EO-L, Landsat 8)

---

## 📚 References

1. Wang et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP. — SSIM metric
2. Shi et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.* CVPR. — PixelShuffle
3. Woo et al. (2018). *CBAM: Convolutional Block Attention Module.* ECCV. — Attention mechanism
4. Wang et al. (2023). *SSL4EO-L: Datasets and Foundation Models for Landsat Imagery.* NeurIPS. — Dataset

---

## 📜 License

Provided as-is for educational and research demonstration purposes.

---

**Project Status:**
- ✅ Architecture & preprocessing implemented
- ✅ Training pipeline complete
- ✅ API & web frontend ready
- ⏳ Training in progress (Kaggle GPU)
