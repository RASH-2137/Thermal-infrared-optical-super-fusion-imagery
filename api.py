"""
FastAPI Backend for Thermal Super-Resolution

Simple REST API for running inference on thermal/optical image pairs.
CPU-only support for easy deployment.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import cv2
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Optional

from inference import run_inference, load_model, generate_demo_data
from model import ThermalSuperResolutionNet

app = FastAPI(
    title="Thermal Super-Resolution API",
    description="API for thermal infrared super-resolution using optical guidance",
    version="1.0.0"
)

# Serve static files (frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model_cache = None
MODEL_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    """Get or load the model (cached)."""
    global _model_cache
    if _model_cache is None:
        model, error = load_model(MODEL_PATH, DEVICE)
        if error:
            raise RuntimeError(f"Failed to load model: {error}")
        _model_cache = model
    return _model_cache


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend HTML."""
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return """
    <html>
        <body>
            <h1>Thermal Super-Resolution API</h1>
            <p>API is running. Use /infer endpoint or check /docs for API documentation.</p>
        </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(DEVICE)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/infer")
async def infer(
    thermal: UploadFile = File(..., description="Thermal image (grayscale)"),
    optical: UploadFile = File(..., description="Optical image (RGB)"),
    align: bool = True
):
    """
    Run super-resolution inference on uploaded thermal and optical images.
    
    Returns:
        Output image file (PNG)
    """
    # Validate file types
    if not thermal.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Thermal file must be an image")
    if not optical.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Optical file must be an image")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        thermal_path = os.path.join(tmpdir, "thermal.png")
        optical_path = os.path.join(tmpdir, "optical.png")
        output_path = os.path.join(tmpdir, "output.png")
        
        try:
            # Save uploaded files
            with open(thermal_path, "wb") as f:
                content = await thermal.read()
                f.write(content)
            
            with open(optical_path, "wb") as f:
                content = await optical.read()
                f.write(content)
            
            # Run inference
            result_path, error = run_inference(
                thermal_path=thermal_path,
                optical_path=optical_path,
                output_path=output_path,
                align=align
            )
            
            if error:
                raise HTTPException(status_code=500, detail=error)
            
            # Return file
            return FileResponse(
                result_path,
                media_type="image/png",
                filename="super_resolved_thermal.png"
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/demo")
async def demo():
    """
    Run inference with auto-generated demo data.
    Useful for testing without uploading images.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Generate demo data in temp directory
            thermal_path = os.path.join(tmpdir, "thermal.png")
            optical_path = os.path.join(tmpdir, "optical.png")
            
            # Generate synthetic images
            size = 256
            thermal = np.zeros((size, size), dtype=np.float32)
            y, x = np.ogrid[:size, :size]
            center_x, center_y = size // 2, size // 2
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            thermal += 200 * np.exp(-r / 80)
            thermal += 50 * np.sin(x * 0.1) * np.cos(y * 0.1)
            thermal += np.random.randn(size, size) * 10
            thermal = np.clip(thermal, 0, 255).astype(np.uint8)
            thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
            
            optical = np.zeros((size, size, 3), dtype=np.uint8)
            optical[:, :, 0] = 100 + 50 * np.sin(x * 0.05) * np.cos(y * 0.05)
            optical[:, :, 1] = 120 + 40 * np.sin(x * 0.07) * np.cos(y * 0.07)
            optical[:, :, 2] = 80 + 60 * np.sin(x * 0.09) * np.cos(y * 0.09)
            optical = cv2.GaussianBlur(optical, (3, 3), 0)
            
            cv2.imwrite(thermal_path, thermal)
            cv2.imwrite(optical_path, optical)
            
            output_path = os.path.join(tmpdir, "output.png")
            
            # Run inference
            result_path, error = run_inference(
                thermal_path=thermal_path,
                optical_path=optical_path,
                output_path=output_path,
                align=True
            )
            
            if error:
                raise HTTPException(status_code=500, detail=error)
            
            # Return file
            return FileResponse(
                result_path,
                media_type="image/png",
                filename="demo_super_resolved_thermal.png"
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

