🔥 Thermal Infrared Super-Resolution (Optical-Guided)

A research prototype demonstrating an end-to-end pipeline for enhancing low-resolution thermal infrared images using high-resolution optical guidance.

This project focuses on system design, preprocessing correctness, and deployable inference, rather than claiming real-world accuracy.

📌 Project Motivation

Thermal sensors often suffer from low spatial resolution due to hardware limitations, while optical sensors capture rich spatial details.

This project explores a learning-based fusion approach where:

Thermal images preserve radiometric / temperature structure

Optical images provide high-frequency spatial guidance

The output is a super-resolved thermal image (4× upscaling).

⚠️ Important Disclaimer (READ FIRST)

This is a prototype / proof-of-concept project.

The model is not trained on real calibrated thermal datasets

Output quality is not suitable for scientific or operational use

The goal is to demonstrate:

Correct preprocessing

Model–data consistency

Inference integration

End-to-end system design

🚀 Features

✅ Optical-guided thermal super-resolution (4×)

✅ Unified preprocessing for training & inference

✅ Robust inference script with error handling

✅ Demo mode (no external data required)

✅ Web-based interface (upload or demo)

✅ CPU-only inference support

✅ Deployment-ready backend

🧪 Quick Start
Installation
git clone <your-repo-url>
cd thermal-super-resolution
pip install -r requirements.txt

🎯 Demo Mode (Recommended)

No thermal or optical images needed.

python inference.py --demo --visualize


This will:

Generate synthetic demo images

Run the trained model

Save the super-resolved output

Display side-by-side comparison

📸 Run on Your Own Images
python inference.py \
  --thermal thermal.png \
  --optical optical.png \
  --output sr_thermal.png


Thermal and optical images must correspond to the same scene.

🌐 Web Demo (Local)

Start the API server:

# Windows
start_api.bat

# macOS / Linux
chmod +x start_api.sh
./start_api.sh


Open in browser:

http://localhost:8000

Web Features

Upload thermal + optical images

One-click demo mode

Download super-resolved output

Health check & API docs

🔌 API Endpoints
Method	Endpoint	Description
GET	/	Web UI
GET	/health	Health check
GET	/demo	Run demo inference
POST	/infer	Upload images & infer
GET	/docs	Swagger UI

Example:

curl http://localhost:8000/demo -o output.png

🧠 Model Overview (High Level)

Dual input:

Low-resolution thermal (1-channel)

High-resolution optical (3-channel)

Separate feature extraction branches

Feature fusion via attention-based blocks

Upsampling via pixel shuffle (4×)

The architecture is intentionally simple and stable for demonstration.

🧩 Project Structure
thermal-super-resolution/
├── model.py              # Neural network definition
├── preprocess.py         # Shared preprocessing logic
├── train.py              # Training pipeline
├── inference.py          # CLI inference + demo mode
├── api.py                # FastAPI backend
├── static/               # Web frontend
├── demo_data/            # Auto-generated demo images
├── checkpoints/
│   └── best_model.pth    # Trained model
├── requirements.txt
├── Procfile
└── README.md

🧪 Training (Optional)
python train.py


Requires:

training_data/
 ├── thermal/
 └── optical/


Training is not required to run the demo or inference.

🚢 Deployment Notes

Designed for CPU-only inference

Suitable for:

Render

Railway

Local demo servers

Not optimized for high-throughput production workloads

Example start command:

uvicorn api:app --host 0.0.0.0 --port $PORT

🎓 What This Project Demonstrates

✔ End-to-end ML system design
✔ Correct preprocessing alignment
✔ Robust inference handling
✔ Model–data consistency
✔ Deployable demo architecture
✔ Practical engineering over benchmarks

📜 License

Provided as-is for educational and research demonstration purposes.

Project Status:
✅ End-to-End Pipeline Complete
⚠️ Research Prototype
🎯 Demo-Ready & Deployable