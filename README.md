---
title: CropMind AI
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
 
# CropMind AI

CropMind AI is a smart agricultural assistant application designed to help farmers classify crop diseases from images. It uses a custom-trained ResNet34 model served by a FastAPI backend and a clean, responsive frontend.

## Structure
- `app/`: FastAPI backend and inference endpoints.
- `model/`: PyTorch model definition and training scripts.
- `models/`: Trained model checkpoints.
- `frontend/`: Web UI (HTML, CSS, JS) for interacting with the backend.
- `utils/`: Utility scripts.

## Running the App
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `python app/app.py`
3. Access the frontend in your browser.
