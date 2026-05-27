---
title: OMyFish
emoji: 🐟
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
license: mit
---

# OMyFish — Fish Species Identifier

AI-powered fish identification from photos. Upload an image and the app returns the top-3 predicted species with confidence scores, habitat, diet, size, conservation status, and fun facts.

**Live demo:** [huggingface.co/spaces/fenghebonjour/omyfish](https://huggingface.co/spaces/fenghebonjour/omyfish)

## How it works

The app runs in one of two modes:

| Mode | When active | Model |
|---|---|---|
| **Zero-shot demo** | No trained checkpoint present | CLIP (`openai/clip-vit-base-patch32`) |
| **Fine-tuned** | `checkpoints/best.pt` exists | EfficientNet-B3 trained on your dataset |

Zero-shot mode works immediately with no data or training. Fine-tuned mode requires running the training pipeline below.

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app/main.py          # launches at http://localhost:8501
```

## Training a custom model

```bash
# 1. Download and organize a labeled fish dataset
python scripts/download_data.py download crowww/a-large-scale-fish-dataset
python scripts/download_data.py organize data/kaggle_tmp
python scripts/download_data.py stats

# 2. Train
make train

# 3. Evaluate
make eval                          # saves outputs/confusion_matrix.png

# 4. Predict a single image
make predict IMAGE=photo.jpg
```

All hyperparameters live in `configs/config.yaml`. The model switches to fine-tuned mode automatically once `checkpoints/best.pt` exists.

## Deploy to HuggingFace Spaces

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) — **SDK: Docker**, name: `omyfish`
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (role: **write**)
3. Add it as a GitHub secret named `HF_TOKEN` in your repo settings
4. Push to `main` — the deploy workflow handles the rest

## FastAPI backend

```bash
make api                           # http://localhost:8000
# POST /predict  (multipart image)
# GET  /health
```

## Project structure

```
src/            PyTorch training and inference pipeline
app/            Streamlit UI + FastAPI backend + CLIP predictor
configs/        Hyperparameters (config.yaml)
data/metadata/  30-species knowledge base (fish_info.json)
scripts/        Dataset download and organization helpers
checkpoints/    Saved model weights (gitignored)
```
