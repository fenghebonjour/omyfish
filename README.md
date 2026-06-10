---
title: OMyFish
emoji: 🐟
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# OMyFish

An AI-powered field companion for anglers, naturalists, and citizen scientists. Photograph a fish, get an instant species identification with ecological details, log the sighting with GPS coordinates, and track all your observations on a world map.

**Live demo:** [huggingface.co/spaces/fenghebonjour/omyfish](https://huggingface.co/spaces/fenghebonjour/omyfish)

---

## What it does

### Identify
Upload a photo and the AI returns the top-3 candidate species with confidence scores and a full ecological profile:

- Scientific name, habitat, diet
- Maximum recorded size
- Conservation status (color-coded: Least Concern → Endangered)
- Species description and a fun fact

The model knows 13 North American freshwater species (83.6 % val accuracy on EfficientNet-B3). A zero-shot CLIP fallback covers cases where no trained checkpoint is present.

### Log observations
After identification, save the sighting with a location:
- GPS coordinates are extracted automatically from the photo's EXIF data when available
- Manual entry for photos without GPS metadata
- Observations are stored in PostgreSQL (Supabase) with species, confidence, coordinates, and timestamp

### Map
All saved observations appear on an interactive world map (Folium). Switch to the **Map** tab to see every sighting — click a marker for species name, confidence, and timestamp.

### API
A FastAPI backend exposes the same capabilities programmatically:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Raw top-K identification |
| `POST /identify-fish` | Identify + extract GPS + optionally save |
| `GET /observations` | List all stored sightings |
| `GET /observations/geojson` | GeoJSON export for external GIS tools |
| `GET /health` | Service health check |

---

## Modes

| Mode | When active | Model |
|---|---|---|
| **Fine-tuned** | `checkpoints/best.pt` present | EfficientNet-B3 trained on your dataset |
| **Zero-shot demo** | No checkpoint | CLIP (`openai/clip-vit-base-patch32`) |

Zero-shot requires no data or training. Fine-tuned mode delivers higher accuracy on the species you train on.

---

## Quick start (local)

```bash
pip install -r requirements.txt
make app          # Streamlit UI at http://localhost:8501
make api          # FastAPI at http://localhost:8000
```

---

## Training a custom model

The training pipeline supports EfficientNet-B3, ResNet-50, and ViT. All hyperparameters live in `configs/training.yaml`.

```bash
# 1. Obtain labeled images — one folder per species
#    data/raw/<species_name>/*.jpg

# Download a public dataset (optional)
python research/scripts/download_data.py download crowww/a-large-scale-fish-dataset
python research/scripts/download_data.py organize data/kaggle_tmp --output data/raw

# Download North American freshwater species from iNaturalist
make download-na-freshwater

# 2. Train
make train                    # saves checkpoints/best.pt + classes.json

# 3. Resume an interrupted run
make resume

# 4. Evaluate
make eval                     # prints per-class report, saves outputs/confusion_matrix.png

# 5. Export for edge / mobile
python -c "from services.fish_ai.predictors.efficientnet import FishPredictor; \
           FishPredictor('checkpoints/best.pt').export_onnx()"
```

Once `checkpoints/best.pt` exists the app switches to fine-tuned mode automatically.

---

## Species knowledge base

`data/metadata/fish_info.json` contains profiles for 39 species. Each entry includes scientific name, habitat, diet, max size, conservation status, description, and a fun fact. Add entries here to enrich the identification cards shown in the UI.

---

## Persistent storage (Supabase)

Set the `DATABASE_URL` environment variable (or HuggingFace Space secret) to a PostgreSQL connection string. Use the **Connection Pooler → Session mode (port 5432)** URL for hosted environments like HF Spaces that lack IPv6 outbound access. The app falls back to SQLite when no `DATABASE_URL` is set.

---

## Deploy to HuggingFace Spaces

1. Create a Space — **SDK: Docker**, name: `omyfish`
2. Generate a write token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add it as a GitHub secret `HF_TOKEN`
4. Push to `main` — the GitHub Actions workflow deploys automatically

---

## Project structure

```
apps/
  omyfish_api/        FastAPI backend (predict, observations, GeoJSON)
  omyfish_web/        Streamlit web UI

services/
  fish_ai/
    model/            EfficientNet-B3 / ResNet-50 / ViT classifier
    predictors/       Inference wrappers (fine-tuned + CLIP zero-shot)
    training/         Training loop, dataset, augmentations, evaluation
  gis_service/        EXIF GPS extraction, GeoJSON builder

shared/               Schemas, config, events

data/
  raw/                Training images: data/raw/<class_name>/*.jpg
  metadata/           fish_info.json — 39-species knowledge base

configs/
  training.yaml       All hyperparameters

checkpoints/          best.pt + classes.json (required for inference)
research/scripts/     Dataset download helpers
```
