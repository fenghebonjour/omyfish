# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt          # install all dependencies

make train                                # train the model
make eval                                 # evaluate + save confusion_matrix.png
make app                                  # launch Streamlit UI (port 8501)
make api                                  # launch FastAPI server (port 8000)
make predict IMAGE=path/to/fish.jpg       # CLI prediction

# Dataset helpers (run from repo root)
python research/scripts/download_data.py download crowww/a-large-scale-fish-dataset
python research/scripts/download_data.py organize data/kaggle_tmp --output data/raw
python research/scripts/download_data.py stats

# North American freshwater species (iNaturalist, no auth needed)
make download-na-freshwater                    # 400 images × 8 species
make download-na-freshwater COUNT=600          # custom count
# Single species:
python research/scripts/download_data.py inaturalist \
    --taxon "Micropterus salmoides" --label largemouth_bass --count 400

# Export model for edge deployment
python -c "from services.fish_ai.predictors.efficientnet import FishPredictor; FishPredictor('checkpoints/best.pt').export_onnx()"
```

## Repository Structure

```
apps/
  omyfish_api/        FastAPI backend
    db/engine.py      SQLAlchemy engine + init_db (SQLite or PostGIS)
    main.py           All API routes (monolithic for now — splits in Phase 5)
  omyfish_web/
    main.py           Streamlit frontend

services/
  fish_ai/
    predictors/
      efficientnet.py FishPredictor — trained EfficientNet model
      clip.py         CLIPFishPredictor — zero-shot fallback
    model/
      classifier.py   FishClassifier + build_model (timm backbone)
    training/
      train.py        Training loop (run via make train)
      evaluate.py     Evaluation + confusion matrix (run via make eval)
      dataset.py      FishDataset — auto-split and pre-split layouts
      transforms.py   Albumentations pipelines (train + val)
      utils.py        seed_everything, count_parameters
  gis_service/
    exif.py           extract_exif_gps — EXIF GPS extraction

shared/               Future: domain models, schemas, constants, utils
data/
  raw/                Training images: data/raw/<class_name>/*.jpg
  metadata/           fish_info.json — species knowledge base
configs/
  training.yaml       All hyperparameters for training
checkpoints/          best.pt + classes.json (required for inference)
research/
  scripts/            Standalone helper scripts (download_data.py)
```

## Architecture

### Data layout

`FishDataset` (`services/fish_ai/training/dataset.py`) accepts two folder layouts:

1. **Auto-split** — single flat folder; stratified 80/20 split applied in code:
   ```
   data/raw/<class_name>/*.jpg
   ```
2. **Pre-split** — detected automatically when `train/` and `val/` subdirs exist:
   ```
   data/raw/train/<class_name>/*.jpg
   data/raw/val/<class_name>/*.jpg
   ```

Class names come from folder names (case-sensitive at load time, normalized to lowercase+underscores at lookup time). Folder names must loosely match the `species` keys in `data/metadata/fish_info.json` for metadata to appear — spaces and hyphens are normalized to underscores automatically in `efficientnet.py`.

### Model (`services/fish_ai/model/classifier.py`)

`FishClassifier` wraps a `timm` backbone with `num_classes=0` (raw feature output), then adds a 2-layer classification head. The `embed()` method returns L2-normalized features for similarity-search use cases. Architecture is set in `configs/training.yaml` under `model.architecture`:
- `efficientnet_b3` (default, 300×300 input, best accuracy/speed tradeoff)
- `resnet50`
- `vit_base_patch16_224` (requires `image_size: 224` in config)

### Training flow (`services/fish_ai/training/train.py`)

- `num_classes` is detected from the dataset at runtime and overwrites the config value
- `WeightedRandomSampler` handles class imbalance without oversampling raw data
- Mixed-precision (AMP) enabled automatically when CUDA is available; gradient clipping at norm 1.0
- Saves `checkpoints/best.pt` (highest val accuracy) and `checkpoints/classes.json` alongside it — **both files are required for inference**
- W&B logging: set `logging.use_wandb: true` in config

### Inference flow (`services/fish_ai/predictors/efficientnet.py`)

`FishPredictor` loads the checkpoint and its sibling `classes.json`. Returns top-K predictions with confidence scores and metadata. Flags uncertain results when top confidence < `UNCERTAIN_THRESHOLD` (0.30). The `export_onnx()` method exports to ONNX opset 17 for edge/mobile deployment.

### App layer

- **Streamlit** (`apps/omyfish_web/main.py`): single-page upload → prediction → species card with conservation status color coding. Model is loaded once via `@st.cache_resource`.
- **FastAPI** (`apps/omyfish_api/main.py`): `POST /predict`, `POST /identify-fish`, `GET/POST /observations`, `GET /health`. Predictor is lazy-loaded on first request. CORS is open (`allow_origins=["*"]`) — restrict in production.

### Augmentation (`services/fish_ai/training/transforms.py`)

Training pipeline includes `RandomFog` and `GaussianBlur` to simulate turbid/underwater conditions. Validation uses only resize + ImageNet normalization.

### Evaluation (`services/fish_ai/training/evaluate.py`)

Standalone `evaluate()` prints a per-class classification report and saves a confusion matrix PNG to `outputs/`. The `gradcam_heatmap()` function generates Grad-CAM visualizations for EfficientNet/ResNet backbones; ViT requires a different target layer.

### All hyperparameters

Live in `configs/training.yaml`. The only field that must not be manually set before training is `model.num_classes` — it is overwritten from the dataset automatically.
