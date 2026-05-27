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

# Dataset helpers
python scripts/download_data.py download crowww/a-large-scale-fish-dataset
python scripts/download_data.py organize data/kaggle_tmp --output data/raw
python scripts/download_data.py stats

# Export model for edge deployment
python -c "from src.predict import FishPredictor; FishPredictor('checkpoints/best.pt').export_onnx()"
```

## Architecture

### Data layout

`FishDataset` (`src/dataset.py`) accepts two folder layouts:

1. **Auto-split** — single flat folder; stratified 80/20 split applied in code:
   ```
   data/raw/<class_name>/*.jpg
   ```
2. **Pre-split** — detected automatically when `train/` and `val/` subdirs exist:
   ```
   data/raw/train/<class_name>/*.jpg
   data/raw/val/<class_name>/*.jpg
   ```

Class names come from folder names (case-sensitive at load time, normalized to lowercase+underscores at lookup time). Folder names must loosely match the `species` keys in `data/metadata/fish_info.json` for metadata to appear — spaces and hyphens are normalized to underscores automatically in `predict.py`.

### Model (`src/model.py`)

`FishClassifier` wraps a `timm` backbone with `num_classes=0` (raw feature output), then adds a 2-layer classification head. The `embed()` method returns L2-normalized features for similarity-search use cases. Architecture is set in `configs/config.yaml` under `model.architecture`:
- `efficientnet_b3` (default, 300×300 input, best accuracy/speed tradeoff)
- `resnet50`
- `vit_base_patch16_224` (requires `image_size: 224` in config)

### Training flow (`src/train.py`)

- `num_classes` is detected from the dataset at runtime and overwrites the config value
- `WeightedRandomSampler` handles class imbalance without oversampling raw data
- Mixed-precision (AMP) enabled automatically when CUDA is available; gradient clipping at norm 1.0
- Saves `checkpoints/best.pt` (highest val accuracy) and `checkpoints/classes.json` alongside it — **both files are required for inference**
- W&B logging: set `logging.use_wandb: true` in config

### Inference flow (`src/predict.py`)

`FishPredictor` loads the checkpoint and its sibling `classes.json`. Returns top-K predictions with confidence scores and metadata. Flags uncertain results when top confidence < `UNCERTAIN_THRESHOLD` (0.30). The `export_onnx()` method exports to ONNX opset 17 for edge/mobile deployment.

### App layer

- **Streamlit** (`app/main.py`): single-page upload → prediction → species card with conservation status color coding. Model is loaded once via `@st.cache_resource`.
- **FastAPI** (`app/api.py`): `POST /predict` (multipart image) and `GET /health`. Predictor is lazy-loaded on first request. CORS is open (`allow_origins=["*"]`) — restrict in production.

### Augmentation (`src/transforms.py`)

Training pipeline includes `RandomFog` and `GaussianBlur` to simulate turbid/underwater conditions. Validation uses only resize + ImageNet normalization.

### Evaluation (`src/evaluate.py`)

Standalone `evaluate()` prints a per-class classification report and saves a confusion matrix PNG to `outputs/`. The `gradcam_heatmap()` function generates Grad-CAM visualizations for EfficientNet/ResNet backbones; ViT requires a different target layer.

### All hyperparameters

Live in `configs/config.yaml`. The only field that must not be manually set before training is `model.num_classes` — it is overwritten from the dataset automatically.
