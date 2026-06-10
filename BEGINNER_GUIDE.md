# Beginner's Guide to omyfish

Welcome to the omyfish project! This guide will help you understand the codebase quickly, whether you're new to machine learning or just new to *this* project.

---

## 🎯 What is omyfish?

**TL;DR:** A fish species classifier that uses deep learning to identify fish from photos. Upload an image → get predictions with confidence scores.

**The system has three main parts:**
1. **AI Brain** — A trained neural network that recognizes fish (in `services/fish_ai/`)
2. **Training Engine** — Code that teaches the brain from examples (in `services/fish_ai/training/`)
3. **User Interfaces** — Ways to use it (web app + REST API in `apps/`)

---

## 📂 The Architecture at a Glance

```
omyfish/
├── services/                    ← The business logic
│   ├── fish_ai/
│   │   ├── model/              # The AI model (neural network)
│   │   ├── predictors/         # Classes that make predictions
│   │   ├── training/           # Code to train the model
│   │   └── tests/              # Model tests
│   ├── gis_service/            # GPS/location utilities
│   └── analytics_service/      # Analytics utilities
├── apps/                        ← User-facing interfaces
│   ├── omyfish_api/            # FastAPI REST backend
│   └── omyfish_web/            # Streamlit web UI
├── shared/                      ← Common utilities
│   ├── schemas/                # Data structures (shared)
│   ├── models/                 # Domain models
│   ├── constants/              # App-wide constants
│   └── config.py               # Configuration
├── data/                        ← Fish photos & metadata
│   ├── raw/                    # Training images
│   └── metadata/               # Species information
├── configs/                     ← Hyperparameters
│   └── training.yaml           # Training settings
├── checkpoints/                 ← Saved models (best.pt + classes.json)
├── CLAUDE.md                    ← Technical reference
└── requirements.txt             ← Python dependencies
```

---

## 🏃 Quick Start (5 minutes)

### Try the Web App
```bash
make app
# Opens http://localhost:8501
# Upload a fish photo → see predictions
```

### Try the API
```bash
make api
# Server runs on http://localhost:8000
# Read docs at http://localhost:8000/docs

# Example prediction:
curl -F "image=@fish.jpg" http://localhost:8000/predict
```

### Train a New Model
```bash
# 1. Add fish photos to data/raw/<species>/
# 2. Run:
make train

# Saves best model to checkpoints/best.pt
```

### Make a Prediction
```bash
make predict IMAGE=path/to/fish.jpg
```

---

## 🗂️ Where to Find Things

| Need | Location |
|------|----------|
| **Model code** | [services/fish_ai/model/classifier.py](services/fish_ai/model/classifier.py) |
| **Training logic** | [services/fish_ai/training/train.py](services/fish_ai/training/train.py) |
| **Making predictions** | [services/fish_ai/predictors/efficientnet.py](services/fish_ai/predictors/efficientnet.py) |
| **Web app** | [apps/omyfish_web/main.py](apps/omyfish_web/main.py) |
| **API routes** | [apps/omyfish_api/routes/](apps/omyfish_api/routes/) |
| **Database setup** | [apps/omyfish_api/db/engine.py](apps/omyfish_api/db/engine.py) |
| **Data loading** | [services/fish_ai/training/dataset.py](services/fish_ai/training/dataset.py) |
| **Settings** | [configs/training.yaml](configs/training.yaml) |
| **Fish species info** | [data/metadata/fish_info.json](data/metadata/fish_info.json) |

---

## 🧠 Core Concepts (The Essentials)

### What's a "Model"?
A model is a trained mathematical function. Think of it as an expert that learned from thousands of examples.

```
Training phase:
  Expert sees 1000 fish photos labeled with species
  → Learns: "Goldfish have X colors, Y shape"

Prediction phase:
  You show unknown photo
  → Expert says: "90% sure that's a goldfish"
```

### Training vs Inference
- **Training** = Teaching the model from examples (slow, happens once)
- **Inference** = Using the trained model to predict (fast, happens many times)

### Data Augmentation
The system artificially modifies training images to create more variety:
- Flip horizontally
- Add fog/blur (simulating underwater conditions)
- Adjust brightness/contrast

Why? So the model learns "goldfish is goldfish" regardless of conditions, not just from specific training examples.

### Class Imbalance
Some fish species have more training photos than others. The training code uses `WeightedRandomSampler` to handle this fairly.

---

## 📚 Learning Path (Pick Your Level)

### For ML Beginners (Start Here!)
**Goal:** Understand what each part does (2-3 hours)

1. Read [README.md](README.md) — 10 min
2. Read [CLAUDE.md](CLAUDE.md) — 20 min (skip deep details, get the overview)
3. Skim [configs/training.yaml](configs/training.yaml) — 5 min
4. Explore [data/metadata/fish_info.json](data/metadata/fish_info.json) — 5 min
5. Run `make app` and play with it — 10 min
6. Read [services/fish_ai/model/classifier.py](services/fish_ai/model/classifier.py) — 20 min
7. Read [services/fish_ai/training/dataset.py](services/fish_ai/training/dataset.py) — 20 min
8. Read [services/fish_ai/training/train.py](services/fish_ai/training/train.py) — 30 min

**You now understand:** What the project does, how data flows, how training works.

---

### For ML Engineers (Want Details?)
**Goal:** Understand the entire architecture (4-6 hours)

Add these to the beginner path:

1. Read [services/fish_ai/training/transforms.py](services/fish_ai/training/transforms.py) — 20 min
   - Data augmentation pipeline

2. Read [services/fish_ai/predictors/efficientnet.py](services/fish_ai/predictors/efficientnet.py) — 30 min
   - How predictions are made
   - Confidence thresholds

3. Read [services/fish_ai/training/evaluate.py](services/fish_ai/training/evaluate.py) — 30 min
   - Model evaluation & metrics
   - Grad-CAM visualization

4. Study [apps/omyfish_api/main.py](apps/omyfish_api/main.py) — 30 min
   - REST API structure
   - Route organization

5. Review [apps/omyfish_api/repositories/](apps/omyfish_api/repositories/) — 20 min
   - Database access patterns

**You now understand:** Everything, including advanced details, architectural patterns, and potential improvements.

---

### For Web Developers (Just Want the Frontend?)
**Goal:** Understand the app layer (1-2 hours)

1. Run `make app` and `make api` — 5 min
2. Read [apps/omyfish_web/main.py](apps/omyfish_web/main.py) — 30 min
   - Streamlit app structure
   - How it calls the model

3. Read [apps/omyfish_api/main.py](apps/omyfish_api/main.py) — 30 min
   - API endpoints
   - How to integrate with the model

4. Explore [apps/omyfish_api/routes/](apps/omyfish_api/routes/) — 20 min
   - Individual endpoint implementations

**You now understand:** How to add new endpoints, modify the UI, call the prediction service.

---

## 🔧 Common Tasks

### Adding a New Fish Species
1. Create folder: `data/raw/<species_name>/`
2. Add `.jpg` images
3. Update `data/metadata/fish_info.json` with species info
4. Run `make train`

### Adding a New API Endpoint
1. Create file in `apps/omyfish_api/routes/my_feature.py`
2. Define your route with FastAPI decorators
3. Import and include it in `apps/omyfish_api/main.py`

### Improving Model Accuracy
1. Adjust hyperparameters in `configs/training.yaml` (batch size, learning rate, num epochs)
2. Add more diverse training data
3. Run `make train` to retrain

### Checking Model Performance
```bash
make eval
# Generates confusion_matrix.png showing accuracy per species
```

---

## 📖 File Navigation Map

**Start here for understanding:**
- Overview: [README.md](README.md)
- Technical details: [CLAUDE.md](CLAUDE.md)

**Core training/prediction:**
- Model architecture: [services/fish_ai/model/classifier.py](services/fish_ai/model/classifier.py)
- Load data: [services/fish_ai/training/dataset.py](services/fish_ai/training/dataset.py)
- Transform data: [services/fish_ai/training/transforms.py](services/fish_ai/training/transforms.py)
- Train model: [services/fish_ai/training/train.py](services/fish_ai/training/train.py)
- Make predictions: [services/fish_ai/predictors/efficientnet.py](services/fish_ai/predictors/efficientnet.py)
- Evaluate: [services/fish_ai/training/evaluate.py](services/fish_ai/training/evaluate.py)

**Apps & APIs:**
- Web UI: [apps/omyfish_web/main.py](apps/omyfish_web/main.py)
- API server: [apps/omyfish_api/main.py](apps/omyfish_api/main.py)
- API routes: [apps/omyfish_api/routes/](apps/omyfish_api/routes/)
- Database: [apps/omyfish_api/db/](apps/omyfish_api/db/)

**Configuration:**
- Training config: [configs/training.yaml](configs/training.yaml)
- Species info: [data/metadata/fish_info.json](data/metadata/fish_info.json)
- Dependencies: [requirements.txt](requirements.txt)

---

## ❓ Troubleshooting

**Q: "Which file should I look at to understand X?"**

- Model predictions → [services/fish_ai/predictors/efficientnet.py](services/fish_ai/predictors/efficientnet.py)
- How training works → [services/fish_ai/training/train.py](services/fish_ai/training/train.py)
- How to add API endpoints → [apps/omyfish_api/routes/](apps/omyfish_api/routes/)
- How web UI works → [apps/omyfish_web/main.py](apps/omyfish_web/main.py)
- Data loading → [services/fish_ai/training/dataset.py](services/fish_ai/training/dataset.py)

**Q: "Where do I modify training settings?"**
A: [configs/training.yaml](configs/training.yaml)

**Q: "How do I see what species the model knows about?"**
A: [data/metadata/fish_info.json](data/metadata/fish_info.json)

**Q: "What Python packages are used?"**
A: [requirements.txt](requirements.txt)

---

## 🎓 Key Takeaways

1. **The flow is simple:** Data → Train → Model → Predict → User sees result
2. **Everything is modular:** Each service/app is independent
3. **Configuration is centralized:** Most settings are in `configs/training.yaml`
4. **Two interfaces to choose from:** Web UI for users, REST API for developers
5. **It's just PyTorch + FastAPI + Streamlit** — standard tools, clean code

---

## 🚀 Next Steps

1. **Run something:** `make app` or `make api` or `make predict IMAGE=test.jpg`
2. **Read the code:** Start with [CLAUDE.md](CLAUDE.md) for a roadmap
3. **Modify & experiment:** Change a hyperparameter and retrain
4. **Ask questions:** Look at the code, search for patterns, trace imports

**Remember:** The best way to learn is by doing. Don't be afraid to read the actual code files!

---

**Happy learning! 🐟**
