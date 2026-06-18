# OMyFish — Session Log
**Project:** `/home/bigblue/omyfish` | GitHub: `fenghebonjour/omyfish` | Live: `huggingface.co/spaces/fenghebonjour/omyfish`

---

## Session 2026-06-17 — Enterprise Architecture Expansion

Created two enterprise re-architectures of OMyFish, each as a sibling repo:

### omyfish-dotnet ([github.com/fenghebonjour/omyfish-dotnet](https://github.com/fenghebonjour/omyfish-dotnet))

Full enterprise scaffold with:
- **.NET 10** · ASP.NET Core Minimal APIs · Entity Framework Core · Npgsql
- **YARP** reverse proxy as API gateway
- **Clean Architecture** — Domain / Application / Infrastructure / API layers per service
- **CQRS** via MediatR (`ICommand<T>` / `IQuery<T>` / pipeline behaviors)
- **MassTransit + RabbitMQ** (Quorum Queues, DLQ, retry policies)
- Services: ApiGateway, IdentityService, SpeciesService, ObservationService, NotificationService, AIService (Python reused)
- Real C# domain models: `Species`, `Observation`, `Prediction`, `ConfidenceScore`, `GpsCoordinates`
- SQL migrations (PostGIS DDL + spatial index + radius search helper function)
- Helm chart, K8s HPA YAMLs, GitLab CI/CD pipeline
- `ARCHITECTURE.md` with ASCII diagrams, DDD bounded contexts, scaling table (10→100K users), 6-phase migration roadmap

### omyfish-java ([github.com/fenghebonjour/omyfish-java](https://github.com/fenghebonjour/omyfish-java))

Full enterprise scaffold with:
- **Java 21** · Spring Boot 3.x · Spring Data JPA · Hibernate Spatial · Spring AMQP
- **Hexagonal Architecture** (Ports & Adapters) — `domain/port/in/`, `domain/port/out/`, `adapter/in/`, `adapter/out/`
- **Event-Driven** via RabbitMQ + Spring AMQP (`@RabbitListener`, Quorum Queues)
- **Maven multi-module** parent POM with shared-domain and shared-events modules
- Services: api-gateway, identity-service, species-service, observation-service, notification-service, ai-service (Python reused)
- Real Java domain models: `Species` (AggregateRoot), `Observation`, `ConfidenceScore`, `GpsCoordinates` (records)
- Flyway SQL migrations (PostGIS DDL)
- `ARCHITECTURE.md` with ASCII diagrams, Hexagonal Architecture per-service diagram, JVM tuning notes, scaling table

### Shared decisions across both

- Python AI service (`services/fish_ai/`) reused as independent HTTP microservice in both
- Same infrastructure: PostgreSQL + PostGIS, RabbitMQ Quorum Queues, MinIO / S3, Prometheus + Grafana + Jaeger
- Same frontend: Next.js 15 + TypeScript + MapLibre GL JS
- Same CI/CD: GitLab CI/CD with staging + manual production deploy
- Naming: `omyfish` (Python origin, unchanged) · `omyfish-dotnet` · `omyfish-java`

---

**Date:** 2026-05-27

---

## 1. Project Scaffold

Built a full-stack AI fish species identifier from scratch. Files created:

| File | Purpose |
|---|---|
| `src/transforms.py` | Albumentations pipeline with underwater simulation (fog, blur, color jitter) |
| `src/dataset.py` | `FishDataset` — auto-split or pre-split folder layout, `WeightedRandomSampler` |
| `src/model.py` | `FishClassifier` — timm backbone + 2-layer head, `embed()` for similarity search |
| `src/train.py` | Training loop: AMP, grad clipping, cosine LR, early stopping, optional W&B |
| `src/predict.py` | `FishPredictor` — top-K inference, metadata lookup, ONNX export |
| `src/evaluate.py` | Classification report, confusion matrix PNG, Grad-CAM |
| `src/utils.py` | `seed_everything`, `count_parameters` |
| `app/main.py` | Streamlit UI — auto-detects trained model vs CLIP fallback |
| `app/api.py` | FastAPI — `POST /predict`, `GET /health` |
| `app/clip_predictor.py` | CLIP zero-shot predictor with prompt ensembling |
| `streamlit_app.py` | Root-level entry point for HuggingFace Spaces |
| `data/metadata/fish_info.json` | 33-species knowledge base (habitat, diet, size, conservation, fun facts) |
| `scripts/download_data.py` | Kaggle download + folder organizer + dataset stats |
| `configs/config.yaml` | All hyperparameters |
| `Dockerfile` | Container for HF Spaces (port 7860) |
| `Makefile` | `make train / eval / app / api / predict` |

---

## 2. GitHub Setup

- Initialized git repo, pushed to `https://github.com/fenghebonjour/omyfish`
- **Issue:** First push failed — active gh session was `fenghebonjour` but tried `banfdev/omyfish`
- **Fix:** Changed target to `fenghebonjour/omyfish`
- **Issue:** Workflow files rejected — token missing `workflow` scope
- **Fix:** User pushed directly with PAT: `git push https://fenghebonjour:TOKEN@github.com/...`

---

## 3. GitHub Actions CI/CD

Two workflows added under `.github/workflows/`:

**`ci.yml`** — runs on every push:
- Installs `pyyaml`
- Validates `configs/config.yaml` and `fish_info.json`
- Syntax-checks all Python files with `py_compile`

**`deploy.yml`** — runs on push to `main`:
- Uses `huggingface_hub.upload_folder()` to push files to HF Spaces
- Skips gracefully if `HF_TOKEN` secret is not set

### Bugs fixed along the way
| Error | Cause | Fix |
|---|---|---|
| CI: `ModuleNotFoundError: yaml` | `pyyaml` not installed in runner | Added `pip install pyyaml` step |
| Deploy: `job-level if secrets.HF_TOKEN` | Secrets not available in job-level `if` | Moved check into a step with `GITHUB_OUTPUT` |
| Deploy: silent no-op | `git push --force` fails silently with unrelated histories | Switched to `huggingface_hub.upload_folder()` |
| Deploy: `colorTo: cyan` rejected | HF only allows: red, yellow, green, blue, indigo, purple, pink, gray | Changed to `colorTo: blue` |

---

## 4. HuggingFace Spaces Deployment

### Journey
1. **HF Spaces no longer has a Streamlit SDK** — only Gradio, Docker, Static
2. Created Space with **Docker SDK → Streamlit template**
3. Template created its own `Dockerfile` running `src/streamlit_app.py` (default spiral demo)
4. Our `git push --force` didn't replace Space files — unrelated git histories silently failed
5. Switched deploy to `huggingface_hub.upload_folder()` — works without shared git history
6. `README.md` frontmatter rejected — `colorTo: cyan` not valid → changed to `blue`
7. **App showed "Welcome to Streamlit!"** — HF template's `streamlit_app.py` was running, not ours
8. **Fix:** Created `streamlit_app.py` at repo root (what template's Dockerfile expects), updated our `Dockerfile` to also point to it
9. **403 AxiosError on file upload** — Streamlit's CORS/XSRF protection blocks uploads through HF's reverse proxy
10. **Fix:** Added `--server.enableCORS=false --server.enableXsrfProtection=false` to Dockerfile CMD

### Final working Dockerfile CMD
```dockerfile
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
```

---

## 5. CLIP Predictor Optimization

### Problem
Walleye, American Shad, and Smallmouth Bass failed to identify — they weren't in `fish_info.json`, so CLIP never considered them.

### Changes
**`data/metadata/fish_info.json`** — added 3 new species (now 33 total):
- `walleye` (Sander vitreus) — glassy eyes, low-light predator
- `american_shad` (Alosa sapidissima) — historically vital East Coast food fish
- `smallmouth_bass` (Micropterus dolomieu) — hardest-fighting freshwater sport fish

**`app/clip_predictor.py`** — upgraded from single prompt to **prompt ensembling**:
- 8 prompt templates per species (e.g. *"a photo of a walleye fish"*, *"a walleye swimming in water"*, etc.)
- Text embeddings precomputed and averaged at load time → faster inference
- Direct cosine similarity between image and cached text embeddings

### Bug: `AttributeError: 'BaseModelOutputWithPooling' has no attribute 'norm'`
- **Cause:** `CLIPModel.get_text_features()` returned a wrapped output object in the transformers version on HF Spaces, not a raw tensor
- **Fix:** Replaced `get_text_features()` / `get_image_features()` with direct access to `model.text_model` + `model.text_projection` and `model.vision_model` + `model.visual_projection` — stable across all transformers versions

---

## 6. Tech Stack Summary

| Layer | Technology |
|---|---|
| ML backbone | PyTorch + timm (EfficientNet-B3 / ResNet50 / ViT) |
| Zero-shot demo | CLIP `openai/clip-vit-base-patch32` via HuggingFace transformers |
| Augmentation | albumentations (incl. underwater simulation) |
| UI | Streamlit |
| API | FastAPI |
| Deployment | Docker → HuggingFace Spaces |
| CI/CD | GitHub Actions + huggingface_hub |
| Knowledge base | Hand-curated JSON (33 species) |

---

## 7. Key Commands

```bash
# Local dev
make install
streamlit run app/main.py        # http://localhost:8501

# Training (when you have labeled data)
make train
make eval

# Predict single image
make predict IMAGE=photo.jpg

# Dataset
python scripts/download_data.py download crowww/a-large-scale-fish-dataset
python scripts/download_data.py organize data/kaggle_tmp
python scripts/download_data.py stats
```
