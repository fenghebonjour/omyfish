import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="OMyFish API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.predict import FishPredictor
        _predictor = FishPredictor("checkpoints/best.pt", "data/metadata/fish_info.json")
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    return _get_predictor().predict(image, top_k=top_k)
