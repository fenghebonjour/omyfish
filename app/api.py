import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="OMyFish API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor = None
_db_ready = False


def _get_predictor():
    global _predictor
    if _predictor is None:
        from src.predict import FishPredictor
        _predictor = FishPredictor("checkpoints/best.pt", "data/metadata/fish_info.json")
    return _predictor


def _ensure_db():
    global _db_ready
    if not _db_ready:
        try:
            from app.database import init_db
            init_db()
            _db_ready = True
        except Exception:
            pass


@app.on_event("startup")
def startup():
    _ensure_db()


@app.get("/health")
def health():
    return {"status": "ok", "db": _db_ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    image = Image.open(io.BytesIO(await file.read()))
    return _get_predictor().predict(image, top_k=top_k)


@app.post("/identify-fish")
async def identify_fish(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    save: bool = Form(False),
    user_id: Optional[str] = Form(None),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    image = Image.open(io.BytesIO(await file.read()))
    result = _get_predictor().predict(image, top_k=top_k)

    coords = None
    if latitude is not None and longitude is not None:
        coords = (latitude, longitude)
        result["location_source"] = "manual"
    else:
        from app.gis import extract_exif_gps
        exif = extract_exif_gps(image)
        if exif:
            coords = exif
            result["location_source"] = "exif"

    if coords:
        result["latitude"], result["longitude"] = coords

    if save and coords:
        top = result["predictions"][0]
        meta = top.get("metadata") or {}
        result["observation_id"] = _insert_observation(
            species_name=top["species"],
            scientific_name=meta.get("scientific_name"),
            confidence=top["confidence"],
            lat=coords[0],
            lon=coords[1],
            user_id=user_id,
        )

    return result


class ObservationIn(BaseModel):
    species_name: str
    scientific_name: Optional[str] = None
    confidence: float
    latitude: float
    longitude: float
    user_id: Optional[str] = None
    source: str = "manual"


@app.post("/observations")
def create_observation(obs: ObservationIn):
    _ensure_db()
    obs_id = _insert_observation(
        obs.species_name, obs.scientific_name, obs.confidence,
        obs.latitude, obs.longitude, obs.user_id, obs.source,
    )
    return {"id": obs_id, "status": "created"}


@app.get("/observations")
def list_observations(limit: int = 100):
    _ensure_db()
    from app.database import get_db
    with get_db() as db:
        rows = db.execute(
            text("""
                SELECT id, species_name, scientific_name, confidence,
                       timestamp, latitude, longitude, image_url, user_id, source
                FROM observations ORDER BY timestamp DESC LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@app.get("/observations/geojson")
def observations_geojson(limit: int = 1000):
    _ensure_db()
    from app.database import get_db
    with get_db() as db:
        rows = db.execute(
            text("""
                SELECT id, species_name, scientific_name, confidence,
                       timestamp, latitude, longitude, image_url, user_id, source
                FROM observations ORDER BY timestamp DESC LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
            "properties": {k: v for k, v in _row_to_dict(r).items()
                           if k not in ("latitude", "longitude")},
        }
        for r in rows
    ]
    return {"type": "FeatureCollection", "features": features}


# ── helpers ───────────────────────────────────────────────────────────────────

def _insert_observation(species_name, scientific_name, confidence, lat, lon,
                        user_id=None, source="upload", image_url=None):
    from app.database import IS_POSTGIS, new_id, get_db
    _ensure_db()
    with get_db() as db:
        if IS_POSTGIS:
            row = db.execute(
                text("""
                    INSERT INTO observations
                      (species_name, scientific_name, confidence,
                       latitude, longitude, geom, user_id, source, image_url)
                    VALUES
                      (:species, :sci, :conf, :lat, :lon,
                       ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                       :uid, :source, :img)
                    RETURNING id
                """),
                dict(species=species_name, sci=scientific_name, conf=confidence,
                     lat=lat, lon=lon, uid=user_id, source=source, img=image_url),
            ).fetchone()
            return str(row[0])
        else:
            obs_id = new_id()
            db.execute(
                text("""
                    INSERT INTO observations
                      (id, species_name, scientific_name, confidence,
                       latitude, longitude, user_id, source, image_url)
                    VALUES
                      (:id, :species, :sci, :conf, :lat, :lon, :uid, :source, :img)
                """),
                dict(id=obs_id, species=species_name, sci=scientific_name, conf=confidence,
                     lat=lat, lon=lon, uid=user_id, source=source, img=image_url),
            )
            return obs_id


def _row_to_dict(row):
    d = dict(row._mapping)
    ts = d.get("timestamp")
    if ts and hasattr(ts, "isoformat"):
        d["timestamp"] = ts.isoformat()
    if d.get("id"):
        d["id"] = str(d["id"])
    return d
