import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
METADATA_PATH = str(ROOT / "data" / "metadata" / "fish_info.json")

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="OMyFish — Fish Species Identifier",
    page_icon="🐟",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    if CHECKPOINT_PATH.exists():
        from src.predict import FishPredictor
        return FishPredictor(str(CHECKPOINT_PATH), METADATA_PATH), "trained"
    from app.clip_predictor import CLIPFishPredictor
    return CLIPFishPredictor(METADATA_PATH), "clip"


predictor, mode = load_predictor()

st.title("🐟 OMyFish")

tab_identify, tab_map = st.tabs(["Identify", "Map"])

# ── Identify tab ─────────────────────────────────────────────────────────────

with tab_identify:
    st.caption("Upload a fish photo and AI will identify the species.")

    if mode == "clip":
        st.info(
            "Running in **zero-shot demo mode** using CLIP — no custom training needed. "
            "Run `make train` with labeled data for a fine-tuned model."
        )

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        cache_key = f"result_{uploaded.name}_{uploaded.size}"
        if cache_key not in st.session_state:
            with st.spinner("Identifying..."):
                st.session_state[cache_key] = predictor.predict(image, top_k=3)
        result = st.session_state[cache_key]

        if result["uncertain"]:
            st.warning(result["message"])

        medals = ["🥇", "🥈", "🥉"]
        for i, pred in enumerate(result["predictions"]):
            pct = pred["confidence"] * 100
            with st.expander(f"{medals[i]} **{pred['species']}** — {pct:.1f}%", expanded=(i == 0)):
                meta = pred["metadata"]
                if meta:
                    c1, c2 = st.columns(2)
                    with c1:
                        if "scientific_name" in meta:
                            st.markdown(f"*{meta['scientific_name']}*")
                        if "habitat" in meta:
                            st.markdown(f"**Habitat:** {meta['habitat']}")
                        if "diet" in meta:
                            st.markdown(f"**Diet:** {meta['diet']}")
                    with c2:
                        if "max_size_cm" in meta:
                            st.markdown(f"**Max size:** {meta['max_size_cm']} cm")
                        if "conservation_status" in meta:
                            status = meta["conservation_status"]
                            icon = "🔴" if "Endangered" in status else "🟡" if "Vulnerable" in status or "Threatened" in status else "🟢"
                            st.markdown(f"**Conservation:** {icon} {status}")
                    if "description" in meta:
                        st.markdown(meta["description"])
                    if "fun_fact" in meta:
                        st.info(f"💡 {meta['fun_fact']}")
                else:
                    st.markdown("*No metadata available for this species.*")
                st.progress(pred["confidence"])

        # ── Save observation ──────────────────────────────────────────────────
        st.divider()
        st.subheader("📍 Save Observation")

        from app.gis import extract_exif_gps
        exif_coords = extract_exif_gps(image)

        if exif_coords:
            st.success(f"GPS found in image EXIF: {exif_coords[0]:.5f}, {exif_coords[1]:.5f}")

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=float(exif_coords[0]) if exif_coords else 0.0, format="%.6f", step=0.0001)
        with col2:
            lon = st.number_input("Longitude", value=float(exif_coords[1]) if exif_coords else 0.0, format="%.6f", step=0.0001)

        if st.button("Save Observation", type="primary"):
            if lat == 0.0 and lon == 0.0 and not exif_coords:
                st.warning("Enter a location before saving.")
            else:
                try:
                    from app.database import IS_POSTGIS, new_id, engine, init_db
                    from sqlalchemy import text
                    init_db()
                    top = result["predictions"][0]
                    meta = top.get("metadata") or {}
                    src = "exif" if exif_coords else "manual"
                    with engine.connect() as conn:
                        if IS_POSTGIS:
                            conn.execute(text("""
                                INSERT INTO observations
                                  (species_name, scientific_name, confidence,
                                   latitude, longitude, geom, source)
                                VALUES
                                  (:species, :sci, :conf, :lat, :lon,
                                   ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                                   :source)
                            """), dict(species=top["species"], sci=meta.get("scientific_name"),
                                       conf=top["confidence"], lat=lat, lon=lon, source=src))
                        else:
                            conn.execute(text("""
                                INSERT INTO observations
                                  (id, species_name, scientific_name, confidence,
                                   latitude, longitude, source)
                                VALUES
                                  (:id, :species, :sci, :conf, :lat, :lon, :source)
                            """), dict(id=new_id(), species=top["species"],
                                       sci=meta.get("scientific_name"),
                                       conf=top["confidence"], lat=lat, lon=lon, source=src))
                        conn.commit()
                    st.success(f"Observation saved — {top['species']} at ({lat:.4f}, {lon:.4f})")
                except Exception as e:
                    st.error(f"Could not save: {e}")

# ── Map tab ───────────────────────────────────────────────────────────────────

with tab_map:
    try:
        import folium
        from streamlit_folium import st_folium
        from app.database import engine, init_db
        from sqlalchemy import text

        init_db()
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT species_name, scientific_name, confidence, timestamp,
                       latitude, longitude
                FROM observations
                ORDER BY timestamp DESC
                LIMIT 1000
            """)).fetchall()

        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
        for r in rows:
            sci = f"<br><i>{r.scientific_name}</i>" if r.scientific_name else ""
            ts = r.timestamp
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts or "")[:16]
            popup_html = (
                f"<b>{r.species_name}</b>{sci}<br>"
                f"{r.confidence * 100:.1f}% confidence<br>"
                f"{ts_str}"
            )
            folium.Marker(
                location=[r.latitude, r.longitude],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=r.species_name,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

        st_folium(m, width=None, height=550, returned_objects=[])
        st.caption(f"{len(rows)} observation{'s' if len(rows) != 1 else ''} stored")

    except ImportError:
        st.info("Install `folium` and `streamlit-folium` to enable the map view.")
    except Exception as e:
        st.error(f"Map error: {e}")
