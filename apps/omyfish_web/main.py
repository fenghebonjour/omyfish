import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="OMyFish — Fish Species Identifier",
    page_icon="🐟",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_ai_service():
    from services.fish_ai.service import FishAIService
    from shared.config import settings
    return FishAIService.build(settings.checkpoint_path, settings.metadata_path)


ai_service = load_ai_service()


@st.fragment
def save_observation_form(result, image):
    from services.gis_service.service import GISService
    from apps.omyfish_api.db.engine import ensure_db
    from apps.omyfish_api.repositories.observation_repository import ObservationRepository
    from shared.schemas.observation import ObservationCreate

    gis = GISService()
    exif_coords = gis.extract_gps(image)

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
                ensure_db()
                top = result["predictions"][0]
                meta = top.get("metadata") or {}
                obs = ObservationCreate(
                    species_name=top["species"],
                    scientific_name=meta.get("scientific_name"),
                    confidence=top["confidence"],
                    latitude=lat,
                    longitude=lon,
                    source="exif" if exif_coords else "manual",
                )
                ObservationRepository().create(obs)
                st.toast(f"Observation saved — {top['species']} at ({lat:.4f}, {lon:.4f})", icon="✅")
                try:
                    st.rerun(scope="app")
                except TypeError:
                    st.rerun()
            except Exception as e:
                st.error(f"Could not save: {e}")


st.markdown("""<style>
/* Reserve scrollbar gutter permanently so content width never shifts by 10px.
   scrollbar-gutter:stable only fires on elements with overflow:auto/scroll,
   so it is safe to apply broadly — non-scrolling elements are unaffected. */
*, *::before, *::after { scrollbar-gutter: stable; }
[data-testid="stAppViewContainer"],
.appview-container { overflow-y: scroll !important; }
</style>""", unsafe_allow_html=True)

st.title("🐟 OMyFish")

tab_identify, tab_map = st.tabs(["Identify", "Map"])

# ── Identify tab ─────────────────────────────────────────────────────────────

with tab_identify:
    st.caption("Upload a fish photo and AI will identify the species.")

    if ai_service.mode == "clip":
        st.info(
            "Running in **zero-shot demo mode** using CLIP — no custom training needed. "
            "Run `make train` with labeled data for a fine-tuned model."
        )

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        cache_key = f"result_{uploaded.name}_{uploaded.size}"
        if cache_key not in st.session_state:
            with st.spinner("Identifying..."):
                st.session_state[cache_key] = ai_service.predict(image, top_k=3)
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

        st.divider()
        st.subheader("📍 Save Observation")
        save_observation_form(result, image)

# ── Map tab ───────────────────────────────────────────────────────────────────

with tab_map:
    try:
        import folium
        from streamlit_folium import st_folium
        from apps.omyfish_api.db.engine import ensure_db
        from apps.omyfish_api.repositories.observation_repository import ObservationRepository

        ensure_db()
        rows = ObservationRepository().list(1000)

        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
        for r in rows:
            sci = f"<br><i>{r['scientific_name']}</i>" if r.get("scientific_name") else ""
            ts = r.get("timestamp", "")
            ts_str = ts[:16] if isinstance(ts, str) else (ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts or "")[:16])
            popup_html = (
                f"<b>{r['species_name']}</b>{sci}<br>"
                f"{r['confidence'] * 100:.1f}% confidence<br>"
                f"{ts_str}"
            )
            folium.Marker(
                location=[r["latitude"], r["longitude"]],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=r["species_name"],
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

        st_folium(m, width=None, height=550, returned_objects=[], key=f"map_{len(rows)}")
        st.caption(f"{len(rows)} observation{'s' if len(rows) != 1 else ''} stored")

    except ImportError:
        st.info("Install `folium` and `streamlit-folium` to enable the map view.")
    except Exception as e:
        st.error(f"Map error: {e}")
