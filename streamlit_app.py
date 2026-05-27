import os
import sys
from pathlib import Path

# Root-level entry point for HuggingFace Spaces and Streamlit Cloud.
# app/main.py is the equivalent for running locally from project root.
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
    layout="centered",
)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    if CHECKPOINT_PATH.exists():
        from src.predict import FishPredictor
        return FishPredictor(str(CHECKPOINT_PATH), METADATA_PATH), "trained"
    from app.clip_predictor import CLIPFishPredictor
    return CLIPFishPredictor(METADATA_PATH), "clip"


st.title("🐟 OMyFish")
st.caption("Upload a fish photo and AI will identify the species.")

predictor, mode = load_predictor()

if mode == "clip":
    st.info(
        "Running in **zero-shot demo mode** using CLIP — no custom training needed. "
        "Run `make train` with labeled data for a fine-tuned model."
    )

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)

    with st.spinner("Identifying..."):
        result = predictor.predict(image, top_k=3)

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
