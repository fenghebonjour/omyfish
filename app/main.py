import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="OMyFish — Fish Species Identifier",
    page_icon="🐟",
    layout="centered",
)


@st.cache_resource
def load_predictor():
    from src.predict import FishPredictor
    return FishPredictor("checkpoints/best.pt", "data/metadata/fish_info.json")


st.title("🐟 OMyFish")
st.caption("Upload a fish photo and AI will identify the species.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)

    with st.spinner("Identifying..."):
        try:
            predictor = load_predictor()
            result = predictor.predict(image, top_k=3)
        except FileNotFoundError:
            st.error("No trained model found at `checkpoints/best.pt`. Run `make train` first.")
            st.stop()

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
