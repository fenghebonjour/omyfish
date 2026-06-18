import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
from PIL import Image
from shared.config import settings

st.set_page_config(
    page_title="OMyFish — Fish Species Identifier",
    page_icon="🐟",
    layout="wide",
)

# Reserve scrollbar gutter so the layout doesn't shift 10px when the scrollbar
# appears/disappears during reruns.  Placed after set_page_config (required to
# be first) and before any widget so it lands high in the delta stream.
st.html('<style>html{overflow-y:scroll!important}</style>')

_checkpoint_exists = Path(settings.checkpoint_path).exists()

# ── Token persistence ─────────────────────────────────────────────────────────
# Tokens are stored in two places:
# 1. st.query_params — survives page refresh (URL-based)
# 2. .local_session file — survives server restarts on local dev
#    (on HF Spaces the file is ephemeral, so users re-login after a Space restart)

_SESSION_FILE = ROOT / ".local_session"


def _save_session(token: str):
    try:
        _SESSION_FILE.write_text(token)
    except Exception:
        pass


def _clear_session():
    try:
        _SESSION_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _restore_from_token(token: str) -> bool:
    from jose import JWTError, jwt
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        from apps.omyfish_api.db.engine import ensure_db
        from apps.omyfish_api.repositories.user_repository import UserRepository
        ensure_db()
        u = UserRepository().get_by_id(payload["sub"])
        if u and u["is_active"]:
            st.session_state["auth_user"] = {"id": u["id"], "email": u["email"], "role": u["role"]}
            st.query_params["token"] = token
            return True
    except (JWTError, Exception):
        pass
    return False


if "auth_user" not in st.session_state:
    _token = st.query_params.get("token")
    if not _token and _SESSION_FILE.exists():
        _token = _SESSION_FILE.read_text().strip()
    if _token:
        if not _restore_from_token(_token):
            st.query_params.pop("token", None)
            _clear_session()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def _auth_repo():
    from apps.omyfish_api.db.engine import ensure_db
    from apps.omyfish_api.repositories.user_repository import UserRepository
    ensure_db()
    return UserRepository()


@st.fragment
def _auth_sidebar():
    from apps.omyfish_api.auth import hash_password, verify_password

    user = st.session_state.get("auth_user")

    if user:
        st.markdown(f"**{user['email']}**")
        st.caption(f"Role: {user['role']}")
        if st.button("Log out", use_container_width=True):
            del st.session_state["auth_user"]
            st.query_params.pop("token", None)
            _clear_session()
            st.rerun()
        return

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log in", use_container_width=True, key="btn_login"):
            repo = _auth_repo()
            u = repo.get_by_email(email)
            if u and verify_password(password, u["hashed_password"]) and u["is_active"]:
                st.session_state["auth_user"] = {"id": u["id"], "email": u["email"], "role": u["role"]}
                from apps.omyfish_api.auth import create_access_token
                _tok = create_access_token(u["id"], u["role"])
                st.query_params["token"] = _tok
                _save_session(_tok)
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_register:
        new_email = st.text_input("Email", key="reg_email")
        new_pw = st.text_input("Password", type="password", key="reg_pw")
        if st.button("Create account", use_container_width=True, key="btn_register"):
            repo = _auth_repo()
            if repo.get_by_email(new_email):
                st.error("Email already registered.")
            elif len(new_pw) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                u = repo.create(new_email, hash_password(new_pw))
                st.session_state["auth_user"] = {"id": u["id"], "email": u["email"], "role": u["role"]}
                from apps.omyfish_api.auth import create_access_token
                _tok = create_access_token(u["id"], u["role"])
                st.query_params["token"] = _tok
                _save_session(_tok)
                st.rerun()


with st.sidebar:
    st.header("Account")
    _auth_sidebar()
    st.divider()


@st.cache_resource(show_spinner="Loading model...")
def load_ai_service(model: str):
    from services.fish_ai.service import FishAIService
    if model == "efficientnet":
        from services.fish_ai.predictors.efficientnet import FishPredictor
        return FishAIService(FishPredictor(settings.checkpoint_path, settings.metadata_path), "trained")
    from services.fish_ai.predictors.clip import CLIPFishPredictor
    return FishAIService(CLIPFishPredictor(settings.metadata_path), "clip")


if _checkpoint_exists:
    with st.sidebar:
        st.header("Model")
        _model_choice = st.radio(
            "Backend",
            ["EfficientNet-B3", "CLIP (zero-shot)"],
            index=0,
            help="EfficientNet-B3 is the fine-tuned model (83.6% val accuracy). CLIP is zero-shot with no training.",
        )
    model_key = "efficientnet" if _model_choice == "EfficientNet-B3" else "clip"
else:
    model_key = "clip"

ai_service = load_ai_service(model_key)


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

    auth_user = st.session_state.get("auth_user")
    if not auth_user:
        st.info("Log in to save observations.")

    if st.button("Save Observation", type="primary", disabled=not auth_user):
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
                    user_id=auth_user["id"],
                )
                ObservationRepository().create(obs)
                st.toast(f"Observation saved — {top['species']} at ({lat:.4f}, {lon:.4f})", icon="✅")
                try:
                    st.rerun(scope="app")
                except TypeError:
                    st.rerun()
            except Exception as e:
                st.error(f"Could not save: {e}")


st.title("🐟 OMyFish")

tab_identify, tab_map = st.tabs(["Identify", "Map"])

# ── Identify tab ─────────────────────────────────────────────────────────────

with tab_identify:
    st.caption("Upload a fish photo and AI will identify the species.")

    if ai_service.mode == "clip":
        st.info(
            "Running in **zero-shot CLIP mode** — no custom training needed. "
            + ("Switch to EfficientNet-B3 in the sidebar for the fine-tuned model." if _checkpoint_exists else "Run `make train` with labeled data for a fine-tuned model.")
        )

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        cache_key = f"result_{model_key}_{uploaded.name}_{uploaded.size}"
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
    _map_user = st.session_state.get("auth_user")
    if not _map_user:
        st.info("Log in to view your observations on the map.")
    else:
        try:
            import folium
            from streamlit_folium import st_folium
            from apps.omyfish_api.db.engine import ensure_db
            from apps.omyfish_api.repositories.observation_repository import ObservationRepository

            ensure_db()
            rows = ObservationRepository().list(1000, user_id=_map_user["id"])

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

            st_folium(m, width=None, height=550, returned_objects=[], key=f"map_{_map_user['id']}_{len(rows)}")
            st.caption(f"{len(rows)} observation{'s' if len(rows) != 1 else ''}")

        except ImportError:
            st.info("Install `folium` and `streamlit-folium` to enable the map view.")
        except Exception as e:
            st.error(f"Map error: {e}")
