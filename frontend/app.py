"""Streamlit UI for the agro_moisture.ipynb deployment (calls FastAPI /predict)."""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Soil moisture predictor",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_DEFAULT = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.markdown(
    """
<style>
    :root {
        --bg: #0f1419;
        --card: #1a222c;
        --accent: #3d9a6e;
        --accent-dim: #2d6b4f;
        --text: #e8eef4;
        --muted: #8fa3b8;
    }
    .stApp { background: linear-gradient(165deg, #0a0f14 0%, var(--bg) 45%, #121a22 100%); }
    h1 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    .subtitle { color: var(--muted); font-size: 1.05rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }
    div[data-testid="stVerticalBlock"] > div:has(> label) { color: var(--muted); }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
        color: white !important; border: none !important; width: 100%; font-weight: 600;
        padding: 0.65rem 1rem; border-radius: 10px;
    }
    .stButton > button:hover { filter: brightness(1.08); }
    section[data-testid="stSidebar"] { background-color: var(--card); }
    .result-card {
        background: var(--card); border-radius: 14px; padding: 1.25rem 1.5rem;
        border: 1px solid rgba(61,154,110,0.25); margin-top: 1rem;
    }
    .result-value { font-size: 2.4rem; font-weight: 700; color: var(--accent); }
    .result-meta { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

col_title, _ = st.columns([3, 1])
with col_title:
    st.markdown("# 🌱 Soil moisture")
    st.markdown(
        '<p class="subtitle">Live inference from your trained Gradient Boosting model '
        "(ONNX) — same pipeline as <code>agro_moisture.ipynb</code>.</p>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### Connection")
    api_base = st.text_input("API base URL", value=API_DEFAULT, help="In Docker Compose, use http://backend:8000 from other containers; use http://localhost:8000 from your browser.")
    st.caption("Health: `/health` · Metrics: `/metrics`")

crops = ["Beans", "Lettuce", "Maize", "Tomatoes", "Wheat"]

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.05)
        temp_c = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=22.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)
        month = st.slider("Month (seasonality)", 1, 12, 4)
    with c2:
        fert = st.number_input("Fertilizer recommended (kg/ha)", min_value=0.0, value=70.0, step=1.0)
        irrig = st.number_input("Irrigation recommended (mm)", min_value=0.0, value=20.0, step=0.5)
        crop = st.selectbox("Crop type", crops, index=3)
    st.markdown("**Recent soil moisture (%)** — matches notebook lag / rolling features")
    s1, s2, s3 = st.columns(3)
    with s1:
        m1 = st.number_input("Previous day", 0.0, 100.0, 45.0, 0.5, key="lag1")
    with s2:
        m2 = st.number_input("Two days ago", 0.0, 100.0, 44.0, 0.5, key="lag2")
    with s3:
        m3 = st.number_input("Rolling mean (3d)", 0.0, 100.0, 43.0, 0.5, key="roll3")
    submitted = st.form_submit_button("Predict soil moisture")

if submitted:
    url = f"{api_base}/predict"
    payload = {
        "soil_ph": soil_ph,
        "temperature_c": temp_c,
        "humidity_pct": humidity,
        "fertilizer_kg_ha": fert,
        "irrigation_mm": irrig,
        "month": int(month),
        "crop": crop,
        "moisture_lag1": m1,
        "moisture_lag2": m2,
        "moisture_rolling_mean3": m3,
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        pct = data["soil_moisture_percent"]
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-value">{pct:.2f}%</div>'
            f'<div class="result-meta">Model version: {data.get("model_version", "?")} · '
            f'Latency: {data.get("latency_ms", "?")} ms · Features: {data.get("n_features", "?")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.info("Ensure the API is running (`docker compose up`) and the URL matches your setup.")
