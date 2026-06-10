"""
Plant Disease Detection System
A deep learning-powered web app for real-time plant disease classification.
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import io

from models.predictor import DiseasePredictor
from utils.preprocessing import preprocess_image
from utils.visualization import (
    plot_confidence_chart,
    overlay_gradcam,
    display_disease_info,
)
from utils.disease_info import DISEASE_INFO, CLASS_NAMES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp { background-color: #0d1117; color: #e6edf3; }

    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .disease-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    .badge-danger  { background:#3d1f1f; color:#f85149; border:1px solid #f85149; }
    .badge-warning { background:#2d2208; color:#e3b341; border:1px solid #e3b341; }
    .badge-success { background:#1a2d1a; color:#3fb950; border:1px solid #3fb950; }
    .metric-box {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .upload-hint { color: #8b949e; font-size: 0.85rem; margin-top: 0.5rem; }
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        color: #58a6ff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stSidebar"] { background: #161b22; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Model Architecture",
        ["EfficientNetB0", "MobileNetV3"],
        help="EfficientNetB0: Higher accuracy | MobileNetV3: Faster inference",
    )
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
        help="Predictions below this threshold are marked uncertain."
    )
    show_gradcam = st.toggle("Show GradCAM Heatmap", value=True)
    show_top_k = st.slider("Show Top-K Predictions", 1, 10, 5)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    info = {
        "EfficientNetB0": {"params": "5.3M", "accuracy": "96.4%", "latency": "~120ms"},
        "MobileNetV3":    {"params": "3.0M", "accuracy": "94.1%", "latency": "~55ms"},
    }[model_choice]

    col1, col2, col3 = st.columns(3)
    for col, (k, v) in zip([col1, col2, col3], info.items()):
        with col:
            st.metric(k.replace("_", " ").title(), v)

    st.markdown("---")
    st.caption("PlantGuard AI v1.0 · Built with TensorFlow & Streamlit")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 PlantGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Real-time plant disease detection powered by deep learning</div>',
    unsafe_allow_html=True,
)

# ── Load predictor ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(model_name: str):
    return DiseasePredictor(model_name=model_name)

with st.spinner(f"Loading {model_choice} model…"):
    predictor = load_predictor(model_choice)

# ── Upload ────────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-hint">Supported: JPG, PNG, WEBP · Max 10 MB</p>',
        unsafe_allow_html=True,
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Disease", use_container_width=True)
    else:
        st.info("Upload a leaf image to get started.")
        analyze_btn = False

# ── Prediction ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    if uploaded and analyze_btn:
        with st.spinner("Running inference…"):
            t0 = time.perf_counter()
            img_array = preprocess_image(image)
            predictions, top_k_labels, top_k_probs = predictor.predict(
                img_array, top_k=show_top_k
            )
            elapsed = (time.perf_counter() - t0) * 1000

        top_label = top_k_labels[0]
        top_prob  = top_k_probs[0]
        disease_data = DISEASE_INFO.get(top_label, {})

        # Severity badge
        severity = disease_data.get("severity", "unknown")
        badge_class = {
            "high":   "badge-danger",
            "medium": "badge-warning",
            "low":    "badge-success",
            "none":   "badge-success",
        }.get(severity, "badge-warning")

        # Summary card
        st.markdown(
            f"""
            <div class="result-card">
                <h3 style="color:#e6edf3;margin:0 0 .5rem">{top_label.replace('_',' ')}</h3>
                <span class="disease-badge {badge_class}">⚠ {severity.upper()} SEVERITY</span>
                <span class="disease-badge" style="background:#1c2a3a;color:#58a6ff;border:1px solid #58a6ff">
                    🎯 {top_prob*100:.1f}% Confidence
                </span>
                <p style="color:#8b949e;font-size:.9rem;margin-top:.8rem">
                    {disease_data.get('description', 'No description available.')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{top_prob*100:.1f}%</div>'
                '<div class="metric-label">Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{elapsed:.0f}ms</div>'
                '<div class="metric-label">Inference Time</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(CLASS_NAMES)}</div>'
                '<div class="metric-label">Disease Classes</div></div>',
                unsafe_allow_html=True,
            )

        # Alert if below threshold
        if top_prob < confidence_threshold:
            st.warning(
                f"⚠ Confidence ({top_prob*100:.1f}%) is below your threshold "
                f"({confidence_threshold*100:.0f}%). Result may be unreliable."
            )

        # Chart
        st.markdown('<div class="section-header">Top Predictions</div>', unsafe_allow_html=True)
        fig = plot_confidence_chart(top_k_labels, top_k_probs)
        st.plotly_chart(fig, use_container_width=True)

        # GradCAM
        if show_gradcam:
            st.markdown('<div class="section-header">GradCAM Heatmap</div>', unsafe_allow_html=True)
            heatmap_img = overlay_gradcam(image, img_array, predictor)
            st.image(heatmap_img, caption="Regions influencing prediction", use_container_width=True)

        # Treatment
        st.markdown('<div class="section-header">Recommended Treatment</div>', unsafe_allow_html=True)
        display_disease_info(disease_data)

    elif not uploaded:
        st.markdown(
            """
            <div class="result-card" style="text-align:center;padding:3rem 1.5rem">
                <div style="font-size:3rem">🔬</div>
                <div style="color:#8b949e;margin-top:1rem">
                    Upload a leaf image to detect diseases using AI
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "PlantGuard AI · Supports 38 disease classes across 14 crop species · "
    "Trained on PlantVillage dataset · Not a substitute for expert agronomic advice."
)
