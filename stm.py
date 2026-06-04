import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import os

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Railway Track Defect Detection",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --primary:   #1a3a5c;
    --accent:    #0072ce;
    --success:   #00875a;
    --danger:    #c0392b;
    --warning:   #d97706;
    --bg:        #f4f6f9;
    --surface:   #ffffff;
    --border:    #dce3ec;
    --text:      #1a2535;
    --muted:     #6b7a92;
    --radius:    12px;
    --shadow:    0 2px 12px rgba(26,58,92,0.08);
    --shadow-lg: 0 8px 32px rgba(26,58,92,0.13);
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
}

/* App background */
.stApp { background: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1280px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--primary) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e8eef5 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 0 !important; }
[data-testid="stSidebar"] label { color: #b8c8db !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
    color: white !important;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* ── Headings ── */
h1, h2, h3, h4 { color: var(--primary) !important; font-weight: 700 !important; }

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, var(--primary) 0%, #1e5799 60%, #0072ce 100%);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: var(--shadow-lg);
}
.app-header .icon { font-size: 3rem; line-height:1; }
.app-header h1 { color: white !important; font-size: 1.9rem !important; margin: 0 !important; line-height:1.2; }
.app-header p  { color: rgba(255,255,255,0.75) !important; margin: 0.25rem 0 0; font-size: 0.95rem; }
.badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: white !important;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-left: auto;
    white-space: nowrap;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted) !important;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 1rem;
}
.metric-icon {
    width: 48px; height: 48px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
}
.metric-icon.blue   { background: #e8f1fb; }
.metric-icon.red    { background: #fdecea; }
.metric-icon.green  { background: #e6f4ee; }
.metric-value { font-size: 2rem; font-weight: 700; line-height: 1; color: var(--text) !important; }
.metric-label { font-size: 0.78rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

/* ── Status banners ── */
.status-ok {
    background: #eaf6f0;
    border-left: 4px solid var(--success);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    display: flex; align-items: center; gap: 0.8rem;
    margin: 1rem 0;
}
.status-ok .icon { font-size: 1.4rem; }
.status-ok .title { font-weight: 700; color: var(--success) !important; font-size: 1rem; }
.status-ok .sub   { color: #2d7a52 !important; font-size: 0.85rem; margin-top: 2px; }

.status-warn {
    background: #fdf2f2;
    border-left: 4px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    display: flex; align-items: center; gap: 0.8rem;
    margin: 1rem 0;
}
.status-warn .icon { font-size: 1.4rem; }
.status-warn .title { font-weight: 700; color: var(--danger) !important; font-size: 1rem; }
.status-warn .sub   { color: #922b21 !important; font-size: 0.85rem; margin-top: 2px; }

/* ── Confidence bar ── */
.conf-bar-wrap { margin: 1rem 0; }
.conf-bar-label { display: flex; justify-content: space-between; margin-bottom: 6px; }
.conf-bar-label span { font-size: 0.82rem; color: var(--muted) !important; font-weight: 500; }
.conf-bar-label strong { font-size: 0.9rem; color: var(--text) !important; font-family: 'DM Mono', monospace; }
.conf-track { height: 10px; background: #e8ecf1; border-radius: 99px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.conf-fill.green { background: linear-gradient(90deg, #00875a, #34d399); }
.conf-fill.red   { background: linear-gradient(90deg, #c0392b, #f87171); }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #0072ce, #0059a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 2.2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(0,114,206,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0059a0, #004080) !important;
    box-shadow: 0 6px 18px rgba(0,114,206,0.4) !important;
    transform: translateY(-1px);
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: white !important;
    color: var(--accent) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #e8f3fb !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}

/* ── Info box ── */
.info-box {
    background: #eef4fb;
    border: 1px solid #c5d9ee;
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin: 1rem 0;
}
.info-box .icon { font-size: 1.5rem; margin-top: 2px; }
.info-box p { margin: 0; color: #1a3a5c !important; font-size: 0.88rem; line-height: 1.6; }

/* ── Steps list ── */
.steps { list-style: none; padding: 0; margin: 0; }
.steps li {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
    color: var(--text) !important;
}
.steps li:last-child { border-bottom: none; }
.step-num {
    width: 24px; height: 24px;
    background: var(--accent);
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}

/* ── Section divider ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--primary) !important;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Raw data table ── */
.raw-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.raw-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); }
.raw-table td:first-child { color: var(--muted) !important; font-weight: 500; width: 45%; }
.raw-table td:last-child  { font-family: 'DM Mono', monospace; color: var(--text) !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    color: var(--muted) !important;
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer strong { color: var(--primary) !important; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="icon">🚆</div>
    <div>
        <h1>Railway Track Defect Detection</h1>
        <p>AI-powered inspection system using MobileNetV2 deep learning model</p>
    </div>
    <div class="badge">MobileNetV2 · TensorFlow</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Configuration")

default_model_path = os.path.join("output", "railway_defect_detection_model.h5")
model_path = st.sidebar.text_input(
    "Model path (.h5)",
    value=default_model_path,
    help="Path to your trained .h5 file"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
    help="Minimum confidence to accept a prediction"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Specifications")
st.sidebar.markdown("""
**Architecture** — MobileNetV2  
**Input resolution** — 224 × 224 px  
**Task** — Binary classification  
**Classes** — Defective / Non-Defective  
**Training** — ImageNet pre-trained + fine-tuned  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This system uses transfer learning on MobileNetV2 
to classify railway track images as defective 
or non-defective with high accuracy.
""")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        try:
            import tensorflow as tf
            keras = tf.keras
        except ImportError:
            import keras
        from keras.layers import (
            BatchNormalization, Dense, DepthwiseConv2D, Conv2D
        )

        class FixedBatchNorm(BatchNormalization):
            def __init__(self, **kwargs):
                kwargs.pop('renorm', None)
                kwargs.pop('renorm_clipping', None)
                kwargs.pop('renorm_momentum', None)
                super().__init__(**kwargs)

        class FixedDense(Dense):
            def __init__(self, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(**kwargs)

        class FixedDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(**kwargs)

        class FixedConv2D(Conv2D):
            def __init__(self, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(**kwargs)

        model = keras.models.load_model(
            path,
            custom_objects={
                'BatchNormalization': FixedBatchNorm,
                'Dense':              FixedDense,
                'DepthwiseConv2D':    FixedDepthwiseConv2D,
                'Conv2D':             FixedConv2D,
            }
        )
        return model, None
    except Exception as e:
        return None, str(e)

IMG_SIZE = 224

def predict(model, pil_image: Image.Image):
    img_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    img_array  = np.array(img_resized.convert("RGB"), dtype=np.float32) / 255.0
    img_batch  = np.expand_dims(img_array, 0)
    raw        = model.predict(img_batch, verbose=0)[0][0]

    if raw > 0.5:
        label      = "Non-Defective"
        confidence = float(raw)
        color_bgr  = (0, 160, 80)
    else:
        label      = "Defective"
        confidence = float(1.0 - raw)
        color_bgr  = (30, 40, 192)

    orig_cv  = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w     = orig_cv.shape[:2]
    thick    = max(8, w // 55)
    cv2.rectangle(orig_cv, (0, 0), (w-1, h-1), color_bgr, thick)

    banner_h = max(56, h // 9)
    overlay  = orig_cv.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color_bgr, -1)
    cv2.addWeighted(overlay, 0.6, orig_cv, 0.4, 0, orig_cv)

    text  = f"  {label}   {confidence*100:.1f}%"
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = min(w, h) / 550
    thick_t = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick_t)
    ty = (banner_h + th) // 2
    cv2.putText(orig_cv, text, (12, ty), font, scale, (255, 255, 255), thick_t, cv2.LINE_AA)

    result_pil = Image.fromarray(cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB))
    return label, confidence, result_pil, float(raw)

# ── Railway track relevance checker ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_imagenet_classifier():
    try:
        import tensorflow as tf
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        return model
    except Exception:
        return None

RAILWAY_KEYWORDS = {
    'rail', 'track', 'railway', 'railroad', 'train', 'locomotive',
    'steel', 'iron', 'gravel', 'ballast', 'sleeper', 'tie',
    'freight', 'subway', 'metro', 'platform', 'station',
    'wreck', 'rust', 'metal', 'structure', 'beam', 'bridge',
    'road', 'path', 'ground', 'surface', 'pavement', 'lane',
    'stone', 'rock', 'pebble', 'gravel_pit', 'cliff',
}

def is_railway_track(pil_image):
    """Returns (is_relevant: bool, reason: str)"""
    classifier = load_imagenet_classifier()
    if classifier is None:
        return True, ""

    try:
        import tensorflow as tf
        img = pil_image.resize((224, 224)).convert("RGB")
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(np.array(img, dtype=np.float32), 0)
        )
        preds = classifier.predict(arr, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

        top_labels = [label.lower().replace(' ', '_') for (_, label, _) in decoded]
        top_scores = [float(score) for (_, _, score) in decoded]

        for label, score in zip(top_labels, top_scores):
            for keyword in RAILWAY_KEYWORDS:
                if keyword in label:
                    return True, f"Detected: {label} ({score*100:.1f}%)"

        if top_scores[0] < 0.4:
            return True, "Ambiguous image — passing to railway model"

        best = f"{top_labels[0]} ({top_scores[0]*100:.1f}%)"
        return False, best

    except Exception:
        return True, ""

# ── How to use ────────────────────────────────────────────────────────────────
with st.expander("📖  How to Use This System", expanded=False):
    st.markdown("""
<ul class="steps">
  <li><span class="step-num">1</span>Set the correct <strong>model path</strong> in the sidebar (default: <code>output/railway_defect_detection_model.h5</code>)</li>
  <li><span class="step-num">2</span>Upload a <strong>JPG or PNG</strong> image of a railway track</li>
  <li><span class="step-num">3</span>Click <strong>Run Detection</strong> to analyse the image</li>
  <li><span class="step-num">4</span>Review the result, confidence score, and annotated image</li>
  <li><span class="step-num">5</span>Download the annotated image for your records if needed</li>
</ul>
<br>
<div class="info-box">
  <div class="icon">ℹ️</div>
  <p><strong>Note:</strong> This model is a binary image classifier — it analyses the entire image and returns a single Defective / Non-Defective prediction. Adjust the <em>Confidence Threshold</em> in the sidebar if you want to filter out low-confidence predictions.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📁 Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & drop or browse a railway track image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        <div class="icon">📂</div>
        <p>No image uploaded yet. Please select a <strong>JPG or PNG</strong> image of a railway track to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Images side by side
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="card"><div class="card-title">📷 Original Image</div>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Validate model ────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        st.markdown(f"""
        <div class="status-warn">
            <div class="icon">⚠️</div>
            <div>
                <div class="title">Model File Not Found</div>
                <div class="sub">Path: <code>{model_path}</code> — please update the model path in the sidebar.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner("Loading model weights…"):
        model, err = load_model(model_path)

    if err:
        st.error(f"Could not load model: {err}")
        st.stop()

    # ── Detect button ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Run Analysis</div>', unsafe_allow_html=True)
    run_btn = st.button("🔍  Run Detection", type="primary")

    if run_btn:
        # ── Relevance check ───────────────────────────────────────────────────
        with st.spinner("Checking image relevance…"):
            relevant, reason = is_railway_track(pil_image)

        if not relevant:
            st.markdown(f"""
            <div class="status-warn">
                <div class="icon">🚫</div>
                <div>
                    <div class="title">Irrelevant Image Detected</div>
                    <div class="sub">This does not appear to be a railway track image.
                    ImageNet classifier identified it as: <strong>{reason}</strong>.
                    Please upload a railway track image for accurate detection.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        with st.spinner("Running inference on image…"):
            label, confidence, annotated, raw_val = predict(model, pil_image)

        with col2:
            st.markdown('<div class="card"><div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Analysis Summary</div>', unsafe_allow_html=True)

        is_defective = label == "Defective"
        defective_val = 1 if is_defective else 0
        nd_val        = 0 if is_defective else 1
        conf_color    = "red" if is_defective else "green"
        conf_pct      = f"{confidence*100:.1f}"

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-icon blue">🖼️</div>
                <div>
                    <div class="metric-value">1</div>
                    <div class="metric-label">Images Analysed</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon {'red' if is_defective else 'green'}">{'⚠️' if is_defective else '✅'}</div>
                <div>
                    <div class="metric-value" style="color:{'#c0392b' if is_defective else '#00875a'}">{label}</div>
                    <div class="metric-label">Classification Result</div>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon blue">📈</div>
                <div>
                    <div class="metric-value">{conf_pct}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📉 Confidence Score</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label">
                    <span>Model confidence in <strong>{label}</strong> prediction</span>
                    <strong>{conf_pct}%</strong>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {conf_color}" style="width:{conf_pct}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Status ────────────────────────────────────────────────────────────
        if is_defective:
            st.markdown(f"""
            <div class="status-warn">
                <div class="icon">⚠️</div>
                <div>
                    <div class="title">Defective Track Detected — {conf_pct}% confidence</div>
                    <div class="sub">This track section shows signs of defects. Immediate physical inspection is strongly recommended before further use.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-ok">
                <div class="icon">✅</div>
                <div>
                    <div class="title">Track in Good Condition — {conf_pct}% confidence</div>
                    <div class="sub">No defects detected in this section. Track appears structurally sound based on visual analysis.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if confidence < confidence_threshold:
            st.markdown(f"""
            <div class="info-box">
                <div class="icon">⚠️</div>
                <p><strong>Low confidence warning:</strong> The model's confidence ({conf_pct}%) is below your threshold ({confidence_threshold*100:.0f}%). 
                Consider re-examining the image or adjusting the threshold in the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">⬇️ Export</div>', unsafe_allow_html=True)
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        col_dl, col_sp = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="⬇️  Download Annotated Image",
                data=buf.getvalue(),
                file_name="railway_detection_result.png",
                mime="image/png"
            )

        # ── Raw output ────────────────────────────────────────────────────────
        with st.expander("🔬  Technical Details / Raw Output"):
            st.markdown(f"""
<table class="raw-table">
  <tr><td>Predicted Class</td><td>{label}</td></tr>
  <tr><td>Confidence Score</td><td>{confidence*100:.4f}%</td></tr>
  <tr><td>Raw Sigmoid Output</td><td>{raw_val:.6f}</td></tr>
  <tr><td>Decision Threshold</td><td>0.5 (sigmoid midpoint)</td></tr>
  <tr><td>Model Architecture</td><td>MobileNetV2 + Custom Head</td></tr>
  <tr><td>Input Resolution</td><td>{IMG_SIZE} × {IMG_SIZE} px</td></tr>
  <tr><td>Model Path</td><td>{model_path}</td></tr>
  <tr><td>Image File</td><td>{uploaded_file.name}</td></tr>
</table>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Railway Track Defect Detection System</strong> &nbsp;·&nbsp;
    MobileNetV2 Transfer Learning &nbsp;·&nbsp; TensorFlow / Keras<br>
    <span style="font-size:0.75rem; opacity:0.7;">Built for academic research and infrastructure safety monitoring</span>
</div>
""", unsafe_allow_html=True)