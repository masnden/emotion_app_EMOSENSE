import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from utils import get_video_sequence

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Emotion Detection | CNN-BiLSTM",
    page_icon="🎭",
    layout="wide"
)

# ======================================================
# CUSTOM STYLE (ELEGANT UI)
# ======================================================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9aa0a6;
    margin-bottom: 30px;
}

.card {
    background: linear-gradient(135deg, #ff4b4b, #ff9068);
    padding: 30px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0px 10px 35px rgba(0,0,0,0.45);
}

.emotion {
    font-size: 38px;
    font-weight: 700;
}

.confidence {
    font-size: 16px;
    margin-top: 12px;
}

.status-on {
    color: #00ff88;
    font-weight: bold;
}

.status-off {
    color: #ff6b6b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# CONSTANT
# ======================================================
EMOTIONS = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_bilstm_final.h5")

model = load_model()

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="title">🎭 Real-Time Emotion Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">CNN + BiLSTM berbasis urutan ekspresi wajah (20 frame)</div>',
    unsafe_allow_html=True
)

# ======================================================
# CONTROL
# ======================================================
run = st.toggle("▶️ Aktifkan Webcam", value=False)

if run:
    st.markdown('<p class="status-on">Status: RUNNING</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="status-off">Status: IDLE</p>', unsafe_allow_html=True)

st.divider()

# ======================================================
# LAYOUT
# ======================================================
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("### 🎥 Webcam Input")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Emotion Prediction")
    emotion_box = st.empty()
    confidence_bar = st.progress(0)

# ======================================================
# WEBCAM LOOP
# ======================================================
if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # tampilkan frame
        frame_placeholder.image(rgb, channels="RGB")

        # ambil sequence 20 frame
        seq = get_video_sequence(rgb)

        if seq is not None:
            preds = model.predict(seq, verbose=0)
            idx = np.argmax(preds)
            label = EMOTIONS[idx]
            confidence = float(np.max(preds)) * 100

            emotion_box.markdown(
                f"""
                <div class="card">
                    <div class="emotion">{label}</div>
                    <div class="confidence">Confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence_bar.progress(int(confidence))

    cap.release()
