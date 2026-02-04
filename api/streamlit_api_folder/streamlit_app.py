import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="PneumoDetect AI",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "best_chest_xray_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload best_chest_xray_model.h5")
        return None

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # warm-up
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    model.predict(dummy, verbose=0)

    return model


model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# -----------------------------
# Prediction Logic
# -----------------------------
def predict(image):
    processed = preprocess_image(image)
    score = model.predict(processed, verbose=0)[0][0]

    if score > 0.5:
        return {
            "label": "PNEUMONIA",
            "confidence": score * 100,
            "color": "red"
        }
    else:
        return {
            "label": "NORMAL",
            "confidence": (1 - score) * 100,
            "color": "green"
        }


# -----------------------------
# UI
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🫁 PneumoDetect AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-Powered Pneumonia Detection from Chest X-rays</p>",
    unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("🔬 Analyze X-ray"):
        with st.spinner("Analyzing..."):
            start = time.time()
            result = predict(image)
            elapsed = time.time() - start

        st.divider()

        if result["label"] == "PNEUMONIA":
            st.error(f"🩺 **Diagnosis: PNEUMONIA**")
        else:
            st.success(f"✅ **Diagnosis: NORMAL**")

        st.markdown(
            f"""
            **Confidence:** {result['confidence']:.2f}%  
            **Analysis Time:** {elapsed:.2f} seconds
            """
        )

        st.warning(
            "⚠️ This AI tool is for educational purposes only. "
            "Always consult a certified medical professional."
        )

else:
    st.info("👆 Upload a chest X-ray image to begin analysis.")

st.divider()

st.markdown(
    "<p style='text-align:center;'>Developed by <b>Monika</b> | MobileNetV2 | TensorFlow</p>",
    unsafe_allow_html=True
)
