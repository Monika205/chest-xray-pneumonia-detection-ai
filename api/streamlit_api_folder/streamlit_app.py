import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import time
import base64
from datetime import datetime
from fpdf import FPDF
import pydicom
import matplotlib.cm as cm

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="PneumoDetect AI",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Load model (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "best_chest_xray_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found")
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
# Image helpers
# -----------------------------
def dicom_to_pil_image(dicom_bytes):
    dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
    arr = dicom.pixel_array
    arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(img):
    arr = preprocess_image(img)
    score = model.predict(arr, verbose=0)[0][0]

    if score > 0.5:
        return "PNEUMONIA", score * 100
    else:
        return "NORMAL", (1 - score) * 100


# -----------------------------
# Simple AI focus (fallback CAM)
# -----------------------------
def create_ai_focus(img):
    base = np.array(img.resize((224, 224)))
    h, w, _ = base.shape
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((x - w//2)**2 + (y - h//2)**2) / (w*h/6))
    heat = cm.jet(mask)[:, :, :3] * 255
    overlay = (0.6 * base + 0.4 * heat).astype(np.uint8)
    return Image.fromarray(overlay)


# -----------------------------
# PDF generator
# -----------------------------
def generate_pdf(result, confidence, image, cam):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PneumoDetect AI - Medical Report", 0, 1, "C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Date: {datetime.now()}", 0, 1)
    pdf.cell(0, 8, f"Diagnosis: {result}", 0, 1)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", 0, 1)
    pdf.ln(5)

    img_buf = io.BytesIO()
    image.save(img_buf, format="PNG")
    img_buf.seek(0)

    cam_buf = io.BytesIO()
    cam.save(cam_buf, format="PNG")
    cam_buf.seek(0)

    pdf.image(img_buf, x=15, y=60, w=80)
    pdf.image(cam_buf, x=110, y=60, w=80)

    output = pdf.output(dest="S")
    return output.encode("latin-1")


def pdf_download_link(pdf_bytes, filename):
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF</a>'


# -----------------------------
# UI
# -----------------------------
st.title("🫁 PneumoDetect AI")
st.caption("AI-powered Pneumonia Detection from Chest X-rays")

uploaded = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png", "dcm"]
)

if uploaded and model:
    if uploaded.name.endswith(".dcm"):
        image = dicom_to_pil_image(uploaded.read())
    else:
        image = Image.open(uploaded)

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("🔬 Analyze"):
        with st.spinner("Analyzing..."):
            start = time.time()
            result, confidence = predict(image)
            elapsed = time.time() - start

        if result == "PNEUMONIA":
            st.error(f"🩺 PNEUMONIA detected ({confidence:.2f}%)")
        else:
            st.success(f"✅ NORMAL ({confidence:.2f}%)")

        st.info(f"⏱ Analysis time: {elapsed:.2f} seconds")

        cam = create_ai_focus(image)
        st.image(cam, caption="AI Focus (illustrative)", use_container_width=True)

        pdf = generate_pdf(result, confidence, image, cam)
        st.markdown(
            pdf_download_link(pdf, "PneumoDetect_Report.pdf"),
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Developed by **Monika** | MobileNetV2 | TensorFlow")
