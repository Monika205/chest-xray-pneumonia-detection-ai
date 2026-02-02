import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import time
import base64
from fpdf import FPDF
from datetime import datetime
import pydicom
import matplotlib.cm as cm

# --- 1. PATH FIX FOR GITHUB ---
# This ensures the app finds the model in the folder above it
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Moves up from 'streamlit_api_folder' to 'api' to find the .h5 file
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "best_chest_xray_model.h5")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            # Load with compile=False to avoid version mismatch errors
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        else:
            st.error(f"File Not Found: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- 3. IMAGE PROCESSING ---
def preprocess(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def dicom_to_pil(dicom_bytes):
    dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
    pixel_array = dicom.pixel_array
    rescaled = (255 * (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))).astype(np.uint8)
    return Image.fromarray(rescaled).convert('RGB')

# --- 4. UI DESIGN ---
st.set_page_config(page_title="PneumoDetect AI", layout="centered")

# Custom CSS for your branding
st.markdown("""
    <style>
    .main { background-color: #0c0634; color: white; }
    .stButton>button { background: linear-gradient(135deg, #6366f1, #a855f7); color: white; width: 100%; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 PneumoDetect AI")
st.write("Professional Chest X-Ray Analysis")

# --- 5. MAIN LOGIC ---
model = load_model()

uploaded_file = st.file_uploader("Upload X-Ray", type=['png', 'jpg', 'jpeg', 'dcm'])

if uploaded_file and model:
    # Process image
    if uploaded_file.name.endswith('.dcm'):
        img = dicom_to_pil(uploaded_file.read())
    else:
        img = Image.open(uploaded_file)
    
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    if st.button("ANALYZE X-RAY"):
        with st.spinner("Processing..."):
            processed_img = preprocess(img)
            prediction = model.predict(processed_img, verbose=0)[0][0]
            
            # Result Logic
            label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            conf = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            
            st.subheader(f"Result: {label}")
            st.progress(conf / 100)
            st.write(f"Confidence: {conf:.2f}%")
            
            if label == "PNEUMONIA":
                st.warning("🚨 High probability of Pneumonia detected. Please consult a doctor.")
            else:
                st.success("✅ The X-ray appears to be normal.")

elif not model:
    st.warning("Waiting for model to load... Please check if 'best_chest_xray_model.h5' is in the 'api' folder.")

st.divider()
st.caption("Developed by Monika | AI Associate Engineer")
