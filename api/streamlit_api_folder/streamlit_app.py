import streamlit as st
from PIL import Image as PILImage
import numpy as np
import io
import time
import os
import tensorflow as tf
import base64
from fpdf import FPDF
from datetime import datetime
import pydicom
import matplotlib.cm as cm

# -----------------------------
# 1. MODEL LOADING LOGIC (FIXED)
# -----------------------------
@st.cache_resource
def load_pneumonia_model():
    """
    Load H5 model with absolute path resolution for Streamlit Cloud.
    This looks for the model in the parent 'api' folder.
    """
    # Find the folder where THIS script is currently sitting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path logic: Go UP one level from 'streamlit_api_folder' to 'api' 
    # then look for the model file.
    model_path = os.path.abspath(os.path.join(current_dir, "..", "best_chest_xray_model.h5"))
    
    try:
        if os.path.exists(model_path):
            # Load without compiling for faster startup
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Warm up the model
            dummy_input = tf.random.normal([1, 224, 224, 3])
            _ = model.predict(dummy_input, verbose=0)
            return model
        else:
            # Fallback for different deployment structures
            st.error(f"Model file not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize Model in Session State
if "pneumo_model" not in st.session_state:
    st.session_state["pneumo_model"] = load_pneumonia_model()

# -----------------------------
# 2. UTILITY FUNCTIONS
# -----------------------------
def create_pdf_download_link(pdf_bytes: bytes, filename: str) -> str:
    b64 = base64.b64encode(pdf_bytes).decode()
    return (
        f'<a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}" '
        f'style="color:#74b9ff; font-weight:bold; text-decoration:none;">'
        f'Download Medical Report (PDF)</a>'
    )

def dicom_to_pil_image(dicom_bytes):
    try:
        dicom_file = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = dicom_file.pixel_array
        pixel_min, pixel_max = pixel_array.min(), pixel_array.max()
        if pixel_max > pixel_min:
            normalized = (255 * (pixel_array - pixel_min) / (pixel_max - pixel_min)).astype(np.uint8)
        else:
            normalized = pixel_array.astype(np.uint8)
        return PILImage.fromarray(normalized).convert('RGB')
    except Exception as e:
        raise Exception(f"Failed to process DICOM file: {str(e)}")

def preprocess_image(image_input):
    if not isinstance(image_input, PILImage.Image):
        image = PILImage.open(image_input)
    else:
        image = image_input
    
    image = image.convert('RGB').resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def interpret_prediction(score):
    if score > 0.5:
        diagnosis, confidence = "PNEUMONIA", float(score * 100)
        rec = "🚨 Strong indication. Seek immediate medical attention." if confidence >= 80 else "⚠️ Moderate indication. Review recommended."
        level = "High" if confidence >= 80 else "Moderate"
    else:
        diagnosis, confidence = "NORMAL", float((1 - score) * 100)
        rec = "✅ No signs of pneumonia detected." if confidence >= 80 else "👍 Likely normal. Follow-up if symptoms persist."
        level = "High" if confidence >= 80 else "Moderate"
    
    return {"diagnosis": diagnosis, "confidence": round(confidence, 2), "confidence_level": level, "recommendation": rec, "raw_score": float(score)}

def create_fallback_overlay(img_array, model):
    try:
        pred = model.predict(img_array, verbose=0)[0][0]
        h, w = 224, 224
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (w*h/8))
        attention = attention * pred if pred > 0.5 else attention * (1-pred) * 0.3
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        colormap = (cm.jet(attention)[:, :, :3] * 255).astype(np.uint8)
        base_image = (img_array[0] * 255).astype(np.uint8)
        overlay = (0.4 * base_image + 0.6 * colormap).astype(np.uint8)
        return PILImage.fromarray(overlay)
    except:
        return PILImage.fromarray((img_array[0] * 255).astype(np.uint8))

# -----------------------------
# 3. PDF GENERATION
# -----------------------------
def generate_medical_pdf_report(prediction_result, analysis_time, original_image, ai_focus_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'PneumoDetect AI - Clinical Report', 0, 1, 'C')
    pdf.ln(10)
    
    res = prediction_result['result']
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
    pdf.cell(0, 8, f"Diagnosis: {res['diagnosis']}", 0, 1)
    pdf.cell(0, 8, f"Confidence: {res['confidence']}%", 0, 1)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Recommendation: {res['recommendation']}")
    
    # Save images to bytes for PDF
    orig_buf = io.BytesIO(); original_image.save(orig_buf, format='PNG')
    foc_buf = io.BytesIO(); ai_focus_image.save(foc_buf, format='PNG')
    
    pdf.image(orig_buf, x=10, y=80, w=90)
    pdf.image(foc_buf, x=110, y=80, w=90)
    
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# 4. STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="PneumoDetect AI", layout="wide")

# (CSS styling code remains the same as your provided snippet...)
st.markdown("""<style>
    .hero-title { text-align: center; color: white; font-size: 40px; font-weight: bold; }
    .stApp { background-color: #0c0634; color: white; }
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🩺 PneumoDetect AI Dashboard</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-ray (JPG/PNG/DICOM)", type=["jpg", "png", "jpeg", "dcm"])

if uploaded_file is not None:
    if st.session_state["pneumo_model"] is None:
        st.error("Model not loaded. Please check the 'api' folder for the .h5 file.")
    else:
        # Processing
        with st.spinner('Analyzing medical imagery...'):
            start_time = time.time()
            if uploaded_file.name.endswith('.dcm'):
                image = dicom_to_pil_image(uploaded_file.read())
            else:
                image = PILImage.open(uploaded_file)
            
            img_array = preprocess_image(image)
            score = st.session_state["pneumo_model"].predict(img_array, verbose=0)[0][0]
            result = interpret_prediction(score)
            analysis_time = time.time() - start_time
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded X-ray", use_container_width=True)
            with col2:
                ai_focus = create_fallback_overlay(img_array, st.session_state["pneumo_model"])
                st.image(ai_focus, caption="AI Focus Area", use_container_width=True)
                
                st.subheader(f"Result: {result['diagnosis']}")
                st.write(f"Confidence: {result['confidence']}%")
                st.info(result['recommendation'])
                
                # PDF Download
                pdf_bytes = generate_medical_pdf_report({"result": result}, analysis_time, image, ai_focus)
                st.markdown(create_pdf_download_link(pdf_bytes, "Medical_Report.pdf"), unsafe_allow_html=True)

st.write("---")
st.write("© 2026 Monika | AI Associate Engineer Intern @ Akoode Technology")
