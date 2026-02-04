import streamlit as st
import requests
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PneumoDetectAI",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 PneumoDetectAI")
st.subheader("AI-powered Pediatric Pneumonia Detection")
st.markdown(
    "Upload a pediatric chest X-ray to receive an AI-assisted pneumonia screening result."
)

# --------------------------------------------------
# FASTAPI BACKEND URL
# --------------------------------------------------
# 🔴 CHANGE THIS when you deploy FastAPI (Render / Railway / Local)
API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Analyzing image..."):
            try:
                # Send image to FastAPI
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success(f"🩺 Diagnosis: **{result['diagnosis']}**")
                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence']}%"
                    )

                    st.info(
                        f"**Confidence Level:** {result['confidence_level']}\n\n"
                        f"**Recommendation:** {result['recommendation']}"
                    )

                    st.caption(
                        "⚠️ This tool is for preliminary screening only. "
                        "Always consult a medical professional."
                    )

                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")
