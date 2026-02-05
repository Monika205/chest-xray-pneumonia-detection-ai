import os
import streamlit as st
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_chest_xray_model.h5"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_chest_xray_model.h5")
    return model

model = load_model()

# --------------------------------------------------
# TITLE & INTRO
# --------------------------------------------------
st.title("🫁 PneumoDetectAI")
st.subheader("AI-powered Pneumonia Detection from Chest X-rays")

st.markdown("""
This application uses a **Deep Learning CNN model** to analyze **pediatric chest X-ray images**
and predict whether **pneumonia is present or not**.

⚠️ *This tool is for educational purposes only and not a medical diagnosis.*
""")

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    st.markdown("### Click below to analyze the X-ray")

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Analyzing X-ray using AI model..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]

        st.divider()

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------
        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"🦠 **Pneumonia Detected**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"✅ **Normal (No Pneumonia Detected)**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.markdown("""
        ### 🩺 Interpretation
        - The model analyzes lung patterns in the X-ray
        - Higher confidence indicates stronger prediction
        - Always consult a medical professional
        """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Developed as a Deep Learning project using CNN & Streamlit")
