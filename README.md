# 🩺 Pediatric Chest X-Ray Pneumonia Detection System
### Advanced AI-Driven Clinical Decision Support | Internship Project @ Akoode Technology

<p align="center">
  <img src="demo/AI_Detects_Pneumonia_Saves_Childhoods.gif" alt="System Preview" style="width: 100%; max-width: 1000px; height: auto; border-radius: 10px;" />
</p>

---

## 🚀 Internship Overview
This project was conceptualized and develop during my professional tenure as an **Intern - AI Associate Engineer** at **Akoode Technology, Gurugram**. 

The primary objective was to move beyond theoretical models and build a **production-ready pipeline** capable of assisting clinicians in pediatric pneumonia screening. My work focused on bridging the gap between high-accuracy deep learning and **clinical interpretability**.

**Key Technical Achievements:**
* **96.4% Sensitivity:** Prioritizing the detection of positive cases to ensure patient safety in screening.
* **Cross-Operator Stability:** Validated the model on 485 independent samples to ensure performance doesn't drop across different hospital equipment.
* **XAI Integration:** Developed a Grad-CAM visualization layer to explain the "why" behind every AI diagnosis.

---

## 🛠️ My Engineering Contributions
To ensure this project met professional standards, I implemented the following specialized modules:

### 1. Robust Data Engineering
I authored custom preprocessing scripts to perform **Histogram Equalization** and **Contrast Limited Adaptive Histogram Equalization (CLAHE)**. This ensures the AI is not confused by variations in X-ray lighting or exposure.

### 2. Explainable AI (XAI)

I integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Instead of a "black box" prediction, the system highlights the specific regions of the lungs where opacities are detected, building trust with medical users.

### 3. Professional Medical Format Support
I utilized the `pydicom` library to allow the system to ingest **.dcm (DICOM)** files directly, extracting critical metadata while maintaining the high-fidelity resolution required for diagnostic accuracy.

---

## 📊 Performance Analytics
| Metric | Result | Clinical Impact |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **96.4%** | Minimizes false negatives (highly safe for screening). |
| **ROC-AUC Score** | **0.964** | Excellent diagnostic discrimination. |
| **Inference Time** | **~1.5s** | Optimized for real-time clinical workflows. |

---

## 🧠 Technology Stack
* **Deep Learning:** TensorFlow, Keras, MobileNetV2.
* **Vision & Metadata:** OpenCV, Pydicom, PIL.
* **Architecture:** FastAPI (Scalable Backend), Streamlit (Intuitive UI).
* **Reporting:** FPDF2 for automated clinical PDF generation.

---

## 📂 Quick Launch Guide

### 1. Clone & Setup
```bash
git clone [https://github.com/Monika205/chest-xray-pneumonia-detection-ai](https://github.com/Monika205/chest-xray-pneumonia-detection-ai)
cd chest-xray-pneumonia-detection-ai/api
python -m venv venv
venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r ../requirements.txt

```

### 3. Run the Dashboard

```bash
streamlit run streamlit_api_folder/streamlit_app.py

```

---

## 🏥 Clinical Disclaimer

This project is a **Research Prototype** developed for educational and internship purposes. It is not an FDA-approved medical device. All outputs should be interpreted as a second opinion by a certified radiologist.

---

## 📞 Professional Connect

**Monika** *Intern - AI Associate Engineer @ Akoode Technology* *B.Tech Data Science @ BML Munjal University*

📧 **Email:** [monikadhingra205@gmail.com](mailto:monikadhingra205@gmail.com)

🐙 **GitHub:** [Monika205](https://github.com/Monika205)

💼 **LinkedIn:** [linkedin-Monika Dhingra](www.linkedin.com/in/monika-dhingra-742b95304)

---

*© 2025 Monika | Developed as part of the Akoode Technology AI Research Initiative.*

```
