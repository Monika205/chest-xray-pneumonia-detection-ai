"""
FastAPI Server for Pediatric Chest X-Ray Pneumonia Detection

Author: Monika
Project: PneumoDetectAI
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="🏥 PneumoDetectAI - Pediatric Pneumonia Detection API",
    description="Clinical-grade AI pneumonia screening: 86% cross-operator validation accuracy, 96.4% sensitivity (485 samples)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.vercel.app", 
        "https://*.netlify.app",  
        "https://*.onrender.com", 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_info = {
    "loaded": False,
    "load_time": None,
    "model_path": None,
    "performance": {
        "accuracy": 86.0,
        "sensitivity": 96.4,
        "specificity": 74.8,
        "false_positive_rate": 25.2,
        "roc_auc": 0.964,
        "pr_auc": 0.968
    }
}

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model, model_info
    try:
        model_paths = [
            Path("../models/best_chest_xray_model.h5"), 
            Path("models/best_chest_xray_model.h5"),
            Path("./best_chest_xray_model.h5")
        ]
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                model_info.update({
                    "loaded": True,
                    "load_time": datetime.now().isoformat(),
                    "model_path": str(model_path)
                })
                logger.info("✅ Model loaded successfully!")
                break
        else:
            logger.error("❌ Model file not found in any expected location")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def interpret_prediction(prediction_score: float) -> dict:
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "Strong indication of pneumonia. Recommend immediate medical attention."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "Moderate indication of pneumonia. Medical review recommended."
        else:
            confidence_level = "Low"
            recommendation = "Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            confidence_level = "Low"
            recommendation = "Unclear result. Manual review by radiologist recommended."
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "recommendation": recommendation,
        "raw_score": float(prediction_score)
    }

@app.get("/")
def read_root():
    return {
        "message": "🏥 PneumoDetectAI pneumonia detection API",
        "status": "running",
        "model_loaded": model_info["loaded"]
    }

@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)[0]
    result = interpret_prediction(prediction)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False
    )
