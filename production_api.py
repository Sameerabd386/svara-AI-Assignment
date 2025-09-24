from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re
from typing import List
import uvicorn

# Load model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = FastAPI(title="SvaraAI Reply Classifier")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]

label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

@app.get("/")
async def root():
    return {
        "message": "SvaraAI Reply Classifier API",
        "model": "Logistic Regression",
        "accuracy": "95.38%"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    cleaned_text = clean_text(request.text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized)[0].max()

    return PredictionResponse(
        label=label_names[prediction],
        confidence=float(confidence)
    )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    predictions = []
    for text in request.texts:
        cleaned_text = clean_text(text)
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized)[0].max()

        predictions.append({
            "text": text,
            "label": label_names[prediction],
            "confidence": float(confidence)
        })

    return BatchPredictionResponse(predictions=predictions)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)