from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "model-latest.pkl"

model = None


class PredictionInput(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(input: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    arr = np.array(input.features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": int(pred)}


@app.get("/metrics")
def metrics():
    return {"requests": 1}
