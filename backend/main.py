import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Healthcare Prediction API")

# --- CORS (allow frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")
ANEMIA_MODEL_PATH = os.path.join(MODELS_DIR, "anemia_model.pkl")
HEART_MODEL_PATH  = os.path.join(MODELS_DIR, "heart_model.pkl")

# --- Input Schemas (frontend fields) ---
class AnemiaInput(BaseModel):
    Age: float | None = None
    Hemoglobin: float | None = None
    MCV: float | None = None
    MCH: float | None = None
    MCHC: float | None = None

class HeartInput(BaseModel):
    age: float | None = None
    sex: float | None = None
    cp: float | None = None
    trestbps: float | None = None
    chol: float | None = None
    fbs: float | None = None
    restecg: float | None = None
    thalach: float | None = None
    exang: float | None = None
    oldpeak: float | None = None
    slope: float | None = None
    ca: float | None = None
    thal: float | None = None

# --- Load models (bundles contain {'model', 'features'}) ---
anemia_bundle = joblib.load(ANEMIA_MODEL_PATH) if os.path.exists(ANEMIA_MODEL_PATH) else None
heart_bundle  = joblib.load(HEART_MODEL_PATH)  if os.path.exists(HEART_MODEL_PATH)  else None

@app.get("/")
def root():
    return {"message": "Healthcare Prediction API is running 🚀"}

def prepare_features_from_payload(bundle, payload_dict: dict):
    """
    Build a row vector matching the model's feature order.
    Raise HTTP 400 if any required feature is missing.
    """
    features = bundle["features"]
    row = []
    missing = []
    for f in features:
        v = payload_dict.get(f)
        if v is None:
            missing.append(f)
        else:
            row.append(float(v))
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}. "
                                                   f"Your model expects: {features}")
    return np.array([row])

@app.post("/predict/anemia")
def predict_anemia(data: AnemiaInput):
    if anemia_bundle is None:
        return {"error": "Anemia model not trained yet."}
    payload = data.dict()
    X = prepare_features_from_payload(anemia_bundle, payload)
    pred_int = int(anemia_bundle["model"].predict(X)[0])
    label = "Positive" if pred_int == 1 else "Negative"
    return {"prediction": label}

@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    if heart_bundle is None:
        return {"error": "Heart model not trained yet."}
    payload = data.dict()
    X = prepare_features_from_payload(heart_bundle, payload)
    pred_int = int(heart_bundle["model"].predict(X)[0])
    label = "Positive" if pred_int == 1 else "Negative"
    return {"prediction": label}
