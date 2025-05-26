"""
Helper used by both CLI scripts and Flask to run inference.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


def _prepare(features: dict) -> np.ndarray:
    """
    Convert an incoming dict to a scaled NumPy array in model feature order.
    """
    df = pd.DataFrame([features])
    arr = _scaler.transform(df)
    return arr


def predict_transaction(features: dict) -> str:
    """
    Returns 'Fraudulent' or 'Legit' for a single transaction represented as a dict.
    """
    arr = _prepare(features)
    pred = _model.predict(arr)  # -1 = anomaly
    return "Fraudulent" if pred[0] == -1 else "Legit"
