from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="UPTC Energy API")

model = load_model("modelo_uptc_chi.keras")
scaler_X = joblib.load("scaler_X_uptc_chi.pkl")
scaler_y = joblib.load("scaler_y_uptc_chi.pkl")

@app.get("/")
def root():
    return {"status": "ok", "message": "UPTC Energy API running"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    X_scaled = scaler_X.transform(df)
    X_scaled = X_scaled.values.astype(np.float32)

    Xs = X_scaled.reshape(1, 48, X_scaled.shape[1])
    Xl = X_scaled.reshape(1, 168, X_scaled.shape[1])
    Xlag = np.zeros((1, 2), dtype=np.float32)

    y_pred = model.predict([Xs, Xl, Xlag], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred)

    return {
        "sede": "UPTC_CHI",
        "prediccion_kwh": float(y_pred[0][0])
    }
