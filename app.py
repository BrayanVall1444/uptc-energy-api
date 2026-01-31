from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="UPTC Energy API")

TARGET = "energia_total_kwh"

model = load_model("modelo_uptc_chi.keras")
scaler_X = joblib.load("scaler_X_uptc_chi.pkl")
scaler_y = joblib.load("scaler_y_uptc_chi.pkl")

X_COLS = list(scaler_X.feature_names_in_)
FULL_COLS = [TARGET] + X_COLS

@app.get("/")
def root():
    return {"status": "ok", "message": "UPTC Energy API running"}

@app.post("/predict")
def predict(payload: dict):
    try:
        short_window = payload["short_window"]
        long_window = payload["long_window"]
        lags = payload["lags"]

        if len(short_window) != 48 or len(long_window) != 168:
            raise HTTPException(status_code=400, detail="short_window=48 y long_window=168")

        df_short = pd.DataFrame(short_window)
        df_long = pd.DataFrame(long_window)

        if TARGET not in df_short.columns or TARGET not in df_long.columns:
            raise HTTPException(status_code=400, detail="short_window y long_window deben incluir energia_total_kwh")

        for col in X_COLS:
            if col not in df_short.columns:
                df_short[col] = 0.0
            if col not in df_long.columns:
                df_long[col] = 0.0

        df_short = df_short.reindex(columns=FULL_COLS, fill_value=0.0)
        df_long = df_long.reindex(columns=FULL_COLS, fill_value=0.0)

        y_short_scaled = scaler_y.transform(df_short[[TARGET]]).astype(np.float32)
        y_long_scaled = scaler_y.transform(df_long[[TARGET]]).astype(np.float32)

        x_short_scaled = scaler_X.transform(df_short[X_COLS]).astype(np.float32)
        x_long_scaled = scaler_X.transform(df_long[X_COLS]).astype(np.float32)

        Xs = np.concatenate([y_short_scaled, x_short_scaled], axis=1).reshape(1, 48, len(FULL_COLS))
        Xl = np.concatenate([y_long_scaled, x_long_scaled], axis=1).reshape(1, 168, len(FULL_COLS))

        lags_arr = np.array(lags, dtype=np.float32).reshape(-1, 1)
        lags_scaled = scaler_y.transform(lags_arr).reshape(1, 2).astype(np.float32)

        y_pred_scaled = model.predict([Xs, Xl, lags_scaled], verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        return {"sede": "UPTC_CHI", "prediccion_kwh": float(y_pred[0][0])}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
