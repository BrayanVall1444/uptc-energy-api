from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="UPTC Energy API")

model = load_model("modelo_uptc_chi.keras")
scaler_X = joblib.load("scaler_X_uptc_chi.pkl")
scaler_y = joblib.load("scaler_y_uptc_chi.pkl")

X_COLS = scaler_X.feature_names_in_.tolist()

@app.get("/")
def root():
    return {"status": "ok", "message": "UPTC Energy API running"}

@app.post("/predict")
def predict(payload: dict):
    try:
        short_w = payload["short_window"]
        long_w = payload["long_window"]
        lags = payload["lags"]

        if len(short_w) != 48 or len(long_w) != 168:
            raise HTTPException(
                status_code=400,
                detail="short_window debe tener 48 registros y long_window 168 registros"
            )

        df_short = pd.DataFrame(short_w)
        df_long = pd.DataFrame(long_w)

        for col in X_COLS:
            if col not in df_short:
                df_short[col] = 0.0
            if col not in df_long:
                df_long[col] = 0.0

        df_short = df_short[X_COLS]
        df_long = df_long[X_COLS]

        Xs = scaler_X.transform(df_short).astype(np.float32).reshape(1, 48, len(X_COLS))
        Xl = scaler_X.transform(df_long).astype(np.float32).reshape(1, 168, len(X_COLS))
        Xlag = np.array(lags, dtype=np.float32).reshape(1, 2)

        y_pred = model.predict([Xs, Xl, Xlag], verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred)

        return {
            "sede": "UPTC_CHI",
            "prediccion_kwh": float(y_pred[0][0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
