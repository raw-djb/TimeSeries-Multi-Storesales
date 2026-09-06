import os
import json
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor

MODELS_DIR = "models"
PROCESSED_DIR = os.path.join("data", "processed")

app = FastAPI(title="TimeSeries Multi Store Sales API")

model = XGBRegressor()
model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))

with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)

PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, "predictions.csv")
predictions_df = pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else None


class FeatureVector(BaseModel):
    features: Dict[str, float]


class BatchFeatureVectors(BaseModel):
    rows: List[Dict[str, float]]


@app.get("/")
def root():
    return {
        "message": "TimeSeries Multi Store Sales API",
        "version": "1.0.0",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "single_prediction": "/predict (POST)",
            "batch_prediction": "/predict_batch (POST)",
            "forecast": "/forecast/{shop_id}/{item_id} (GET)"
        },
        "status": "API is running successfully",
        "n_features": len(FEATURE_COLUMNS)
    }


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURE_COLUMNS)}


@app.get("/forecast/{shop_id}/{item_id}")
def forecast(shop_id: int, item_id: int):

    if predictions_df is not None:
        row = predictions_df[(predictions_df.shop_id == shop_id) & (predictions_df.item_id == item_id)]
        if not row.empty:
            return {
                "shop_id": shop_id,
                "item_id": item_id,
                "item_cnt_month": float(row.iloc[0]["item_cnt_month"]),
                "source": "precomputed"
            }


    try:
        features = {col: 0.0 for col in FEATURE_COLUMNS}
        features['date_block_num'] = 34.0
        features['shop_id'] = float(shop_id)
        features['item_id'] = float(item_id)

        X = pd.DataFrame([features])[FEATURE_COLUMNS]
        pred = model.predict(X).clip(0, 20)

        return {
            "shop_id": shop_id,
            "item_id": item_id,
            "item_cnt_month": float(pred[0]),
            "source": "model_prediction"
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"no forecast available for shop {shop_id}, item {item_id}: {str(e)}"
        )


@app.post("/predict")
def predict(payload: FeatureVector):
    row = {col: payload.features.get(col, 0.0) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred = model.predict(X).clip(0, 20)
    return {"item_cnt_month": float(pred[0])}


@app.post("/predict_batch")
def predict_batch(payload: BatchFeatureVectors):
    rows = [{col: r.get(col, 0.0) for col in FEATURE_COLUMNS} for r in payload.rows]
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    preds = model.predict(X).clip(0, 20)
    return {"predictions": [float(p) for p in preds]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)