import os
import json
import time

import pandas as pd
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"


def load_data():
    return pd.read_pickle(os.path.join(PROCESSED_DIR, "data.pkl"))


def split_data(data):
    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month'].clip(0, 20)
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month'].clip(0, 20)
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    return X_train, Y_train, X_valid, Y_valid, X_test


def train_model(X_train, Y_train, X_valid, Y_valid):
    model = XGBRegressor(
        max_depth=10,
        n_estimators=1000,
        min_child_weight=0.5,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.1,
        seed=42,
        eval_metric="rmse",
        early_stopping_rounds=20
    )

    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True
    )

    return model

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    data = load_data()
    X_train, Y_train, X_valid, Y_valid, X_test = split_data(data)

    ts = time.time()
    model = train_model(X_train, Y_train, X_valid, Y_valid)
    elapsed = time.time() - ts

    model_path = os.path.join(MODELS_DIR, "xgb_model.json")
    model.save_model(model_path)

    feature_columns = list(X_train.columns)
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    metadata = {
        "train_seconds": elapsed,
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "n_features": len(feature_columns)
    }
    with open(os.path.join(MODELS_DIR, "train_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
