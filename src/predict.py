import os
import json

import pandas as pd
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"


def load_model():
    model = XGBRegressor()
    model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    return model, feature_columns


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = pd.read_pickle(os.path.join(PROCESSED_DIR, "data.pkl"))
    test_index = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_index.pkl"))

    model, feature_columns = load_model()

    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    X_test = X_test[feature_columns]

    predictions = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test_index.index,
        "item_cnt_month": predictions
    })
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    lookup = pd.DataFrame({
        "shop_id": test_index["shop_id"].values,
        "item_id": test_index["item_id"].values,
        "item_cnt_month": predictions
    })
    lookup_path = os.path.join(PROCESSED_DIR, "predictions.csv")
    lookup.to_csv(lookup_path, index=False)

    print("wrote", submission_path)
    print("wrote", lookup_path)


if __name__ == "__main__":
    main()
