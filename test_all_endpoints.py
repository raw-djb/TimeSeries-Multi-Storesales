import requests
import json
import pandas as pd
from typing import Dict, Any
import numpy as np

BASE_URL = "http://localhost:8000"

ALL_FEATURES = [
    "date_block_num",
    "shop_id",
    "item_id",
    "shop_name_split_1_encode",
    "shop_name_split_2_encode",
    "item_category_id",
    "item_cat_split_1_encode",
    "item_cat_split_2_encode",
    "item_type",
    "item_name3",
    "item_cnt_month_lag_1",
    "item_cnt_month_lag_2",
    "item_cnt_month_lag_3",
    "date_avg_item_cnt_lag_1",
    "date_item_avg_item_cnt_lag_1",
    "date_item_avg_item_cnt_lag_2",
    "date_item_avg_item_cnt_lag_3",
    "date_shop_avg_item_cnt_lag_1",
    "date_shop_avg_item_cnt_lag_2",
    "date_shop_avg_item_cnt_lag_3",
    "date_shop_item_avg_item_cnt_lag_1",
    "date_shop_item_avg_item_cnt_lag_2",
    "date_shop_item_avg_item_cnt_lag_3",
    "date_shop_itemcat1_avg_item_cnt_lag_1",
    "date_shop_itemcat1_avg_item_cnt_lag_2",
    "date_shop_itemcat1_avg_item_cnt_lag_3",
    "date_shop_itemcat2_avg_item_cnt_lag_1",
    "date_shop_itemcat2_avg_item_cnt_lag_2",
    "date_shop_itemcat2_avg_item_cnt_lag_3",
    "date_shop_itemcat_avg_item_cnt_lag_1",
    "date_shop_itemcat_avg_item_cnt_lag_2",
    "date_shop_itemcat_avg_item_cnt_lag_3",
    "date_shop_itemtype_avg_item_cnt_lag_1",
    "date_shop_itemtype_avg_item_cnt_lag_2",
    "date_shop_itemtype_avg_item_cnt_lag_3",
    "date_shop_itemname3_avg_item_cnt_lag_1",
    "date_shop_itemname3_avg_item_cnt_lag_2",
    "date_shop_itemname3_avg_item_cnt_lag_3",
    "item_avg_item_price",
    "date_avg_item_price_lag_1",
    "date_avg_item_price_lag_2",
    "date_avg_item_price_lag_3"
]


def create_feature_dict(shop_id: int = 1, item_id: int = 100, date_block_num: int = 34) -> Dict[str, float]:

    features = {feat: 0.0 for feat in ALL_FEATURES}


    features["date_block_num"] = float(date_block_num)
    features["shop_id"] = float(shop_id)
    features["item_id"] = float(item_id)

    return features


def test_health():

    print("\n" + "=" * 60)
    print("TESTING HEALTH ENDPOINT")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f" Health Check Passed")
        print(f"   Status: {data['status']}")
        print(f"   Features: {data['n_features']}")
        return True
    else:
        print(f" Health Check Failed: {response.status_code}")
        return False


def test_single_prediction():

    print("\n" + "=" * 60)
    print("TESTING SINGLE PREDICTION")
    print("=" * 60)

    features = create_feature_dict(shop_id=1, item_id=100)


    features["item_cnt_month_lag_1"] = 5.0
    features["item_cnt_month_lag_2"] = 3.0
    features["item_cnt_month_lag_3"] = 2.0

    payload = {"features": features}

    print(f"Testing prediction for Shop 1, Item 100")
    print(f"Features count: {len(features)}")

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f" Prediction Successful")
        print(f"   Predicted Sales: {data['item_cnt_month']:.2f}")
        return True
    else:
        print(f" Prediction Failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_batch_prediction():

    print("\n" + "=" * 60)
    print("TESTING BATCH PREDICTION")
    print("=" * 60)


    items = [
        {"shop_id": 1, "item_id": 100},
        {"shop_id": 1, "item_id": 200},
        {"shop_id": 2, "item_id": 100},
        {"shop_id": 2, "item_id": 300}
    ]

    rows = []
    for item in items:
        features = create_feature_dict(
            shop_id=item["shop_id"],
            item_id=item["item_id"]
        )
        features["item_cnt_month_lag_1"] = 3.0
        features["item_cnt_month_lag_2"] = 2.0
        features["item_cnt_month_lag_3"] = 1.0
        rows.append(features)

    payload = {"rows": rows}

    print(f"Testing batch prediction for {len(rows)} items")

    response = requests.post(f"{BASE_URL}/predict_batch", json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f" Batch Prediction Successful")
        print(f"   Predictions: {data['predictions']}")
        return True
    else:
        print(f" Batch Prediction Failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_forecast():

    print("\n" + "=" * 60)
    print("TESTING FORECAST ENDPOINT")
    print("=" * 60)

    shop_id = 1
    item_id = 100

    print(f"Testing forecast for Shop {shop_id}, Item {item_id}")

    response = requests.get(f"{BASE_URL}/forecast/{shop_id}/{item_id}")

    if response.status_code == 200:
        data = response.json()
        print(f" Forecast Retrieved")
        print(f"   Shop ID: {data['shop_id']}")
        print(f"   Item ID: {data['item_id']}")
        print(f"   Predicted Sales: {data['item_cnt_month']:.2f}")
        return True
    else:
        print(f"   Forecast Failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_with_real_data():

    print("\n" + "=" * 60)
    print("TESTING WITH REAL DATA")
    print("=" * 60)

    try:

        df = pd.read_pickle("data/processed/data.pkl")


        test_data = df[df.date_block_num == 34].head(5)

        print(f"Testing with {len(test_data)} real samples")

        rows = []
        for _, row in test_data.iterrows():
            features = {}
            for col in ALL_FEATURES:
                val = row[col]

                if pd.isna(val) or np.isinf(val):
                    val = 0.0

                try:
                    features[col] = float(val)
                except (ValueError, OverflowError):
                    features[col] = 0.0
            rows.append(features)

        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json={"rows": rows}
        )

        if response.status_code == 200:
            data = response.json()
            print(f" Real Data Test Successful")
            print(f"   Predictions: {data['predictions']}")
            return True
        else:
            print(f" Real Data Test Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f" Could not load real data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("TIMESERIES MULTI-STORE SALES API TEST SUITE")
    print("=" * 60)
    print(f"API URL: {BASE_URL}")
    print(f"Number of features: {len(ALL_FEATURES)}")


    tests = [
        test_health,
        test_single_prediction,
        test_batch_prediction,
        test_forecast,
        test_with_real_data
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)


    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        name = test.__name__.replace("test_", "").replace("_", " ").title()
        status = "PASSED" if result else "FAILED"
        print(f"{i + 1}. {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED! API IS WORKING PERFECTLY!")
    else:
        print("\n⚠Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()