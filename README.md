# Multi-Store Sales Forecasting

A machine learning project that predicts monthly product sales for different shops.

The project uses XGBoost for training and FastAPI to serve predictions through an API.

## Project Structure

```text
.
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── data/
├── models/
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
└── test_all_endpoints.py
```

## Features

* Data preprocessing
* Feature engineering
* Lag features
* XGBoost model
* Sales prediction
* FastAPI API
* Docker support
* API tests

The preprocessing creates features from previous sales, shops, items, categories, and prices.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

Place the dataset files inside:

```text
data/raw/
```

Then run the full pipeline:

```bash
python run.py
```

This runs:

```text
preprocess
train
predict
```

The pipeline runs these three scripts in order.

## Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

The API includes:

```text
GET  /health
POST /predict
POST /predict_batch
GET  /forecast/{shop_id}/{item_id}
```

## Test the API

Start the API, then run:

```bash
python test_all_endpoints.py
```

## Docker

```bash
docker compose up --build
```

The API will run on:

```text
http://localhost:8000
```

## Technologies

* Python
* Pandas
* NumPy
* scikit-learn
* XGBoost
* FastAPI
* Docker
