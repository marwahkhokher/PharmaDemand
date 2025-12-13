# from fastapi import FastAPI
# from fastapi import Response

# app = FastAPI(title="ML Project API")

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.get("/favicon.ico")
# def favicon():
#     return Response(status_code=204)

from pathlib import Path
from typing import List, Dict, Any

import io
import json

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel


# -------------------------------------------------
# PATHS – aligned with your ML_Project layout
# -------------------------------------------------

# This file lives in: ML_Project/app/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "demand_regressor.pkl"
METRICS_PATH = REPORTS_DIR / "demand_model_metrics.json"
COOC_PATH = DATA_PROCESSED_DIR / "bundle_cooccurrence.csv"


# -------------------------------------------------
# LOAD MODEL + METADATA + BUNDLE TABLE AT STARTUP
# (loaded once, reused for all requests)
# -------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run ml/train.py first.")

if not METRICS_PATH.exists():
    raise RuntimeError(f"Metrics file not found at {METRICS_PATH}. Run ml/train.py first.")

if not COOC_PATH.exists():
    raise RuntimeError(
        f"Bundle co-occurrence file not found at {COOC_PATH}. "
        "Run ml/ingest_kaggle_pharmacy.py first."
    )

# Load demand model
MODEL = joblib.load(MODEL_PATH)

# Load metrics + feature columns
with open(METRICS_PATH, "r") as f:
    METRICS = json.load(f)

FEATURE_COLUMNS: List[str] = METRICS["feature_columns"]
BEST_MODEL_NAME: str = METRICS["best_model"]

# Load bundle co-occurrence table
BUNDLE_DF = pd.read_csv(COOC_PATH, dtype={"item_a": str, "item_b": str})
BUNDLE_DF["item_a"] = BUNDLE_DF["item_a"].astype(str).str.strip()
BUNDLE_DF["item_b"] = BUNDLE_DF["item_b"].astype(str).str.strip()

# Build recommendation index: product -> list[(other_product, score)]
# We'll use confidence as the score.
def build_recommend_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    index: Dict[str, List[Dict[str, float]]] = {}

    for _, row in df.iterrows():
        a = str(row["item_a"])
        b = str(row["item_b"])
        conf_a_to_b = float(row["conf_a_to_b"])
        conf_b_to_a = float(row["conf_b_to_a"])

        if conf_a_to_b > 0:
            index.setdefault(a, []).append({"item": b, "score": conf_a_to_b})
        if conf_b_to_a > 0:
            index.setdefault(b, []).append({"item": a, "score": conf_b_to_a})

    # Sort each list by score descending
    for k in index:
        index[k].sort(key=lambda x: x["score"], reverse=True)

    return index


RECOMMEND_INDEX = build_recommend_index(BUNDLE_DF)


# -------------------------------------------------
# Pydantic models for request/response schemas
# -------------------------------------------------

class DemandRequest(BaseModel):
    """
    recent_daily_sales:
        List of recent daily demand values for this product at this pharmacy.
        Can be shorter than 14; we will left-pad with zeros.
        The LAST value is today's demand, older ones come before it.

    date:
        Date for which we are making the prediction (e.g. "2024-12-31").
    """
    pharmacy_id: str
    barcode: str
    date: str
    recent_daily_sales: List[float]


class DemandResponse(BaseModel):
    pharmacy_id: str
    barcode: str
    date: str
    next_day_demand: float
    next_7day_demand: float
    model_name: str


class BundleRequest(BaseModel):
    basket: List[str]
    top_k: int = 5


class BundleResponse(BaseModel):
    basket: List[str]
    recommendations: List[Dict[str, Any]]  # {"barcode": ..., "score": ...}


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def compute_features_from_history(req: DemandRequest) -> Dict[str, float]:
    """
    Turn recent_daily_sales + date into the exact feature set used in training.

    We require at least 1 value in recent_daily_sales.
    If fewer than 14 values, we pad with zeros at the front.
    """
    history = list(req.recent_daily_sales)
    if len(history) == 0:
        raise HTTPException(status_code=400, detail="recent_daily_sales cannot be empty.")

    # Ensure at least 14 days by left-padding with zeros
    if len(history) < 14:
        padding = [0.0] * (14 - len(history))
        history = padding + history

    # Use last X days for different lags/rolling stats
    daily_sales = float(history[-1])
    lag_1 = float(history[-2])
    lag_7 = float(history[-7])
    lag_14 = float(history[-14])

    roll_mean_7 = float(np.mean(history[-7:]))
    roll_mean_14 = float(np.mean(history[-14:]))

    # Date-based features
    dt = pd.to_datetime(req.date)
    day_of_week = int(dt.dayofweek)    # 0=Mon, 6=Sun
    is_weekend = int(day_of_week in (5, 6))
    month = int(dt.month)

    features = {
        "daily_sales": daily_sales,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "roll_mean_7": roll_mean_7,
        "roll_mean_14": roll_mean_14,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": month,
    }
    return features


def make_feature_vector(req: DemandRequest) -> np.ndarray:
    """
    Return a 2D numpy array [[...]] with columns exactly in FEATURE_COLUMNS order.
    """
    feat_dict = compute_features_from_history(req)

    try:
        row = [feat_dict[col] for col in FEATURE_COLUMNS]
    except KeyError as e:
        missing = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Server is misconfigured: missing feature {missing} in compute_features_from_history.",
        )

    return np.array([row], dtype=float)


def recommend_for_basket(basket: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple 'frequently bought together' recommendations using RECOMMEND_INDEX.

    Strategy:
      - For each item in the basket, look up its recommended neighbors.
      - Aggregate scores for each candidate item.
      - Remove items already in the basket.
      - Return top_k by total score.
    """
    if not basket:
        return []

    basket_set = {str(b) for b in basket}

    scores: Dict[str, float] = {}

    for item in basket_set:
        neighbors = RECOMMEND_INDEX.get(str(item), [])
        for n in neighbors:
            candidate = str(n["item"])
            score = float(n["score"])
            if candidate in basket_set:
                continue
            scores[candidate] = scores.get(candidate, 0.0) + score

    # Sort candidates by score descending
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:max(top_k, 0)]

    return [{"barcode": barcode, "score": score} for barcode, score in top]


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="PharmaDemandOps API",
    description="Demand forecasting + bundle recommendation for pharmacy transactions.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """
    Simple health check endpoint.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": BEST_MODEL_NAME,
        "feature_columns": FEATURE_COLUMNS,
    }


@app.post("/predict/demand-next7", response_model=DemandResponse)
def predict_demand_next7(payload: DemandRequest):
    """
    Predict next-day and approximate next-7-day demand for a product.

    NOTE:
      - Model was trained to predict *next-day* demand.
      - We approximate 7-day demand as 7 * next-day prediction
        (this is clearly documented for your report).
    """
    # Build feature vector in the correct order
    X = make_feature_vector(payload)

    # Predict next-day demand
    next_day_pred = float(MODEL.predict(X)[0])

    # Simple approximation for next 7 days
    next_7day_pred = float(next_day_pred * 7.0)

    return DemandResponse(
        pharmacy_id=payload.pharmacy_id,
        barcode=payload.barcode,
        date=payload.date,
        next_day_demand=next_day_pred,
        next_7day_demand=next_7day_pred,
        model_name=BEST_MODEL_NAME,
    )


@app.post("/recommend/bundles", response_model=BundleResponse)
def recommend_bundles(req: BundleRequest):
    """
    Given a list of barcodes in the current basket, return bundle recommendations.
    """
    if not req.basket:
        raise HTTPException(status_code=400, detail="Basket cannot be empty.")

    recs = recommend_for_basket(req.basket, top_k=req.top_k)

    return BundleResponse(
        basket=req.basket,
        recommendations=recs,
    )


@app.post("/recommend/from-file")
async def recommend_from_file(file: UploadFile = File(...), top_k: int = 5):
    """
    Upload a CSV with at least two columns:
      - Invoice
      - barcode

    For each invoice, we compute recommended additional items.
    Returns JSON: { invoice_id: [ {barcode, score}, ... ], ... }
    """
    try:
        contents = await file.read()
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            dtype={"Invoice": str, "barcode": str}
        )
        df["Invoice"] = df["Invoice"].astype(str).str.strip()
        df["barcode"] = df["barcode"].astype(str).str.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {e}")

    required_cols = {"Invoice", "barcode"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {', '.join(required_cols)}",
        )

    results: Dict[str, List[Dict[str, Any]]] = {}

    for invoice_id, group in df.groupby("Invoice"):
        basket = [str(b) for b in group["barcode"].dropna().unique().tolist()]
        recs = recommend_for_basket(basket, top_k=top_k)
        results[str(invoice_id)] = recs

    return {
        "top_k": top_k,
        "results": results,
    }
