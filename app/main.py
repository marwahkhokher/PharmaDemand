from pathlib import Path
from typing import List, Dict, Any, Optional

import io
import json

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel


# -------------------------------------------------
# PATHS – aligned with your ML_Project layout
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "demand_regressor.pkl"
METRICS_PATH = REPORTS_DIR / "demand_model_metrics.json"
COOC_PATH = DATA_PROCESSED_DIR / "bundle_cooccurrence.csv"


# -------------------------------------------------
# SAFE STARTUP LOADS (CI-safe: never crash at import)
# -------------------------------------------------
MODEL = None
METRICS: Dict[str, Any] = {}
FEATURE_COLUMNS: List[str] = [
    # fallback (must match compute_features_from_history keys)
    "daily_sales",
    "lag_1",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_mean_14",
    "day_of_week",
    "is_weekend",
    "month",
]
BEST_MODEL_NAME: str = "unknown"

BUNDLE_DF: Optional[pd.DataFrame] = None
RECOMMEND_INDEX: Dict[str, List[Dict[str, float]]] = {}


def build_recommend_index(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    """
    Builds: product -> list[{item, score}] using conf_a_to_b and conf_b_to_a.
    Requires columns: item_a, item_b, conf_a_to_b, conf_b_to_a
    """
    index: Dict[str, List[Dict[str, float]]] = {}

    required_cols = {"item_a", "item_b", "conf_a_to_b", "conf_b_to_a"}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(
            f"bundle_cooccurrence.csv missing required columns. "
            f"Need {sorted(required_cols)}, found {list(df.columns)}"
        )

    for _, row in df.iterrows():
        a = str(row["item_a"]).strip()
        b = str(row["item_b"]).strip()
        conf_a_to_b = float(row["conf_a_to_b"])
        conf_b_to_a = float(row["conf_b_to_a"])

        if conf_a_to_b > 0:
            index.setdefault(a, []).append({"item": b, "score": conf_a_to_b})
        if conf_b_to_a > 0:
            index.setdefault(b, []).append({"item": a, "score": conf_b_to_a})

    for k in index:
        index[k].sort(key=lambda x: x["score"], reverse=True)

    return index


# Load model (safe)
if MODEL_PATH.exists():
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception:
        MODEL = None

# Load metrics (safe)
if METRICS_PATH.exists():
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            METRICS = json.load(f)

        # Only override if present & correct types
        if isinstance(METRICS.get("feature_columns"), list) and METRICS["feature_columns"]:
            FEATURE_COLUMNS = [str(x) for x in METRICS["feature_columns"]]
        if isinstance(METRICS.get("best_model"), str) and METRICS["best_model"]:
            BEST_MODEL_NAME = METRICS["best_model"]

    except Exception:
        METRICS = {}
        # keep fallback FEATURE_COLUMNS / BEST_MODEL_NAME

# Load co-occurrence (safe)
if COOC_PATH.exists():
    try:
        BUNDLE_DF = pd.read_csv(COOC_PATH, dtype={"item_a": str, "item_b": str})
        BUNDLE_DF["item_a"] = BUNDLE_DF["item_a"].astype(str).str.strip()
        BUNDLE_DF["item_b"] = BUNDLE_DF["item_b"].astype(str).str.strip()

        RECOMMEND_INDEX = build_recommend_index(BUNDLE_DF)
    except Exception:
        BUNDLE_DF = None
        RECOMMEND_INDEX = {}


# -------------------------------------------------
# GUARDS (only error when endpoint is called)
# -------------------------------------------------
def require_model():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available at {MODEL_PATH}. Run ml/train.py to generate it (kept local, not in GitHub).",
        )

def require_cooc():
    if not RECOMMEND_INDEX:
        raise HTTPException(
            status_code=503,
            detail=f"Bundle co-occurrence not available/loaded from {COOC_PATH}. Run ml/ingest_kaggle_pharmacy.py to generate it (or commit small artifact if required).",
        )


# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class DemandRequest(BaseModel):
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
    history = list(req.recent_daily_sales)
    if len(history) == 0:
        raise HTTPException(status_code=400, detail="recent_daily_sales cannot be empty.")

    if len(history) < 14:
        history = ([0.0] * (14 - len(history))) + history

    daily_sales = float(history[-1])
    lag_1 = float(history[-2])
    lag_7 = float(history[-7])
    lag_14 = float(history[-14])

    roll_mean_7 = float(np.mean(history[-7:]))
    roll_mean_14 = float(np.mean(history[-14:]))

    dt = pd.to_datetime(req.date)
    day_of_week = int(dt.dayofweek)
    is_weekend = int(day_of_week in (5, 6))
    month = int(dt.month)

    return {
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


def make_feature_vector(req: DemandRequest) -> np.ndarray:
    feat_dict = compute_features_from_history(req)

    try:
        row = [feat_dict[col] for col in FEATURE_COLUMNS]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: missing feature in FEATURE_COLUMNS: {e}. "
                   f"FEATURE_COLUMNS={FEATURE_COLUMNS}",
        )

    return np.array([row], dtype=float)


def recommend_for_basket(basket: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    require_cooc()

    basket_set = {str(b).strip() for b in basket if str(b).strip()}
    if not basket_set:
        return []

    scores: Dict[str, float] = {}

    for item in basket_set:
        for n in RECOMMEND_INDEX.get(item, []):
            candidate = str(n["item"])
            score = float(n["score"])
            if candidate in basket_set:
                continue
            scores[candidate] = scores.get(candidate, 0.0) + score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[: max(int(top_k), 0)]

    return [{"barcode": barcode, "score": float(score)} for barcode, score in top]


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="PharmaDemandOps API",
    description="Demand forecasting + bundle recommendation for pharmacy transactions.",
    version="1.0.0",
)
from app.routes_metrics import router as metrics_router
app.include_router(metrics_router)

from fastapi.staticfiles import StaticFiles
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# -------------------------------------------------
# Deepchecks Themed Report (inject CSS into HTML)
# -------------------------------------------------
DEEPCHECKS_HTML = REPORTS_DIR / "deepchecks" / "deepchecks_report.html"
DEEPCHECKS_THEME = REPORTS_DIR / "deepchecks" / "theme.css"

@app.get("/reports/deepchecks/themed", response_class=HTMLResponse)
def deepchecks_themed(v: str = "latest"):
    if not DEEPCHECKS_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Deepchecks report not found at {DEEPCHECKS_HTML}",
        )

    # theme.css is optional
    css = ""
    if DEEPCHECKS_THEME.exists():
        css = DEEPCHECKS_THEME.read_text(encoding="utf-8", errors="ignore")

    html = DEEPCHECKS_HTML.read_text(encoding="utf-8", errors="ignore")

    # Inject before </head> if possible
    inject = f"<style>{css}</style>"
    if "</head>" in html:
        html = html.replace("</head>", inject + "</head>")
    else:
        html = inject + html

    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "cooc_loaded": bool(RECOMMEND_INDEX),
        "model_name": BEST_MODEL_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "metrics_file_present": METRICS_PATH.exists(),
        "cooc_file_present": COOC_PATH.exists(),
    }


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/predict/demand-next7", response_model=DemandResponse)
def predict_demand_next7(payload: DemandRequest):
    require_model()

    X = make_feature_vector(payload)

    next_day_pred = float(MODEL.predict(X)[0])  # type: ignore[union-attr]
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
    if not req.basket:
        raise HTTPException(status_code=400, detail="Basket cannot be empty.")

    recs = recommend_for_basket(req.basket, top_k=req.top_k)

    return BundleResponse(
        basket=req.basket,
        recommendations=recs,
    )


@app.post("/recommend/from-file")
async def recommend_from_file(file: UploadFile = File(...), top_k: int = 5):
    require_cooc()

    try:
        contents = await file.read()
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            dtype={"Invoice": str, "barcode": str},
        )
        df["Invoice"] = df["Invoice"].astype(str).str.strip()
        df["barcode"] = df["barcode"].astype(str).str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {e}")

    required_cols = {"Invoice", "barcode"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {', '.join(sorted(required_cols))}. Found: {list(df.columns)}",
        )

    results: Dict[str, List[Dict[str, Any]]] = {}
    for invoice_id, group in df.groupby("Invoice"):
        basket = [str(b) for b in group["barcode"].dropna().unique().tolist()]
        results[str(invoice_id)] = recommend_for_basket(basket, top_k=top_k)

    return {"top_k": int(top_k), "results": results}
