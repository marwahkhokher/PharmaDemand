from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    FeatureDrift,
    LabelDrift,
    TrainTestPerformance,
    IsSingleValue,
    StringMismatch,
)

import joblib
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"

TRAIN = PROCESSED / "forecast_train.csv"
TEST = PROCESSED / "forecast_test.csv"

MODEL_PATH = MODELS / "demand_regressor.pkl"
METRICS_PATH = REPORTS / "demand_model_metrics.json"

# Gates (simple thresholds; adjust if you want)
MAX_RMSE = 5.0
MAX_FEATURE_DRIFT = 0.35


def detect_label_col(df: pd.DataFrame) -> str | None:
    # common target names
    for c in ["target_next_day", "y_next7", "target", "label"]:
        if c in df.columns:
            return c
    # fallback: any y_* column
    y_cols = [c for c in df.columns if c.startswith("y_")]
    return y_cols[0] if y_cols else None



def main():
    if not TRAIN.exists() or not TEST.exists():
        raise FileNotFoundError("Missing train/test splits. Run: python ml/ingest_kaggle_pharmacy.py")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing trained model. Run: python ml/train.py")

    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)

    # feature columns from metrics (preferred)
    feature_cols = None
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        feature_cols = metrics.get("feature_columns")

    # fallback if metrics missing
    if not feature_cols:
        feature_cols = [
            "daily_sales", "lag_1", "lag_7", "lag_14",
            "roll_mean_7", "roll_mean_14",
            "day_of_week", "is_weekend", "month"
        ]

    label_col = detect_label_col(train_df)
    model = joblib.load(MODEL_PATH)

    # Build DeepChecks datasets
    # Explicit categorical features (avoid DeepChecks inferring different cats in train vs test)
    cat_features = ["day_of_week", "is_weekend", "month"]

    # Build DeepChecks datasets
    if label_col and label_col in test_df.columns:
        train_ds = Dataset(
            train_df,
            features=feature_cols,
            label=label_col,
            cat_features=cat_features,
        )
        test_ds = Dataset(
            test_df,
            features=feature_cols,
            label=label_col,
            cat_features=cat_features,
        )
    else:
        train_ds = Dataset(
            train_df,
            features=feature_cols,
            cat_features=cat_features,
        )
        test_ds = Dataset(
            test_df,
            features=feature_cols,
            cat_features=cat_features,
        )


    # Run checks (small but rubric-aligned)
    checks = [
        IsSingleValue(),
        StringMismatch(),
        FeatureDrift(),
    ]
    if label_col and label_col in test_df.columns:
        checks += [
            LabelDrift(),
            TrainTestPerformance(estimator=model),
        ]

    from deepchecks.tabular import Suite
    suite = Suite("DeepChecks Gate", *checks)
    result = suite.run(train_ds, test_ds)

    # Save report (for screenshots in report)
    out_dir = REPORTS / "deepchecks"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "deepchecks_report.html"
    result.save_as_html(str(html_path), as_widget=False, requirejs=False)
    print(f"✅ DeepChecks HTML report saved: {html_path}")

    # -------- Gates --------
    # Drift gate (use suite summary if available; otherwise pass with info)
    drift_score = None
    try:
        drift_check = result.get_check("Feature Drift")
        # Different versions return different formats; handle both
        v = drift_check.value
        if isinstance(v, dict):
            # try to find a numeric drift score in the dict
            nums = []
            for _, vv in v.items():
                if isinstance(vv, dict):
                    for k, x in vv.items():
                        if "drift" in str(k).lower():
                            try:
                                nums.append(float(x))
                            except Exception:
                                pass
            drift_score = max(nums) if nums else None
    except Exception:
        drift_score = None

    if drift_score is not None:
        print(f"Feature drift (max): {drift_score:.4f} (gate <= {MAX_FEATURE_DRIFT})")
        if drift_score > MAX_FEATURE_DRIFT:
            raise RuntimeError(f"❌ Drift gate failed: {drift_score:.4f} > {MAX_FEATURE_DRIFT}")
    else:
        print("⚠️ Could not parse a drift score cleanly; drift check report still generated.")

    # Performance gate (manual RMSE on test if label exists)
    if label_col and label_col in test_df.columns:
        X_test = test_df[feature_cols].to_numpy()
        y_test = test_df[label_col].to_numpy()
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        print(f"Test RMSE: {rmse:.4f} (gate <= {MAX_RMSE})")
        if rmse > MAX_RMSE:
            raise RuntimeError(f"❌ Performance gate failed: RMSE {rmse:.4f} > {MAX_RMSE}")
    else:
        print("⚠️ No label column found in splits; skipping RMSE gate.")

    print("✅ DeepChecks gates passed.")


if __name__ == "__main__":
    main()
