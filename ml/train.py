from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


# -------------------------------------------------
# Paths (aligned with your project)
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    """Load train/val/test CSVs created by ingest_kaggle_pharmacy.py."""
    train_path = DATA_PROCESSED_DIR / "forecast_train.csv"
    val_path = DATA_PROCESSED_DIR / "forecast_val.csv"
    test_path = DATA_PROCESSED_DIR / "forecast_test.csv"

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    val_df = pd.read_csv(val_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    print("Loaded splits:")
    print(" train:", train_df.shape)
    print(" val:", val_df.shape)
    print(" test:", test_df.shape)

    return train_df, val_df, test_df


def prepare_xy(df: pd.DataFrame):
    """
    Given one of the split DataFrames, return X (features) and y (target).
    We will:
      - drop non-numeric / ID columns
      - use target_next_day as y
    """
    target_col = "target_next_day"

    # Columns we DON'T want as features
    drop_cols = [
        "date",
        "pharmacy_id",
        "barcode",
        "target_next_day",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target_col].values

    print("Using feature columns:", feature_cols)
    print("X shape:", X.shape, "y shape:", y.shape)

    return X, y, feature_cols


def evaluate_model(model, X, y, name: str):
    """Return RMSE and MAE for a trained model on given data."""
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae


def main():
    print("=== Training demand forecasting models ===")
    print("Project root:", PROJECT_ROOT)
    print("Processed data dir:", DATA_PROCESSED_DIR)

    # 1) Load data
    train_df, val_df, test_df = load_splits()

    # 2) Prepare X, y
    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    X_test, y_test, _ = prepare_xy(test_df)

    # 3) Baseline model: Linear Regression
    print("\n--- Training baseline: LinearRegression ---")
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    linreg_rmse_val, linreg_mae_val = evaluate_model(linreg, X_val, y_val, "LinearReg (val)")
    linreg_rmse_test, linreg_mae_test = evaluate_model(linreg, X_test, y_test, "LinearReg (test)")

    # 4) Improved model: RandomForestRegressor (simple config for now)
    print("\n--- Training improved: RandomForestRegressor ---")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_rmse_val, rf_mae_val = evaluate_model(rf, X_val, y_val, "RandomForest (val)")
    rf_rmse_test, rf_mae_test = evaluate_model(rf, X_test, y_test, "RandomForest (test)")

    # 5) Decide best model (by validation RMSE)
    print("\n--- Choosing best model (by validation RMSE) ---")
    if rf_rmse_val <= linreg_rmse_val:
        best_model = rf
        best_name = "random_forest"
        best_rmse_val, best_mae_val = rf_rmse_val, rf_mae_val
        best_rmse_test, best_mae_test = rf_rmse_test, rf_mae_test
    else:
        best_model = linreg
        best_name = "linear_regression"
        best_rmse_val, best_mae_val = linreg_rmse_val, linreg_mae_val
        best_rmse_test, best_mae_test = linreg_rmse_test, linreg_mae_test

    print(f"Best model: {best_name}")
    print(f" Val RMSE: {best_rmse_val:.4f}, Val MAE: {best_mae_val:.4f}")
    print(f" Test RMSE: {best_rmse_test:.4f}, Test MAE: {best_mae_test:.4f}")

    # 6) Save best model + metadata
    model_path = MODELS_DIR / "demand_regressor.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")

    metadata = {
        "best_model": best_name,
        "feature_columns": feature_cols,
        "metrics": {
            "val": {
                "rmse": best_rmse_val,
                "mae": best_mae_val,
            },
            "test": {
                "rmse": best_rmse_test,
                "mae": best_mae_test,
            },
        },
    }

    metrics_path = REPORTS_DIR / "demand_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metrics report to: {metrics_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
