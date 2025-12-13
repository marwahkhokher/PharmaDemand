from pathlib import Path
import pandas as pd

# Project root = one level above /tests
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def test_processed_files_exist():
    """Check the key processed outputs exist."""
    required = [
        "forecast_train.csv",
        "forecast_val.csv",
        "forecast_test.csv",
        "bundle_cooccurrence.csv",
        "forecast_daily_panel.csv",
    ]
    missing = [f for f in required if not (PROCESSED / f).exists()]
    assert not missing, f"Missing processed files: {missing}. Run ml/ingest_kaggle_pharmacy.py first."


def test_forecast_schema_columns():
    """Check forecasting split CSV has the required columns for training."""
    path = PROCESSED / "forecast_train.csv"
    assert path.exists(), "forecast_train.csv missing. Run ml/ingest_kaggle_pharmacy.py."

    df = pd.read_csv(path)

    required_cols = {
        "date",
        "pharmacy_id",
        "barcode",
        "daily_sales",
        "lag_1",
        "lag_7",
        "lag_14",
        "roll_mean_7",
        "roll_mean_14",
        "day_of_week",
        "is_weekend",
        "month",
        "target_next_day",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"forecast_train.csv missing columns: {missing}"


def test_bundle_schema_columns():
    """Check bundle co-occurrence table has required columns."""
    path = PROCESSED / "bundle_cooccurrence.csv"
    assert path.exists(), "bundle_cooccurrence.csv missing. Run ml/ingest_kaggle_pharmacy.py."

    df = pd.read_csv(path)
    required_cols = {"item_a", "item_b", "cooc_count", "support", "conf_a_to_b", "conf_b_to_a"}
    missing = required_cols - set(df.columns)
    assert not missing, f"bundle_cooccurrence.csv missing columns: {missing}"


def test_no_empty_splits():
    """Basic sanity: split files are not empty."""
    for name in ["forecast_train.csv", "forecast_val.csv", "forecast_test.csv"]:
        path = PROCESSED / name
        df = pd.read_csv(path)
        assert len(df) > 0, f"{name} is empty."
