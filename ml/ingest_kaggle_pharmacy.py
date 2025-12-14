from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np

# -------------------------------------------------
# CONFIG – fixed for your layout
# -------------------------------------------------

# This file lives in: ML_Project/ml/ingest_kaggle_pharmacy.py
# So project root is ONE level above "ml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw dataset location (YOUR path)
# Make sure inside this folder you see City1, City2, City3, global_test_set.csv
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "archive" / "PharmacyTransactionalDataset"

# Where we will save cleaned / prepared outputs
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For forecasting: choose one pharmacy + top N products
PHARMACY_ID_FOR_FORECAST = "Ph01_Z01_C01"
TOP_N_PRODUCTS_FOR_FORECAST = 20

# For bundles: how many most frequent products to consider (to keep it fast)
TOP_N_BARCODES_FOR_BUNDLES = 200


def load_all_transactions():
    """
    Read all pharmacy CSVs (City*/Zone*/Ph*.csv), add city/zone/pharmacy_id,
    and return one big DataFrame.
    """
    # All CSVs EXCEPT global_test_set.csv
    csv_files = [
        p for p in DATA_ROOT.rglob("*.csv")
        if "global_test_set" not in p.name
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No per-pharmacy CSV files found under {DATA_ROOT}. "
            "Check that you unzipped archive.zip into "
            "data/raw/archive/PharmacyTransactionalDataset/"
        )

    print(f"Found {len(csv_files)} per-pharmacy CSV files.")

    all_dfs = []

    for path in csv_files:
        # Example path parts: [..., 'City1', 'Zone1', 'Ph01_Z01_C01.csv']
        parts = path.parts
        city = parts[-3]          # e.g. City1
        zone = parts[-2]          # e.g. Zone1
        pharmacy_id = path.stem   # e.g. Ph01_Z01_C01

        df = pd.read_csv(path, dtype={"barcode": "string"})

        df["city"] = city
        df["zone"] = zone
        df["pharmacy_id"] = pharmacy_id

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print("Combined shape (all rows, all columns):", df_all.shape)
    return df_all


def basic_cleaning(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Clean types, dates, dosage_form, drop zero Sales_pack, keep only Drugs.
    """
    print("\n=== Basic cleaning ===")

    # Ensure barcode is string
    df_all["barcode"] = df_all["barcode"].astype("string")

    # Convert addeddate to datetime (dataset style: 1/1/2024 etc.)
    df_all["addeddate"] = pd.to_datetime(df_all["addeddate"], format="%m/%d/%Y")

    # Pure date for daily grouping
    df_all["date"] = df_all["addeddate"].dt.date

    # Optional full datetime using time_
    def safe_make_datetime(row):
        try:
            return pd.to_datetime(
                f"{row['addeddate'].strftime('%Y-%m-%d')} {row['time_']}",
                format="%Y-%m-%d %I:%M%p",
                errors="coerce",
            )
        except Exception:
            return pd.NaT

    df_all["datetime"] = df_all.apply(safe_make_datetime, axis=1)

    # Handle missing dosage_form
    df_all["dosage_form"] = df_all["dosage_form"].fillna("Unknown")

    # Sales_pack as int and drop rows where Sales_pack <= 0
    df_all["Sales_pack"] = df_all["Sales_pack"].fillna(0).astype(int)
    zero_count = (df_all["Sales_pack"] <= 0).sum()
    print(f"Rows with Sales_pack <= 0 (will drop): {zero_count}")

    df_all = df_all[df_all["Sales_pack"] > 0].reset_index(drop=True)

    # Keep only type == "Drug" (drop supplies)
    df_all["type"] = df_all["type"].astype(str)
    before_drugs = df_all.shape[0]
    df_all = df_all[df_all["type"] == "Drug"].reset_index(drop=True)
    after_drugs = df_all.shape[0]
    print(f"Dropped {before_drugs - after_drugs} non-drug rows.")

    print("Cleaned shape:", df_all.shape)

    # Save a copy of the cleaned master transactions
    cleaned_path = OUTPUT_DIR / "master_transactions_cleaned.csv"
    df_all.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned master transactions to: {cleaned_path}")

    return df_all


def make_continuous_daily(df_barcode: pd.DataFrame) -> pd.DataFrame:
    """
    Given rows for ONE barcode and ONE pharmacy with columns:
    ['pharmacy_id', 'barcode', 'date', 'daily_sales'],
    return a DataFrame with continuous daily dates and 0 sales filled.
    """
    df_barcode = df_barcode.sort_values("date").set_index("date")

    # Build full date range
    full_idx = pd.date_range(df_barcode.index.min(), df_barcode.index.max(), freq="D")

    df_barcode = df_barcode.reindex(full_idx)

    # Fill NA sales as 0
    df_barcode["daily_sales"] = df_barcode["daily_sales"].fillna(0)

    # Keep pharmacy_id & barcode
    df_barcode["pharmacy_id"] = df_barcode["pharmacy_id"].iloc[0]
    df_barcode["barcode"] = df_barcode["barcode"].iloc[0]

    # Reset index
    df_barcode = df_barcode.reset_index().rename(columns={"index": "date"})
    return df_barcode


def build_forecasting_dataset(df_all: pd.DataFrame):
    """
    For one pharmacy and top N products, build daily time series with features
    and a next-day demand target. Save to CSV.
    """
    print("\n=== Building forecasting dataset ===")
    ph_id = PHARMACY_ID_FOR_FORECAST

    df_ph = df_all[df_all["pharmacy_id"] == ph_id].copy()
    print(f"Rows for {ph_id}:", df_ph.shape[0])

    # Top N barcodes by total Sales_pack
    top_barcodes = (
        df_ph.groupby("barcode")["Sales_pack"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N_PRODUCTS_FOR_FORECAST)
        .index
    )
    top_barcodes = list(top_barcodes)
    print(f"Top {TOP_N_PRODUCTS_FOR_FORECAST} barcodes for forecasting:", top_barcodes)

    df_ph_top = df_ph[df_ph["barcode"].isin(top_barcodes)].copy()

    # Aggregate to daily_sales
    daily = (
        df_ph_top
        .groupby(["pharmacy_id", "barcode", "date"], as_index=False)["Sales_pack"]
        .sum()
    )
    daily = daily.rename(columns={"Sales_pack": "daily_sales"})
    print("Daily aggregated shape:", daily.shape)

    # Make continuous timeline for each barcode
    continuous_list = []

    for bc in top_barcodes:
        df_bc = daily[(daily["pharmacy_id"] == ph_id) & (daily["barcode"] == bc)]
        if df_bc.empty:
            continue
        df_bc_cont = make_continuous_daily(df_bc)
        continuous_list.append(df_bc_cont)

    if not continuous_list:
        raise ValueError("No data found for selected pharmacy/barcodes.")

    daily_panel = pd.concat(continuous_list, ignore_index=True)
    print("Continuous daily panel shape:", daily_panel.shape)

    # Add time-based features
    daily_panel["date"] = pd.to_datetime(daily_panel["date"])
    daily_panel = daily_panel.sort_values(["barcode", "date"]).reset_index(drop=True)

    # Lag features
    for lag in [1, 7, 14]:
        daily_panel[f"lag_{lag}"] = (
            daily_panel
            .groupby("barcode")["daily_sales"]
            .shift(lag)
            .fillna(0)
        )

    # Rolling mean features
    for win in [7, 14]:
        daily_panel[f"roll_mean_{win}"] = (
            daily_panel
            .groupby("barcode")["daily_sales"]
            .rolling(window=win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Calendar features
    daily_panel["day_of_week"] = daily_panel["date"].dt.dayofweek  # 0=Mon, 6=Sun
    daily_panel["is_weekend"] = daily_panel["day_of_week"].isin([5, 6]).astype(int)
    daily_panel["month"] = daily_panel["date"].dt.month

    # Target: next-day demand (simple version; you can extend to next-7-day later)
    daily_panel["target_next_day"] = (
        daily_panel
        .groupby("barcode")["daily_sales"]
        .shift(-1)
    )

    # Drop rows where target is NaN (last day for each series)
    before_drop = daily_panel.shape[0]
    daily_panel = daily_panel.dropna(subset=["target_next_day"]).reset_index(drop=True)
    after_drop = daily_panel.shape[0]
    print(f"Dropped {before_drop - after_drop} rows with no future target.")

    # Save full panel
    panel_path = OUTPUT_DIR / "forecast_daily_panel.csv"
    daily_panel.to_csv(panel_path, index=False)
    print(f"Saved forecasting panel to: {panel_path}")

    # Simple time-based split (adjust dates if needed)
    cutoff_train = pd.to_datetime("2024-10-31")
    cutoff_val = pd.to_datetime("2024-11-30")

    train_df = daily_panel[daily_panel["date"] <= cutoff_train].copy()
    val_df = daily_panel[
        (daily_panel["date"] > cutoff_train) & (daily_panel["date"] <= cutoff_val)
    ].copy()
    test_df = daily_panel[daily_panel["date"] > cutoff_val].copy()

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # Train
    tmp = OUTPUT_DIR / "forecast_train.tmp.csv"
    train_df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT_DIR / "forecast_train.csv")

    # Val
    tmp = OUTPUT_DIR / "forecast_val.tmp.csv"
    val_df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT_DIR / "forecast_val.csv")

    # Test
    tmp = OUTPUT_DIR / "forecast_test.tmp.csv"
    test_df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT_DIR / "forecast_test.csv")


    print("Saved train/val/test splits for forecasting.")


def build_bundle_cooccurrence(df_all: pd.DataFrame):
    """
    Build a simple co-occurrence table for 'frequently bought together' bundles
    using Invoice + barcode.

    To keep it fast, we limit to TOP_N_BARCODES_FOR_BUNDLES most frequent products.
    """
    print("\n=== Building bundle co-occurrence dataset ===")

    # We use all pharmacies, all drugs (already filtered)
    df_basket = df_all[["Invoice", "barcode"]].dropna().copy()

    # Most frequent barcodes
    barcode_counts = (
        df_basket["barcode"]
        .value_counts()
        .head(TOP_N_BARCODES_FOR_BUNDLES)
    )
    popular_barcodes = set(barcode_counts.index)
    print(f"Top {TOP_N_BARCODES_FOR_BUNDLES} popular barcodes selected for bundles.")

    # Baskets: list of barcodes per Invoice
    baskets_series = (
        df_basket
        .groupby("Invoice")["barcode"]
        .apply(lambda x: list(set(x) & popular_barcodes))
    )

    # Co-occurrence counts
    pair_counts = Counter()
    item_counts = Counter()

    for items in baskets_series:
        if len(items) < 2:
            for item in items:
                item_counts[item] += 1
            continue

        for item in items:
            item_counts[item] += 1

        items_sorted = sorted(items)
        for a, b in combinations(items_sorted, 2):
            pair_counts[(a, b)] += 1

    print(f"Unique items in bundles: {len(item_counts)}")
    print(f"Unique item pairs: {len(pair_counts)}")

    # Build pair statistics table
    rows = []
    total_invoices = len(baskets_series)

    for (a, b), cooc in pair_counts.items():
        support = cooc / total_invoices if total_invoices > 0 else 0.0
        conf_a_to_b = cooc / item_counts[a] if item_counts[a] > 0 else 0.0
        conf_b_to_a = cooc / item_counts[b] if item_counts[b] > 0 else 0.0
        rows.append({
            "item_a": a,
            "item_b": b,
            "cooc_count": cooc,
            "support": support,
            "conf_a_to_b": conf_a_to_b,
            "conf_b_to_a": conf_b_to_a,
        })

    df_pairs = pd.DataFrame(rows)
    df_pairs = df_pairs.sort_values("cooc_count", ascending=False).reset_index(drop=True)
    print("Co-occurrence table shape:", df_pairs.shape)

    cooc_path = OUTPUT_DIR / "bundle_cooccurrence.csv"
    df_pairs.to_csv(cooc_path, index=False)
    print(f"Saved bundle co-occurrence table to: {cooc_path}")


def main():
    print("=== PharmaDemandOps Data Preparation ===")
    print("Project root:", PROJECT_ROOT)
    print("Data root:", DATA_ROOT)
    print("Output dir:", OUTPUT_DIR)

    # 1) Load all transactions
    df_all = load_all_transactions()

    # 2) Clean them
    df_clean = basic_cleaning(df_all)

    # 3) Build forecasting dataset (daily features + target)
    build_forecasting_dataset(df_clean)

    # 4) Build bundle co-occurrence for recommendations
    build_bundle_cooccurrence(df_clean)

    print("\nAll done. You can now use the CSVs in data/processed/ for your ML + API.")


if __name__ == "__main__":
    main()
