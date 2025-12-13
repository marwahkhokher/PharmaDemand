from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "transactions_master.csv"

def main():
    df = pd.read_csv(MASTER, low_memory=False)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Null rates (top 10)
    nulls = (df.isna().mean().sort_values(ascending=False).head(10) * 100)
    print("\nTop null columns (%):\n", nulls)

    # Timestamp parse rate
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    print("\nTimestamp valid %:", round(ts.notna().mean() * 100, 2))

    # Barcode sanity
    print("Empty barcode %:", round((df["barcode"].astype(str).str.strip() == "").mean() * 100, 2))

    # Sales_pack distribution
    sp = pd.to_numeric(df["Sales_pack"], errors="coerce")
    print("\nSales_pack stats:")
    print(sp.describe())

    print("\nSales_pack == 0 %:", round((sp.fillna(0) == 0).mean() * 100, 2))

    # Duplicate rows
    print("\nExact duplicate rows:", int(df.duplicated().sum()))

    # Type values (to detect returns/cancel)
    if "type" in df.columns:
        print("\nTop 'type' values:")
        print(df["type"].astype(str).value_counts().head(20))

if __name__ == "__main__":
    main()
