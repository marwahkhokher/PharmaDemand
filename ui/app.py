import os
import time
import json
from pathlib import Path
from datetime import date

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers (API)
# -----------------------------
def api_get_json(path: str, timeout: int = 10):
    r = requests.get(f"{API_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post_json(path: str, payload: dict, timeout: int = 20):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# -----------------------------
# Helpers (Artifacts)
# -----------------------------
def file_status(path: Path):
    return {
        "exists": path.exists(),
        "path": str(path),
        "last_modified": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)) if path.exists() else None),
        "size_mb": (round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None),
    }

def human_badge(ok: bool, label: str):
    return f"{'✅' if ok else '❌'} {label}"

def pick_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------
# Bundle Co-occurrence Loader (Flexible)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_cooc_csv():
    """
    Tries to find a co-occurrence / bundle CSV in common locations.
    Returns (df, chosen_path) or (None, None) if not found.

    IMPORTANT:
    - This function is designed to be flexible with different column names.
    - If your CSV uses different column names, adjust infer logic below.
    """
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "bundle_cooccurrence.csv",
        PROJECT_ROOT / "data" / "processed" / "cooccurrence.csv",
        PROJECT_ROOT / "data" / "processed" / "cooc.csv",
        PROJECT_ROOT / "reports" / "bundle_cooccurrence.csv",
        PROJECT_ROOT / "reports" / "cooccurrence.csv",
        PROJECT_ROOT / "bundle_cooccurrence.csv",
        PROJECT_ROOT / "cooccurrence.csv",
    ]
    chosen = pick_first_existing(candidates)
    if not chosen:
        return None, None

    df = pd.read_csv(chosen)

    # Clean obvious whitespace
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    return df, chosen


def infer_item_columns(df: pd.DataFrame):
    """
    Try to infer "item A" and "item B" columns.
    Common patterns:
      - antecedent / consequent
      - item_a / item_b
      - product_a / product_b
      - barcode_a / barcode_b
      - source / target
    """
    cols = set(df.columns)

    pairs = [
        ("antecedent", "consequent"),
        ("item_a", "item_b"),
        ("product_a", "product_b"),
        ("barcode_a", "barcode_b"),
        ("source", "target"),
        ("from", "to"),
        ("lhs", "rhs"),
    ]
    for a, b in pairs:
        if a in cols and b in cols:
            return a, b

    # fallback heuristic: pick first two object columns
    obj_cols = list(df.select_dtypes(include="object").columns)
    if len(obj_cols) >= 2:
        return obj_cols[0], obj_cols[1]

    return None, None


def infer_score_column(df: pd.DataFrame):
    # Prefer confidence columns first, then support/count
    preferred = ["conf_a_to_b", "conf_b_to_a", "support", "cooc_count"]
    cols = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in cols:
            return cols[p.lower()]
    return None



def recommend_bundles_from_df(df: pd.DataFrame, query_item: str, top_n: int = 5):
    """
    Returns a dataframe of recommendations: recommended_item + strength
    Works whether query_item appears in colA or colB.
    """
    colA, colB = infer_item_columns(df)
    if not colA or not colB:
        return None, "Could not infer item columns from co-occurrence CSV."

    score_col = infer_score_column(df)

    q = str(query_item).strip()

    # match either side
    mask = (df[colA].astype(str) == q) | (df[colB].astype(str) == q)
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["recommended_item", "strength"]), None

    # Determine "other" item
    sub["recommended_item"] = sub.apply(lambda r: r[colB] if str(r[colA]) == q else r[colA], axis=1)

    # Direction-aware scoring:
    # If query matches item_a -> use conf_a_to_b
    # If query matches item_b -> use conf_b_to_a
    if "conf_a_to_b" in df.columns and "conf_b_to_a" in df.columns:
        def _score_row(r):
            if str(r[colA]) == q:
                return r["conf_a_to_b"]
            else:
                return r["conf_b_to_a"]
        sub["strength"] = pd.to_numeric(sub.apply(_score_row, axis=1), errors="coerce")
    elif score_col:
        sub["strength"] = pd.to_numeric(sub[score_col], errors="coerce")
    else:
        sub["strength"] = 1.0


    # Aggregate duplicates
    out = (
        sub.groupby("recommended_item", as_index=False)["strength"]
        .sum()
        .sort_values("strength", ascending=False)
        .head(top_n)
    )

    return out, None


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="PharmaDemandOps", layout="wide")

st.title("PharmaDemandOps")
st.caption("Forecast demand for the next 7 days and recommend frequently bought-together bundles.")

# -----------------------------
# Fetch system info
# -----------------------------
health = safe_try(lambda: api_get_json("/health"), default=None)
metrics = safe_try(lambda: api_get_json("/metrics/latest"), default=None)

# Report URL (cache-busted)
deepchecks_url = f"{API_URL}/reports/deepchecks/deepchecks_report.html?v={int(time.time())}"

# Artifact paths (local)
model_path = PROJECT_ROOT / "models" / "demand_regressor.pkl"
metrics_path = PROJECT_ROOT / "reports" / "demand_model_metrics.json"
report_path = PROJECT_ROOT / "reports" / "deepchecks" / "deepchecks_report.html"

# Load cooc file (local)
cooc_df, cooc_path = load_cooc_csv()


# -----------------------------
# Top status badges
# -----------------------------
api_online = health is not None
model_ready = bool(health and health.get("model_loaded") is True)
metrics_ready = bool(metrics is not None)
deepchecks_ready = report_path.exists()
bundles_ready = (cooc_df is not None)

badge_cols = st.columns(5)
badge_cols[0].markdown(human_badge(api_online, "API Online"))
badge_cols[1].markdown(human_badge(model_ready, "Model Ready"))
badge_cols[2].markdown(human_badge(metrics_ready, "Metrics Available"))
badge_cols[3].markdown(human_badge(deepchecks_ready, "DeepChecks Report"))
badge_cols[4].markdown(human_badge(bundles_ready, "Bundles Data"))


st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab_forecast, tab_bundles, tab_system = st.tabs(["📈 Forecast Demand", "🧺 Bundle Recommendations", "🧪 System & Validation"])


# ============================================================
# TAB 1: Forecast Demand
# ============================================================
with tab_forecast:
    st.subheader("Predict demand for the next 7 days")
    st.write("Enter the last 14 days sales for an item and get an estimated total demand for the next week.")

    left, right = st.columns([1.1, 1.0])

    with left:
        pharmacy_id = st.text_input("Pharmacy Branch ID", value="Ph01_Z01_C01")
        barcode = st.text_input("Product Barcode", value="6251070000000")
        d = st.date_input("Start Date", value=date(2024, 12, 31))

        recent = st.text_area(
            "Last 14 Days Sales (comma-separated)",
            value="0,1,0,2,1,0,3,2,1,0,1,2,0,1",
            help="Example: 14 numbers separated by commas."
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use sample data"):
                st.session_state["recent_sales"] = "0,1,0,2,1,0,3,2,1,0,1,2,0,1"
                st.rerun()
        with c2:
            st.write("")

        if "recent_sales" in st.session_state:
            recent = st.session_state["recent_sales"]

        predict_clicked = st.button("Predict next 7 days ✅", type="primary")

    with right:
        st.subheader("Result")

        if predict_clicked:
            try:
                recent_list = [float(x.strip()) for x in recent.split(",") if x.strip() != ""]
                if len(recent_list) != 14:
                    st.error(f"Please enter exactly 14 values. You provided {len(recent_list)}.")
                else:
                    payload = {
                        "pharmacy_id": pharmacy_id.strip(),
                        "barcode": barcode.strip(),
                        "date": str(d),
                        "recent_daily_sales": recent_list
                    }
                    result = api_post_json("/predict/demand-next7", payload, timeout=30)

                    # Extract values safely
                    next7 = result.get("next_7day_demand", None)
                    next1 = result.get("next_day_demand", None)
                    model_name = result.get("model_name", "unknown")

                    if next7 is None:
                        st.warning("Prediction returned, but next_7day_demand was not found in response.")
                    else:
                        st.success("Prediction successful ✅")
                        st.metric("Predicted total demand (next 7 days)", f"{float(next7):.2f} units")

                        if next1 is not None:
                            st.metric("Predicted next-day demand", f"{float(next1):.2f} units")

                        # Daily split estimate (honest: not true daily forecast unless API returns daily list)
                        avg_per_day = float(next7) / 7.0
                        st.metric("Average per day (simple estimate)", f"{avg_per_day:.2f} units/day")

                        st.caption(f"Model used: **{model_name}**")

                        # Show a simple bar chart (even split estimate)
                        daily_est = [avg_per_day] * 7
                        st.write("Estimated daily split (even distribution for visualization):")
                        st.bar_chart(pd.DataFrame({"day": list(range(1, 8)), "units": daily_est}).set_index("day"))

                    with st.expander("Advanced details (raw API response)"):
                        st.json(result)

            except requests.HTTPError as e:
                st.error(f"API error: {e}")
                # Show response if available
                try:
                    st.code(e.response.text)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("Fill in the inputs and click **Predict next 7 days** to see the forecast.")


# ============================================================
# TAB 2: Bundle Recommendations
# ============================================================
with tab_bundles:
    st.subheader("Frequently Bought Together (Bundle Recommendations)")
    st.write("Enter a product barcode and get items that are commonly purchased with it.")

    left, right = st.columns([1.1, 1.0])

    with left:
        sample_ids = sorted(set(cooc_df["item_a"].astype(str)).union(set(cooc_df["item_b"].astype(str))))
        bundle_barcode = st.selectbox("Select an Item ID", sample_ids[:500], index=0)

        bundle_barcode = st.text_input("Item ID for bundles (from transactions)", value="1063", key="bundle_barcode")
        top_n = st.slider("How many recommendations?", min_value=3, max_value=10, value=5)

        st.write("You entered:", bundle_barcode)
        st.write("Example item_a/item_b from file:", cooc_df[["item_a","item_b"]].head(5))

        recommend_clicked = st.button("Show bundle recommendations 🧺", type="primary")

        st.caption("Note: This panel reads from your co-occurrence CSV (local) unless you later add a bundles API.")

    with right:
        st.subheader("Recommended bundle items")

        if cooc_df is None:
            st.error("Bundle data not found.")
            st.write("Expected a file like:")
            st.code(str(PROJECT_ROOT / "data" / "processed" / "bundle_cooccurrence.csv"))
            st.write("If your co-occurrence file exists with a different name/location, tell me its path and I’ll plug it in.")
        else:
            st.caption(f"Using co-occurrence file: {cooc_path}")

            if recommend_clicked:
                recs, err = recommend_bundles_from_df(cooc_df, bundle_barcode.strip(), top_n=top_n)

                if err:
                    st.error(err)
                    with st.expander("Debug: co-occurrence columns"):
                        st.write(list(cooc_df.columns))
                else:
                    if recs.empty:
                        st.warning("No bundle recommendations found for that barcode in the co-occurrence dataset.")
                    else:
                        st.success("Recommendations ready ✅")

                        # Pretty table
                        recs_display = recs.copy()
                        recs_display["strength"] = recs_display["strength"].astype(float).round(4)
                        # st.dataframe(recs_display, width="stretch", hide_index=True)
                        # Join back extra info (support, cooc_count) for display
                        colA, colB = infer_item_columns(cooc_df)
                        sub = cooc_df[(cooc_df[colA] == bundle_barcode) | (cooc_df[colB] == bundle_barcode)].copy()
                        sub["recommended_item"] = sub.apply(lambda r: r[colB] if str(r[colA]) == bundle_barcode else r[colA], axis=1)

                        # Compute directional confidence as strength
                        sub["confidence"] = sub.apply(lambda r: r["conf_a_to_b"] if str(r[colA]) == bundle_barcode else r["conf_b_to_a"], axis=1)

                        table = (
                            sub.groupby("recommended_item", as_index=False)
                            .agg(confidence=("confidence", "max"), support=("support", "max"), cooc_count=("cooc_count", "max"))
                            .sort_values("confidence", ascending=False)
                            .head(top_n)
                        )

                        table["confidence"] = table["confidence"].round(4)
                        table["support"] = table["support"].round(6)

                        st.dataframe(table, width="stretch", hide_index=True)


                        # Small “bundle idea” demo
                        st.write("Bundle idea (demo):")
                        top_items = recs_display["recommended_item"].head(3).tolist()
                        st.markdown(
                            f"- **Suggested offer:** Buy **{bundle_barcode}** + **{', '.join(map(str, top_items))}** → offer a small discount"
                        )

                        with st.expander("Advanced details (how this is computed)"):
                            st.write("We filter co-occurrence rows where the selected barcode appears, then rank the paired items by a strength column (confidence/lift/support/etc.) if present, otherwise by frequency.")

            else:
                st.info("Enter a product barcode and click **Show bundle recommendations**.")


# ============================================================
# TAB 3: System & Validation
# ============================================================
with tab_system:
    st.subheader("System & Validation")
    st.write("This section is mainly for technical verification and professors: artifacts, metrics, and validation reports.")

    # Section: System Health
    st.markdown("### ✅ System Health")
    if health is None:
        st.error("Could not reach API /health. Is FastAPI running?")
    else:
        # Human-readable checklist
        st.write(human_badge(True, "API reachable"))
        st.write(human_badge(bool(health.get("model_loaded") is True), "Model loaded"))
        st.write(human_badge(bool(health.get("cooc_file_present") is True), "Co-occurrence file present (backend check)"))
        st.write(human_badge(bool(health.get("metrics_file_present") is True), "Metrics file present (backend check)"))

        with st.expander("Advanced details (raw /health JSON)"):
            st.json(health)

    st.divider()

    # Section: Metrics
    st.markdown("### 📊 Model Performance Metrics")
    if metrics is None:
        st.warning("Metrics not available via /metrics/latest.")
    else:
        # Try to extract common structure:
        # metrics might look like:
        # { "best_model": "...", "feature_columns": [...], "metrics": { "val": {...}, "test": {...} } }
        best_model = metrics.get("best_model", metrics.get("model", "unknown"))
        m = metrics.get("metrics", {})

        val = m.get("val", {}) if isinstance(m, dict) else {}
        test = m.get("test", {}) if isinstance(m, dict) else {}

        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", str(best_model))

        # Use safe gets
        val_rmse = val.get("rmse", None)
        val_mae = val.get("mae", None)
        test_rmse = test.get("rmse", None)
        test_mae = test.get("mae", None)

        c2.metric("Validation RMSE", f"{val_rmse:.4f}" if isinstance(val_rmse, (int, float)) else "—")
        c2.metric("Validation MAE", f"{val_mae:.4f}" if isinstance(val_mae, (int, float)) else "—")
        c3.metric("Test RMSE", f"{test_rmse:.4f}" if isinstance(test_rmse, (int, float)) else "—")
        c3.metric("Test MAE", f"{test_mae:.4f}" if isinstance(test_mae, (int, float)) else "—")

        with st.expander("Advanced details (raw /metrics/latest JSON)"):
            st.json(metrics)

    st.divider()

    # Section: DeepChecks
    st.markdown("### 🧪 Data & Model Validation (DeepChecks)")
    if report_path.exists():
        st.success("DeepChecks report found ✅")
    else:
        st.warning("DeepChecks report not found locally. Run: python ml/deepchecks_gate.py")

    st.link_button("Open DeepChecks Report", deepchecks_url)
    st.code(deepchecks_url)

    st.divider()

    # Section: Artifacts
    st.markdown("### 📦 Artifacts")
    artifacts = [
        ("Model file", file_status(model_path)),
        ("Metrics file", file_status(metrics_path)),
        ("DeepChecks HTML report", file_status(report_path)),
        ("Co-occurrence CSV", file_status(cooc_path) if cooc_path else {"exists": False, "path": None, "last_modified": None, "size_mb": None}),
    ]

    art_rows = []
    for name, info in artifacts:
        art_rows.append({
            "Artifact": name,
            "Status": "✅" if info["exists"] else "❌",
            "Path": info["path"],
            "Last Modified": info["last_modified"],
            "Size (MB)": info["size_mb"],
        })

    st.dataframe(pd.DataFrame(art_rows), width="stretch", hide_index=True)
