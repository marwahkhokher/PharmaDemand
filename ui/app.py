from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="PharmaDemand",
    page_icon="💊",
    layout="wide",  # Centered for better width control
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COOC_PATH = PROJECT_ROOT / "data" / "processed" / "bundle_cooccurrence.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_test.csv"

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
DEEPCHECKS_URL = f"{DEFAULT_API_URL}/reports/deepchecks/deepchecks_report.html?v=latest"

# =========================
# FINAL FIX: Perfect balance of gap and width
# =========================
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* CSS Variables */
:root {
  --bg-main: #F5F7FA;
  --bg-card: #FFFFFF;
  --bg-sidebar: #FAFAFA;
  --text-primary: #1F2937;
  --text-secondary: #6B7280;
  --text-muted: #9CA3AF;
  --border: #E5E7EB;
  --primary: #0F766E;
  --success: #10B981;
  --shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* Base settings */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  overflow-x: hidden !important;
}

/* SIDEBAR - Fixed width, always visible */
[data-testid="stSidebar"] {
  width: 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
}

[data-testid="stSidebar"] > div {
  width: 280px !important;
  padding: 1rem !important;
  background: var(--bg-sidebar) !important;
}

[data-testid="collapsedControl"] {
  display: none !important;
}

/* MAIN CONTENT - Perfect spacing */
.main {
  background: var(--bg-main);
  padding: 0 !important;
  margin: 0 !important;
}

/* Page Headers */
.page-header {
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 0.25rem 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0;
}

/* KPI Cards */
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: var(--shadow);
  height: 110px;
}

.kpi-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.5rem;
}

.kpi-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0;
  line-height: 1.1;
}

.kpi-subtitle {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* Chart Container */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: var(--shadow);
  margin-top: 1.5rem;
}

.chart-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Buttons - Beautiful styling */
.stButton > button {
  font-size: 15px !important;
  padding: 0.75rem 1.25rem !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
  border: 1px solid var(--border) !important;
  width: 100% !important;
  text-align: left !important;
}

.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0F766E, #14B8A6) !important;
  color: white !important;
  border: none !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 6px rgba(15, 118, 110, 0.3) !important;
}

.stButton > button[kind="secondary"] {
  background: white !important;
  color: var(--text-secondary) !important;
}

/* Remove branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
  font-size: 14px;
  padding: 0.75rem 1.5rem;
}

/* ===== FORCE MAIN CONTENT WIDTH (works across Streamlit versions) ===== */
div[data-testid="stMainBlockContainer"],
section.main > div.block-container,
div.block-container {
  max-width: 1500px !important;
  width: 1500px !important;
  margin-left: 0 !important;   /* no centering */
  margin-right: auto !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  box-sizing: border-box !important;
}

/* Responsive: don’t overflow on small screens */
@media (max-width: 920px) {
  div[data-testid="stMainBlockContainer"],
  section.main > div.block-container,
  div.block-container {
    width: 100% !important;
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }
}

</style>
""", unsafe_allow_html=True)

# =========================
# Helper Functions
# =========================

@dataclass
class SystemHealth:
    api_online: bool
    model_ready: bool
    metrics_available: bool
    data_available: bool

def _safe_get(d: Dict[str, Any], path: str, default=None):
    """Safely navigate nested dictionary."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

@st.cache_data(show_spinner=False, ttl=10)
def fetch_json(url: str) -> Tuple[Optional[dict], Optional[str]]:
    """Fetch JSON from URL."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False, ttl=60)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV with caching."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def get_system_health(health: Optional[dict], metrics: Optional[dict], cooc_df: Optional[pd.DataFrame]) -> SystemHealth:
    """Determine system health."""
    return SystemHealth(
        api_online=health is not None,
        model_ready=bool(_safe_get(health or {}, "model_loaded", False)),
        metrics_available=metrics is not None,
        data_available=cooc_df is not None and len(cooc_df) > 0,
    )

def infer_item_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Infer item column names."""
    if "item_a" in df.columns and "item_b" in df.columns:
        return "item_a", "item_b"
    candidates = [c for c in df.columns if "item" in c.lower()]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    return "", ""

def recommend_bundles(df: pd.DataFrame, query_item: str, top_n: int = 5) -> pd.DataFrame:
    """Generate bundle recommendations."""
    colA, colB = infer_item_columns(df)
    if not colA or not colB:
        return pd.DataFrame()

    q = str(query_item).strip()
    mask = (df[colA].astype(str) == q) | (df[colB].astype(str) == q)
    sub = df.loc[mask].copy()
    
    if sub.empty:
        return pd.DataFrame()

    sub["recommended_item"] = sub.apply(
        lambda r: r[colB] if str(r[colA]) == q else r[colA], 
        axis=1
    )

    if "conf_a_to_b" in df.columns and "conf_b_to_a" in df.columns:
        sub["confidence"] = sub.apply(
            lambda r: r["conf_a_to_b"] if str(r[colA]) == q else r["conf_b_to_a"],
            axis=1
        )
    else:
        sub["confidence"] = 1.0

    table = (
        sub.groupby("recommended_item", as_index=False)
        .agg(
            confidence=("confidence", "max"),
            support=("support", "max") if "support" in sub.columns else ("recommended_item", "count"),
            cooc_count=("cooc_count", "max") if "cooc_count" in sub.columns else ("recommended_item", "count"),
        )
        .sort_values("confidence", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    for col in ["confidence", "support"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(4)
    
    if "cooc_count" in table.columns:
        table["cooc_count"] = pd.to_numeric(table["cooc_count"], errors="coerce").astype(int)

    return table

# =========================
# Sidebar
# =========================

def render_sidebar(api_url: str, system_health: SystemHealth):
    """Render sidebar."""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; padding-top: 0.5rem;">
            <div style="font-size: 52px; margin-bottom: 0.75rem;">💊</div>
            <div style="font-size: 20px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.5px;">PharmaDemand</div>
            <div style="font-size: 13px; color: var(--text-muted); margin-top: 0.5rem;">MLOps Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div style="font-size: 13px; font-weight: 600; color: var(--text-secondary); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.5px;">Navigation</div>', unsafe_allow_html=True)
        
        pages = [
            ("🏠", "Dashboard"),
            ("📈", "Forecasting"),
            ("🧺", "Bundles"),
            ("📊", "Analytics"),
            ("🔍", "Monitoring"),
            ("📚", "System"),
        ]
        
        if "page" not in st.session_state:
            st.session_state.page = "Dashboard"
        
        for icon, name in pages:
            is_active = st.session_state.page == name
            
            if st.button(
                f"{icon}  {name}",
                use_container_width=True,
                key=f"nav_{name}",
                type="primary" if is_active else "secondary"
            ):
                st.session_state.page = name
                st.rerun()
            
            st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)

# =========================
# Main App
# =========================

def main():
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    
    api_url = st.session_state.api_url
    
    # Fetch data
    health, _ = fetch_json(f"{api_url}/health")
    metrics, _ = fetch_json(f"{api_url}/metrics/latest")
    cooc_df = load_csv(COOC_PATH)
    train_df = load_csv(TRAIN_PATH)
    test_df = load_csv(TEST_PATH)
    
    system_health = get_system_health(health, metrics, cooc_df)
    
    # Render sidebar
    render_sidebar(api_url, system_health)
    
    # Route pages
    page = st.session_state.page
    
    if page == "Dashboard":
        render_dashboard(system_health, health, metrics, cooc_df, train_df)
    elif page == "Forecasting":
        render_forecasting(api_url, system_health)
    elif page == "Bundles":
        render_bundles(cooc_df, system_health)
    elif page == "Analytics":
        render_analytics(train_df, test_df, cooc_df)
    elif page == "Monitoring":
        render_monitoring(metrics, health, api_url)
    elif page == "System":
        render_system(train_df, test_df, cooc_df, metrics)

# =========================
# Page: Dashboard
# =========================

def render_dashboard(system_health: SystemHealth, health: Optional[dict], metrics: Optional[dict], cooc_df: Optional[pd.DataFrame], train_df: Optional[pd.DataFrame]):
    """Dashboard."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🏠 Dashboard</div>
        <div class="page-subtitle">System overview and key performance indicators</div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_name = _safe_get(health or {}, "model_name", "Unknown")
        if len(model_name) > 20:
            model_name = model_name[:17] + "..."
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ACTIVE MODEL</div>
            <div class="kpi-value" style="font-size: 18px;">{model_name}</div>
            <div class="kpi-subtitle">Production model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rmse_test = _safe_get(metrics or {}, "metrics.test.rmse")
        rmse_display = f"{float(rmse_test):.3f}" if rmse_test is not None else "—"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">TEST RMSE</div>
            <div class="kpi-value">{rmse_display}</div>
            <div class="kpi-subtitle">Model accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        n_products = "—"
        if cooc_df is not None:
            colA, colB = infer_item_columns(cooc_df)
            if colA and colB:
                unique_items = pd.concat([cooc_df[colA], cooc_df[colB]]).nunique()
                n_products = f"{unique_items}"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">PRODUCTS</div>
            <div class="kpi-value">{n_products}</div>
            <div class="kpi-subtitle">In catalog</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart Section
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">📊 Recent Sales Trend</div>
    """, unsafe_allow_html=True)
    
    if train_df is not None and "daily_sales" in train_df.columns:
        recent_sales = train_df["daily_sales"].tail(60)
        dates = pd.date_range(end=datetime.now(), periods=len(recent_sales), freq='D')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=recent_sales,
            mode='lines',
            name='Units Sold',
            line=dict(color='#0F766E', width=2),
            fill='tozeroy',
            fillcolor='rgba(15, 118, 110, 0.1)'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Units Sold",
            template="plotly_white",
            height=280,
            margin=dict(l=40, r=20, t=10, b=40),
            hovermode='x unified',
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Sales data not available")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Page: Forecasting
# =========================

def render_forecasting(api_url: str, system_health: SystemHealth):
    """Forecasting page."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📈 Forecasting</div>
        <div class="page-subtitle">Predict demand for pharmacy items</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not system_health.model_ready:
        st.error("⚠️ Model not ready")
        return
    
    # Explanation section
    with st.expander("ℹ️ How Forecasting Works", expanded=False):
        st.markdown("""
        ### 📊 What the Forecasts Mean
        
        **Next Day Demand:**
        - Predicted number of units that will be sold **tomorrow**
        - Based on recent 14-day sales pattern, seasonality (day of week), and historical trends
        - Helps with: Daily inventory planning, staff scheduling
        
        **Next 7-Day Demand:**
        - **Total units** predicted to be sold over the **next 7 days** (cumulative)
        - Accounts for weekly patterns, seasonal variations, and demand trends
        - Helps with: Weekly ordering, stock replenishment planning
        
        ### 🤖 Model Input Features
        - **Recent Sales History:** Last 14 days of daily sales (lag features)
        - **Rolling Averages:** 7-day and 14-day moving averages
        - **Seasonality:** Day of week (weekday vs weekend), month
        - **Product ID & Pharmacy:** Specific demand patterns per item and location
        
        ### 📈 How It's Calculated
        1. Model analyzes your recent 14-day sales pattern
        2. Identifies trends (increasing/decreasing demand)
        3. Applies seasonality adjustments (e.g., weekends may have different patterns)
        4. Generates prediction using trained regression model (XGBoost/Random Forest)
        
        ### 💡 Example Interpretation
        - **Next Day = 5.2 units** → Expect to sell ~5 units tomorrow
        - **Next 7-Day = 28.4 units** → Expect to sell ~28 units total over the next week
        - If next day = 5 and next 7-day = 28, average daily = 28/7 = 4 units/day
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Parameters")
        
        with st.form("forecast_form"):
            pharmacy_id = st.text_input("Pharmacy ID", value="Ph01_Z01_C01", help="Unique pharmacy location identifier")
            barcode = st.text_input("Barcode", value="6251070000000", help="Product barcode/SKU")
            forecast_date = st.date_input("Date", value=datetime.now(), help="Reference date for prediction")
            recent_sales_str = st.text_area(
                "Recent Sales (14 values)",
                value="0,1,0,2,1,0,3,2,1,0,1,2,0,1",
                height=80,
                help="Last 14 days of daily sales, comma-separated (most recent last)"
            )
            submit = st.form_submit_button("Generate Forecast", use_container_width=True)
        
    with col2:
        st.markdown("### Results")
        
        if submit:
            try:
                recent_sales = [float(x.strip()) for x in recent_sales_str.split(",") if x.strip()]
                
                if len(recent_sales) != 14:
                    st.error("❌ Provide exactly 14 values")
                else:
                    payload = {
                        "pharmacy_id": pharmacy_id,
                        "barcode": barcode,
                        "date": forecast_date.isoformat(),
                        "recent_daily_sales": recent_sales,
                    }
                    
                    response = requests.post(
                        f"{api_url}/predict/demand-next7",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        next_day = result.get("next_day_demand")
                        next_7day = result.get("next_7day_demand")
                        
                        st.success("✅ Forecast generated successfully!")
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric(
                            "Next Day", 
                            f"{next_day:.2f}" if isinstance(next_day, (int, float)) else str(next_day),
                            help="Units expected to sell tomorrow"
                        )
                        col_b.metric(
                            "Next 7 Days", 
                            f"{next_7day:.2f}" if isinstance(next_7day, (int, float)) else str(next_7day),
                            help="Total units expected over next week"
                        )
                        
                        # Show interpretation
                        if isinstance(next_7day, (int, float)) and isinstance(next_day, (int, float)):
                            avg_daily = next_7day / 7.0
                            st.info(f"💡 **Interpretation:** Average daily demand over next week: ~{avg_daily:.1f} units/day")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.info("👈 Fill in the form and click **Generate Forecast** to get predictions")
            st.caption("The model will predict demand based on your recent sales pattern and historical trends.")

# =========================
# Page: Bundles
# =========================

def render_bundles(cooc_df: Optional[pd.DataFrame], system_health: SystemHealth):
    """Bundle recommendations."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🧺 Bundles</div>
        <div class="page-subtitle">Product bundle recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    
    if cooc_df is None:
        st.error("Bundle data not available")
        return
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Select Product")
        
        colA, colB = infer_item_columns(cooc_df)
        items = sorted(set(cooc_df[colA].astype(str).tolist() + cooc_df[colB].astype(str).tolist()))
        default_item = "1063" if "1063" in items else items[0]
        
        selected_item = st.selectbox("Product", options=items, index=items.index(default_item))
        top_n = st.slider("Top N", 3, 10, 5)
        find = st.button("Find Bundles", use_container_width=True, type="primary")
    
    with col2:
        st.markdown("### Recommendations")
        
        if find:
            recommendations = recommend_bundles(cooc_df, selected_item, top_n=top_n)
            
            if recommendations.empty:
                st.warning("No recommendations found")
            else:
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
        else:
            st.info("Select product and click Find Bundles")

# =========================
# Page: Analytics - COMPLETE VERSION
# =========================

def render_analytics(train_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame], cooc_df: Optional[pd.DataFrame]):
    """Analytics page with ALL tabs filled."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Analytics</div>
        <div class="page-subtitle">Data insights and patterns</div>
    </div>
    """, unsafe_allow_html=True)
    
    if train_df is None:
        st.error("Training data not available")
        return
    
    tabs = st.tabs(["📈 Sales", "🧺 Bundles", "📅 Patterns"])
    
    # TAB 1: Sales Analysis
    with tabs[0]:
        if "daily_sales" in train_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Distribution")
                fig = px.histogram(train_df, x="daily_sales", nbins=50, color_discrete_sequence=["#0F766E"])
                fig.update_layout(height=300, showlegend=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("### Statistics")
                stats = train_df["daily_sales"].describe()
                st.write(f"**Mean:** {stats['mean']:.2f}")
                st.write(f"**Median:** {stats['50%']:.2f}")
                st.write(f"**Std Dev:** {stats['std']:.2f}")
                st.write(f"**Max:** {stats['max']:.0f}")
                st.write(f"**Min:** {stats['min']:.0f}")
    
    # TAB 2: Bundle Analysis
    with tabs[1]:
        if cooc_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Confidence Distribution")
                if "conf_a_to_b" in cooc_df.columns:
                    fig = px.histogram(
                        cooc_df, 
                        x="conf_a_to_b", 
                        nbins=30,
                        title="Bundle Confidence Scores",
                        labels={"conf_a_to_b": "Confidence"},
                        color_discrete_sequence=["#0D9488"]
                    )
                    fig.update_layout(height=300, showlegend=False, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Confidence data not available")
            
            with col2:
                st.markdown("### Top Item Pairs")
                if "cooc_count" in cooc_df.columns:
                    top_pairs = cooc_df.nlargest(10, "cooc_count")
                    colA, colB = infer_item_columns(cooc_df)
                    top_pairs["pair"] = top_pairs[colA].astype(str) + " + " + top_pairs[colB].astype(str)
                    
                    fig = px.bar(
                        top_pairs,
                        y="pair",
                        x="cooc_count",
                        orientation='h',
                        title="Most Frequent Pairs",
                        color="cooc_count",
                        color_continuous_scale="Teal"
                    )
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        template="plotly_white",
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.info("Co-occurrence data not available")
            
            # Bundle statistics
            st.markdown("### Bundle Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pairs", f"{len(cooc_df):,}")
            with col2:
                if "conf_a_to_b" in cooc_df.columns:
                    st.metric("Avg Confidence", f"{cooc_df['conf_a_to_b'].mean():.3f}")
            with col3:
                if "support" in cooc_df.columns:
                    st.metric("Avg Support", f"{cooc_df['support'].mean():.4f}")
        else:
            st.info("Bundle data not available for analysis")
    
    # TAB 3: Temporal Patterns
    with tabs[2]:
        st.markdown("### Temporal Analysis")
        
        if "daily_sales" in train_df.columns:
            # Sales trend over time
            st.markdown("#### Sales Over Time")
            recent_data = train_df.tail(90)
            if len(recent_data) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=recent_data["daily_sales"],
                    mode='lines',
                    name='Daily Sales',
                    line=dict(color='#0F766E', width=2)
                ))
                fig.update_layout(
                    xaxis_title="Days",
                    yaxis_title="Sales",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Volatility analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Volatility")
                rolling_std = train_df["daily_sales"].rolling(window=7).std()
                fig = px.line(
                    y=rolling_std.dropna(),
                    title="7-Day Rolling Standard Deviation",
                    labels={"y": "Std Dev", "index": "Days"}
                )
                fig.update_layout(height=250, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.markdown("#### Moving Average")
                rolling_mean = train_df["daily_sales"].rolling(window=7).mean()
                fig = px.line(
                    y=rolling_mean.dropna(),
                    title="7-Day Moving Average",
                    labels={"y": "Avg Sales", "index": "Days"}
                )
                fig.update_layout(height=250, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Sales data required for temporal analysis")

# =========================
# Page: Monitoring
# =========================

def render_monitoring(metrics: Optional[dict], health: Optional[dict], api_url: str):
    """Monitoring page."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🔍 Monitoring</div>
        <div class="page-subtitle">Model performance metrics</div>
    </div>
    """, unsafe_allow_html=True)
    
    if metrics is None:
        st.error("Metrics not available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    rmse_test = _safe_get(metrics, "metrics.test.rmse")
    mae_test = _safe_get(metrics, "metrics.test.mae")
    r2_test = _safe_get(metrics, "metrics.test.r2")
    
    with col1:
        st.metric("RMSE (Test)", f"{float(rmse_test):.4f}" if rmse_test else "—")
    with col2:
        st.metric("MAE (Test)", f"{float(mae_test):.4f}" if mae_test else "—")
    with col3:
        st.metric("R² Score", f"{float(r2_test):.4f}" if r2_test else "—")
    
    st.markdown("---")
    st.markdown("### Complete Metrics")
    st.json(metrics)
    
    st.link_button("View DeepChecks Report", DEEPCHECKS_URL, use_container_width=True)

# =========================
# Page: System - WITH DEEPCHECKS BUTTON
# =========================

def render_system(train_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame], cooc_df: Optional[pd.DataFrame], metrics: Optional[dict]):
    """System documentation."""
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📚 System</div>
        <div class="page-subtitle">Complete system documentation</div>
    </div>
    """, unsafe_allow_html=True)
    
    # DEEPCHECKS BUTTON AT TOP
    st.markdown("### 🔍 Validation Reports")
    st.link_button(
        "📊 Open DeepChecks Validation Report",
        DEEPCHECKS_URL,
        use_container_width=False,
        type="primary"
    )
    st.caption("Comprehensive data quality and model validation report")
    
    st.markdown("---")
    
    tabs = st.tabs(["Overview", "Data", "Models", "Evaluation", "MLOps"])
    
    with tabs[0]:
        st.markdown("### Project Summary")
        st.write("""
        **PharmaDemand** is a full-stack MLOps system for pharmaceutical demand forecasting 
        and product bundle recommendations.
        """)
        
        st.markdown("### Architecture")
        st.write("- **Data Layer:** Invoice and sales data processing")
        st.write("- **ML Layer:** Regression models for demand, association rules for bundles")
        st.write("- **API Layer:** FastAPI inference server")
        st.write("- **UI Layer:** Streamlit dashboard")
    
    with tabs[1]:
        st.markdown("### Dataset Statistics")
        
        stats = []
        if train_df is not None:
            stats.append(["Training Set", len(train_df), train_df.shape[1]])
        if test_df is not None:
            stats.append(["Test Set", len(test_df), test_df.shape[1]])
        if cooc_df is not None:
            stats.append(["Bundles", len(cooc_df), cooc_df.shape[1]])
        
        if stats:
            st.dataframe(pd.DataFrame(stats, columns=["Dataset", "Rows", "Columns"]), use_container_width=True, hide_index=True)
    
    with tabs[2]:
        st.markdown("### Demand Forecasting")
        st.write("- **Type:** Regression")
        st.write("- **Features:** Lag features, rolling averages, seasonality")
        st.write("- **Output:** Next-day and 7-day demand")
        
        st.markdown("### Bundle Recommendations")
        st.write("- **Method:** Association rule mining")
        st.write("- **Metrics:** Confidence, support, co-occurrence")
    
    with tabs[3]:
        if metrics:
            st.markdown("### Performance Metrics")
            st.json(metrics)
        else:
            st.info("Metrics not available")
    
    with tabs[4]:
        st.markdown("### Technology Stack")
        st.write("- **ML:** scikit-learn, pandas")
        st.write("- **API:** FastAPI, Uvicorn")
        st.write("- **UI:** Streamlit, Plotly")
        st.write("- **Validation:** DeepChecks")
        st.write("- **Orchestration:** Prefect")
        st.write("- **Containerization:** Docker")

# =========================
# Run
# =========================

if __name__ == "__main__":
    main()