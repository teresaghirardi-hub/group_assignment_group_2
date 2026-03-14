"""
pages/go_live.py — Go Live Page (Part 2.2.2)
=============================================
Fetches live data from SimFin, applies the same ETL transformations
used during training, loads the saved model, and shows a prediction.

ETL functions are defined here directly (not imported from a separate etl.py)
so that Part 1 remains fully in the notebook as required.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

# Make pysimfin.py importable (it lives one folder above pages/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from pysimfin import PySimFin, SimFinAPIError, SimFinNotFoundError, SimFinRateLimitError

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Go Live | AlgoTrader", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(255,255,255,0.06); }
.sig-rise { background: linear-gradient(135deg,rgba(16,185,129,0.12),rgba(16,185,129,0.04)); border: 2px solid #10b981; border-radius: 14px; padding: 28px; text-align: center; }
.sig-fall { background: linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.04)); border: 2px solid #ef4444; border-radius: 14px; padding: 28px; text-align: center; }
.sig-icon { font-size: 3rem; margin-bottom: 6px; }
.sig-label { font-size: 1.8rem; font-weight: 700; }
.sig-sub { color: #94a3b8; font-size: 0.9rem; margin-top: 6px; }
.stat { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 16px; text-align: center; }
.stat-val { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #63b3ed; }
.stat-lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
.sec { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 20px 0 10px 0; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.info { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2); border-radius: 8px; padding: 10px 14px; font-size: 0.87rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 AlgoTrader")
    st.markdown("---")
    st.page_link("app.py",               label="🏠 Home")
    st.page_link("pages/go_live.py",     label="⚡ Go Live")
    st.page_link("pages/backtesting.py", label="🔁 Backtesting")
    st.markdown("---")
    st.markdown("**Settings**")
    api_key      = st.text_input("SimFin API Key", type="password",
                                 value=os.environ.get("SIMFIN_API_KEY", ""))
    TICKERS      = ["AMZN", "AAPL", "MSFT", "GOOG", "TSLA"]
    ticker       = st.selectbox("Select Ticker", TICKERS)
    days_history = st.slider("Days of history to show", 30, 365, 90)
    st.markdown("---")
    run_btn = st.button("⚡ Get Prediction", use_container_width=True, type="primary")

# ── ETL functions (same as notebook — must be identical to training) ───────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8 technical features. Must match etl_nuria.ipynb exactly."""
    df = df.copy()

    # normalise column names — SimFin API returns lowercase
    col = {c.lower(): c for c in df.columns}
    close  = df[col.get("close",  "close")]
    high   = df[col.get("high",   "high")]
    low    = df[col.get("low",    "low")]
    volume = df[col.get("volume", "volume")]

    df["Returns"]       = np.log(close / close.shift(1))
    df["SMA_5"]         = close.rolling(window=5).mean()
    df["SMA_20"]        = close.rolling(window=20).mean()
    df["Volatility_5"]  = df["Returns"].rolling(window=5).std()
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()
    df["Volume_Change"] = volume.pct_change()

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    rs       = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI_14"]      = 100 - (100 / (1 + rs))
    df["Price_Range"] = (high - low) / close

    feature_cols = ["Returns","SMA_5","SMA_20","Volatility_5",
                    "Volatility_20","Volume_Change","RSI_14","Price_Range"]
    return df.dropna(subset=feature_cols).reset_index(drop=True)


def prepare_for_prediction(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Apply ETL to live data and return only the columns the model needs."""
    df = add_technical_features(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features after ETL: {missing}")
    return df[feature_cols].dropna()

# ── Model loader ───────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

@st.cache_resource
def load_model(ticker: str):
    """Load trained pipeline and feature list. Cached so it only loads once."""
    model_path    = MODELS_DIR / f"model_{ticker}.joblib"
    features_path = MODELS_DIR / f"features_{ticker}.txt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun etl_nuria.ipynb first."
        )
    pipeline = joblib.load(model_path)
    with open(features_path) as f:
        features = [line.strip() for line in f if line.strip()]
    return pipeline, features

# ── Chart builder ──────────────────────────────────────────────────────────────
def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    col   = {c.lower(): c for c in df.columns}
    fig   = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df[col["date"]], open=df[col["open"]], high=df[col["high"]],
        low=df[col["low"]], close=df[col["close"]], name=ticker,
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        increasing_fillcolor="rgba(16,185,129,0.25)",
        decreasing_fillcolor="rgba(239,68,68,0.25)",
    ))
    sma20 = df[col["close"]].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df[col["date"]], y=sma20, mode="lines",
        name="SMA 20", line=dict(color="#63b3ed", width=1.5, dash="dot")))
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        title=dict(text=f"{ticker} — Price History", font=dict(color="#e2e8f0", size=15)),
        xaxis=dict(gridcolor="#1e293b", rangeslider_visible=False),
        yaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Go Live")
st.markdown('<p style="color:#64748b;margin-top:-8px;">Real-time predictions powered by live SimFin data.</p>', unsafe_allow_html=True)

if not run_btn:
    st.markdown('<div class="info">👈 Enter your SimFin API key, select a ticker, and click <strong>Get Prediction</strong>.</div>', unsafe_allow_html=True)
    st.stop()

if not api_key:
    st.error("Please enter your SimFin API key in the sidebar.")
    st.stop()

# Load model
try:
    pipeline, model_features = load_model(ticker)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Fetch data
end_date   = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=days_history + 60)).strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker} from SimFin…"):
    try:
        client = PySimFin(api_key=api_key)
        df_raw = client.get_share_prices(ticker, start=start_date, end=end_date)
        df_raw.columns = df_raw.columns.str.lower() 
    except SimFinRateLimitError:
        st.error("Rate limit hit. Wait a moment and retry.")
        st.stop()
    except SimFinNotFoundError:
        st.error(f"Ticker '{ticker}' not found in SimFin.")
        st.stop()
    except SimFinAPIError as e:
        st.error(f"SimFin API error: {e}")
        st.stop()

if df_raw.empty:
    st.error("No data returned. Check ticker and date range.")
    st.stop()

# Apply ETL and predict
try:
    X_live = prepare_for_prediction(df_raw, model_features)
except ValueError as e:
    st.error(f"ETL error: {e}")
    st.stop()

if X_live.empty:
    st.error("Not enough data after ETL. Try a wider date range.")
    st.stop()

X_latest    = X_live.iloc[[-1]]
prediction  = int(pipeline.predict(X_latest)[0])
probability = float(pipeline.predict_proba(X_latest)[0][prediction])

# Price stats
col_map    = {c.lower(): c for c in df_raw.columns}
close_col  = col_map.get("close",  "close")
open_col   = col_map.get("open",   "open")
volume_col = col_map.get("volume", "volume")
date_col   = col_map.get("date",   "date")

latest     = df_raw.iloc[-1]
prev       = df_raw.iloc[-2]
last_close = float(latest[close_col])
day_change = last_close - float(prev[close_col])
day_pct    = (day_change / float(prev[close_col])) * 100

# Layout
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    df_chart       = df_raw.copy()
    df_chart[date_col] = pd.to_datetime(df_chart[date_col])
    df_chart       = df_chart.tail(days_history)
    st.plotly_chart(candlestick_chart(df_chart, ticker), use_container_width=True)

    st.markdown('<p class="sec">Latest Data</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    arrow = "▲" if day_change >= 0 else "▼"
    c_col = "#10b981" if day_change >= 0 else "#ef4444"
    for c, v, l in zip([s1,s2,s3,s4],
        [f"${last_close:.2f}",
         f'<span style="color:{c_col}">{arrow} {day_pct:.2f}%</span>',
         f"${float(latest[open_col]):.2f}" if open_col in df_raw.columns else "—",
         f"{int(float(latest[volume_col])):,}" if volume_col in df_raw.columns else "—"],
        ["Last Close","Day Change","Open","Volume"]):
        c.markdown(f'<div class="stat"><div class="stat-val">{v}</div><div class="stat-lbl">{l}</div></div>', unsafe_allow_html=True)

with right_col:
    st.markdown(f'<p class="sec">Tomorrow\'s Prediction — {ticker}</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f'<div class="sig-rise"><div class="sig-icon">📈</div><div class="sig-label" style="color:#10b981;">PRICE RISE</div><div class="sig-sub">Model predicts tomorrow\'s close will be <strong>higher</strong><br>Confidence: {probability*100:.1f}%</div></div>', unsafe_allow_html=True)
        action, ac = "BUY", "#10b981"
        adesc = "Rise predicted → Buy 1 share"
    else:
        st.markdown(f'<div class="sig-fall"><div class="sig-icon">📉</div><div class="sig-label" style="color:#ef4444;">PRICE FALL</div><div class="sig-sub">Model predicts tomorrow\'s close will be <strong>lower</strong><br>Confidence: {probability*100:.1f}%</div></div>', unsafe_allow_html=True)
        action, ac = "SELL", "#ef4444"
        adesc = "Fall predicted → Sell / avoid buying"

    st.markdown('<p class="sec">Trading Signal</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:18px;text-align:center;"><div style="font-size:2.2rem;font-weight:700;color:{ac};font-family:\'JetBrains Mono\',monospace;">{action}</div><div style="color:#94a3b8;font-size:0.88rem;margin-top:6px;">{adesc}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="sec">Features Used</p>', unsafe_allow_html=True)
    feat_df = X_latest.T.rename(columns={X_latest.index[0]: "Value"})
    feat_df["Value"] = feat_df["Value"].round(6)
    st.dataframe(feat_df, use_container_width=True, height=min(35*len(feat_df)+38, 300))

    pred_date = (date.today() + timedelta(days=1)).strftime("%A, %d %B %Y")
    st.markdown(f'<div class="info" style="margin-top:10px;">Predicting for: <strong>{pred_date}</strong><br>Based on data up to: <strong>{latest[date_col]}</strong></div>', unsafe_allow_html=True)

with st.expander("📋 Raw price data"):
    st.dataframe(df_raw.tail(30), use_container_width=True)
