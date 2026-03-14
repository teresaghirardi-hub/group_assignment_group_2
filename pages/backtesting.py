"""
pages/backtesting.py — Backtesting Page (Part 1.3 + 2.2.3)
============================================================
Simulates a Buy-and-Sell trading strategy on historical data
using the trained ML model's predictions.

Strategy:
  Prediction = 1 (Rise) → BUY  1 share at today's close
  Prediction = 0 (Fall) → SELL all shares at today's close
  Final position liquidated at end of period.

ETL functions defined here directly — no separate etl.py import.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from pysimfin import PySimFin, SimFinAPIError, SimFinNotFoundError, SimFinRateLimitError

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backtesting | AlgoTrader", page_icon="🔁", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(255,255,255,0.06); }
.kpi { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px; text-align: center; }
.kpi-val { font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.kpi-lbl { font-size: 0.78rem; color: #64748b; margin-top: 5px; }
.sec { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 24px 0 12px 0; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.06); }
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
    api_key = st.text_input("SimFin API Key", type="password",
                            value=os.environ.get("SIMFIN_API_KEY", ""))
    TICKERS = ["AMZN", "AAPL", "MSFT", "GOOG", "TSLA"]
    ticker  = st.selectbox("Select Ticker", TICKERS)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("To",   value=date.today() - timedelta(days=1))
    initial_cash = st.number_input("Starting Capital ($)", value=10_000, step=1_000)
    st.markdown("---")
    run_btn = st.button("🔁 Run Backtest", use_container_width=True, type="primary")

# ── ETL functions (identical to etl_nuria.ipynb and go_live.py) ───────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8 technical features. Must match etl_nuria.ipynb exactly."""
    df  = df.copy()
    col = {c.lower(): c for c in df.columns}

    close  = df[col.get("close",  "close")]
    high   = df[col.get("high",   "high")]
    low    = df[col.get("low",    "low")]
    volume = df[col.get("volume", "volume")]

    df["Returns"]       = np.log(close / close.shift(1))
    df["SMA_5"]         = close.rolling(5).mean()
    df["SMA_20"]        = close.rolling(20).mean()
    df["Volatility_5"]  = df["Returns"].rolling(5).std()
    df["Volatility_20"] = df["Returns"].rolling(20).std()
    df["Volume_Change"] = volume.pct_change()

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI_14"]      = 100 - (100 / (1 + rs))
    df["Price_Range"] = (high - low) / close

    feature_cols = ["Returns","SMA_5","SMA_20","Volatility_5",
                    "Volatility_20","Volume_Change","RSI_14","Price_Range"]
    return df.dropna(subset=feature_cols).reset_index(drop=True)

# ── Model loader ───────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

@st.cache_resource
def load_model(ticker: str):
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

# ── Backtest simulation ────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, pipeline, features: list, initial_cash: float) -> pd.DataFrame:
    """
    Simulate Buy-and-Sell strategy over historical data.
    prediction=1 → BUY 1 share | prediction=0 → SELL all shares.
    """
    col_map   = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    date_col  = col_map.get("date",  "date")

    df = add_technical_features(df.copy())
    df = df.dropna(subset=features).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Not enough data after ETL. Try a wider date range.")

    predictions   = pipeline.predict(df[features].values)
    probabilities = pipeline.predict_proba(df[features].values)[:, 1]

    cash    = initial_cash
    shares  = 0
    records = []

    for i in range(len(df) - 1):
        price  = float(df[close_col].iloc[i])
        pred   = int(predictions[i])
        prob   = float(probabilities[i])
        action = "HOLD"

        if pred == 1 and cash >= price:
            shares += 1
            cash   -= price
            action  = "BUY"
        elif pred == 0 and shares > 0:
            cash  += shares * price
            shares = 0
            action = "SELL"

        records.append({
            "Date":            df[date_col].iloc[i],
            "Close":           price,
            "Prediction":      pred,
            "Probability":     round(prob, 4),
            "Action":          action,
            "Shares":          shares,
            "Cash":            round(cash, 2),
            "Portfolio Value": round(cash + shares * price, 2),
        })

    # Liquidate remaining position at last price
    if shares > 0:
        last_price = float(df[close_col].iloc[-1])
        cash      += shares * last_price
        records[-1]["Cash"]            = round(cash, 2)
        records[-1]["Shares"]          = 0
        records[-1]["Portfolio Value"] = round(cash, 2)
        records[-1]["Action"]          = "SELL (final)"

    results = pd.DataFrame(records)
    results["Date"] = pd.to_datetime(results["Date"])
    return results

# ── Charts ─────────────────────────────────────────────────────────────────────
def portfolio_chart(results, ticker, initial_cash):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Portfolio Value"],
        mode="lines", name="ML Strategy",
        line=dict(color="#63b3ed", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,179,237,0.05)"))

    buys  = results[results["Action"] == "BUY"]
    sells = results[results["Action"].isin(["SELL","SELL (final)"])]
    fig.add_trace(go.Scatter(x=buys["Date"],  y=buys["Portfolio Value"],
        mode="markers", name="BUY",  marker=dict(color="#10b981", size=8, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Portfolio Value"],
        mode="markers", name="SELL", marker=dict(color="#ef4444", size=8, symbol="triangle-down")))

    # Buy-and-hold baseline
    shares_bah = initial_cash / float(results["Close"].iloc[0])
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Close"] * shares_bah,
        mode="lines", name="Buy & Hold", line=dict(color="#94a3b8", width=1.5, dash="dot")))
    fig.add_hline(y=initial_cash, line_dash="dash", line_color="rgba(255,255,255,0.12)",
                  annotation_text="Starting capital", annotation_font_color="#64748b")

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        title=dict(text=f"{ticker} — Portfolio vs Buy & Hold", font=dict(color="#e2e8f0", size=15)),
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=400,
    )
    return fig

def price_signals_chart(results, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Close"],
        mode="lines", name="Close", line=dict(color="#63b3ed", width=1.8)))
    buys  = results[results["Action"] == "BUY"]
    sells = results[results["Action"].isin(["SELL","SELL (final)"])]
    fig.add_trace(go.Scatter(x=buys["Date"],  y=buys["Close"],
        mode="markers", name="BUY",  marker=dict(color="#10b981", size=8, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"],
        mode="markers", name="SELL", marker=dict(color="#ef4444", size=8, symbol="triangle-down")))
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        title=dict(text=f"{ticker} — Price with Trade Signals", font=dict(color="#e2e8f0", size=15)),
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## 🔁 Backtesting")
st.markdown('<p style="color:#64748b;margin-top:-8px;">Simulate how the model would have performed on historical data.</p>', unsafe_allow_html=True)
st.markdown('<div class="info"><strong>Strategy: Buy-and-Sell</strong> — Rise predicted → BUY 1 share at close. Fall predicted → SELL all shares at close. Compared against Buy-and-Hold baseline.</div>', unsafe_allow_html=True)

if not run_btn:
    st.info("👈 Configure settings and click **Run Backtest**.")
    st.stop()

if not api_key:
    st.error("Please enter your SimFin API key.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Load model
try:
    pipeline, model_features = load_model(ticker)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Fetch data (extra 60-day buffer for rolling window warm-up)
fetch_start = (start_date - timedelta(days=60)).strftime("%Y-%m-%d")
fetch_end   = end_date.strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker} historical data…"):
    try:
        client = PySimFin(api_key=api_key)
        df_raw = client.get_share_prices(ticker, start=fetch_start, end=fetch_end)
        st.write(df_raw.columns.tolist())
        st.write(df_raw.head(2))
        st.stop()
    except SimFinRateLimitError:
        st.error("Rate limit hit. Wait a moment and retry.")
        st.stop()
    except SimFinNotFoundError:
        st.error(f"Ticker '{ticker}' not found.")
        st.stop()
    except SimFinAPIError as e:
        st.error(f"SimFin API error: {e}")
        st.stop()

if df_raw.empty:
    st.error("No data returned.")
    st.stop()

df_raw.columns = df_raw.columns.str.lower() 

# Run backtest
try:
    with st.spinner("Running simulation…"):
        results = run_backtest(df_raw, pipeline, model_features, float(initial_cash))
except ValueError as e:
    st.error(str(e))
    st.stop()

# Trim to requested date range
results = results[results["Date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
if results.empty:
    st.error("No results in selected range after ETL. Try wider dates.")
    st.stop()

# KPIs
final_value  = float(results["Portfolio Value"].iloc[-1])
pct_return   = ((final_value - initial_cash) / initial_cash) * 100
first_price  = float(results["Close"].iloc[0])
last_price   = float(results["Close"].iloc[-1])
bah_return   = (((initial_cash / first_price * last_price) - initial_cash) / initial_cash) * 100
n_buys       = (results["Action"] == "BUY").sum()
n_sells      = results["Action"].isin(["SELL","SELL (final)"]).sum()

st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, val, lbl, color in zip(
    [k1,k2,k3,k4,k5,k6],
    [f"${final_value:,.0f}", f"{'+' if pct_return>=0 else ''}{pct_return:.1f}%",
     f"{'+' if bah_return>=0 else ''}{bah_return:.1f}%",
     str(n_buys), str(n_sells), f"${initial_cash:,.0f}"],
    ["Final Value","ML Return","Buy & Hold","BUY signals","SELL signals","Starting Capital"],
    ["#63b3ed",
     "#10b981" if pct_return>=0 else "#ef4444",
     "#10b981" if bah_return>=0 else "#ef4444",
     "#10b981","#ef4444","#94a3b8"],
):
    col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
st.plotly_chart(portfolio_chart(results, ticker, initial_cash), use_container_width=True)

st.markdown('<p class="sec">Trade Signals on Price Chart</p>', unsafe_allow_html=True)
st.plotly_chart(price_signals_chart(results, ticker), use_container_width=True)

st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
trade_log = results[results["Action"] != "HOLD"][
    ["Date","Close","Action","Shares","Cash","Portfolio Value","Probability"]
].copy()
trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
st.dataframe(trade_log, use_container_width=True, height=280)

with st.expander("📋 Full daily results"):
    full = results.copy()
    full["Date"] = full["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(full, use_container_width=True)
