"""
pages/backtesting.py — Backtesting Page
========================================
Simulates trading strategies using ML model predictions.

Binary Strategy:
  Prediction = 1 (Rise) → BUY 1 share
  Prediction = 0 (Fall) → SELL all shares

Multi-Class Strategy:
  0 (Big Fall)   → SELL ALL
  1 (Small Fall) → HOLD
  2 (Small Rise) → BUY 1 share
  3 (Big Rise)   → BUY 2 shares
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
from sklearn.linear_model import LogisticRegression

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


# ── ETL functions (must match notebook exactly) ────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24 technical features. Must match etl.ipynb exactly."""
    df = df.copy()

    col = {c.lower(): c for c in df.columns}
    close  = df[col.get("close",  "close")]
    high   = df[col.get("high",   "high")]
    low    = df[col.get("low",    "low")]
    volume = df[col.get("volume", "volume")]
    
    date_col = col.get("date", "date")
    df[date_col] = pd.to_datetime(df[date_col])

    # Original 8
    df["Returns"] = np.log(close / close.shift(1))
    df["SMA_5"]  = close.rolling(window=5).mean()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["Volatility_5"]  = df["Returns"].rolling(window=5).std()
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()
    df["Volume_Change"] = volume.pct_change()

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs       = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["Price_Range"] = (high - low) / close

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    df["BB_Width"] = (bb_upper - bb_lower) / sma_20
    df["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower)

    # Momentum
    df["Momentum_10"] = close / close.shift(10) - 1
    df["Momentum_20"] = close / close.shift(20) - 1

    # ATR Ratio
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["ATR_Ratio"] = atr_14 / close

    # Lagged Returns
    df["Return_Lag1"] = df["Returns"].shift(1)
    df["Return_Lag2"] = df["Returns"].shift(2)
    df["Return_Lag3"] = df["Returns"].shift(3)
    df["Return_Lag5"] = df["Returns"].shift(5)

    # Volume Ratio
    df["Volume_Ratio"] = volume / volume.rolling(20).mean()

    # Day of Week
    df["DayOfWeek"] = df[date_col].dt.dayofweek

    # Distance from SMAs
    df["Dist_SMA_5"] = (close - df["SMA_5"]) / df["SMA_5"]
    df["Dist_SMA_20"] = (close - df["SMA_20"]) / df["SMA_20"]

    return df


# ── Sklearn Compatibility Fix ──────────────────────────────────────────────────

def fix_sklearn_compatibility(obj):
    """
    Fix compatibility issues with models saved in older sklearn versions.
    
    In sklearn 1.5+, the 'multi_class' parameter was removed from LogisticRegression.
    Old models may not have this attribute, but new sklearn code tries to read it.
    
    Solution: ADD the attribute with default value 'deprecated' if it doesn't exist.
    """
    # Direct LogisticRegression
    if isinstance(obj, LogisticRegression):
        if not hasattr(obj, 'multi_class'):
            # Add the attribute with the default value expected by sklearn
            object.__setattr__(obj, 'multi_class', 'deprecated')
        return obj
    
    # Pipeline
    if hasattr(obj, 'steps'):
        for name, step in obj.steps:
            fix_sklearn_compatibility(step)
    
    # VotingClassifier or similar ensemble with estimators_
    if hasattr(obj, 'estimators_'):
        for est in obj.estimators_:
            fix_sklearn_compatibility(est)
    
    # VotingClassifier with estimators (list of tuples)
    if hasattr(obj, 'estimators'):
        for item in obj.estimators:
            if isinstance(item, tuple):
                fix_sklearn_compatibility(item[1])
            else:
                fix_sklearn_compatibility(item)
    
    # Named steps in pipeline
    if hasattr(obj, 'named_steps'):
        for name, step in obj.named_steps.items():
            fix_sklearn_compatibility(step)
    
    return obj


# ── Model loader ───────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"


@st.cache_resource
def load_model(ticker: str, model_type: str):
    model_path = MODELS_DIR / f"model_{ticker}_{model_type}.joblib"
    features_path = MODELS_DIR / f"features_{ticker}_{model_type}.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    pipeline = joblib.load(model_path)
    
    # Fix sklearn compatibility BEFORE using the model
    pipeline = fix_sklearn_compatibility(pipeline)

    with open(features_path) as f:
        features = [line.strip() for line in f if line.strip()]

    return pipeline, features


# ── Backtest simulation ────────────────────────────────────────────────────────

def run_backtest_binary(df: pd.DataFrame, pipeline, feature_cols: list, initial_cash: float):
    """
    Binary strategy: BUY on Rise (1), SELL on Fall (0).
    """
    df = add_technical_features(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    date_col = col_map.get("date", "date")
    
    results = []
    cash = initial_cash
    shares = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        X = row[feature_cols].values.reshape(1, -1)
        pred = int(pipeline.predict(X)[0])
        price = float(row[close_col])
        
        action = "HOLD"
        
        if pred == 1 and cash >= price:  # Rise → BUY
            shares += 1
            cash -= price
            action = "BUY"
        elif pred == 0 and shares > 0:   # Fall → SELL ALL
            cash += shares * price
            action = f"SELL {shares}"
            shares = 0
        
        portfolio_value = cash + shares * price
        
        results.append({
            "Date": row[date_col],
            "Close": price,
            "Prediction": pred,
            "Action": action,
            "Shares": shares,
            "Cash": cash,
            "Portfolio Value": portfolio_value,
        })
    
    # Final liquidation
    if shares > 0:
        final_price = float(df.iloc[-1][close_col])
        cash += shares * final_price
        results[-1]["Action"] = f"SELL {shares} (final)"
        results[-1]["Shares"] = 0
        results[-1]["Cash"] = cash
        results[-1]["Portfolio Value"] = cash
    
    return pd.DataFrame(results)


def run_backtest_multi(df: pd.DataFrame, pipeline, feature_cols: list, initial_cash: float):
    """
    Multi-class strategy:
      0 (Big Fall)   → SELL ALL
      1 (Small Fall) → HOLD
      2 (Small Rise) → BUY 1
      3 (Big Rise)   → BUY 2
    """
    df = add_technical_features(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    date_col = col_map.get("date", "date")
    
    CLASS_NAMES = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]
    
    results = []
    cash = initial_cash
    shares = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        X = row[feature_cols].values.reshape(1, -1)
        pred = int(pipeline.predict(X)[0])
        price = float(row[close_col])
        
        action = "HOLD"
        
        if pred == 0 and shares > 0:        # Big Fall → SELL ALL
            cash += shares * price
            action = f"SELL {shares}"
            shares = 0
        elif pred == 2 and cash >= price:   # Small Rise → BUY 1
            shares += 1
            cash -= price
            action = "BUY 1"
        elif pred == 3 and cash >= 2*price: # Big Rise → BUY 2
            shares += 2
            cash -= 2 * price
            action = "BUY 2"
        elif pred == 3 and cash >= price:   # Big Rise but only afford 1
            shares += 1
            cash -= price
            action = "BUY 1"
        
        portfolio_value = cash + shares * price
        
        results.append({
            "Date": row[date_col],
            "Close": price,
            "Prediction": pred,
            "Pred Name": CLASS_NAMES[pred],
            "Action": action,
            "Shares": shares,
            "Cash": cash,
            "Portfolio Value": portfolio_value,
        })
    
    # Final liquidation
    if shares > 0:
        final_price = float(df.iloc[-1][close_col])
        cash += shares * final_price
        results[-1]["Action"] = f"SELL {shares} (final)"
        results[-1]["Shares"] = 0
        results[-1]["Cash"] = cash
        results[-1]["Portfolio Value"] = cash
    
    return pd.DataFrame(results)


# ── Charts ─────────────────────────────────────────────────────────────────────

def portfolio_chart(results: pd.DataFrame, ticker: str, initial_cash: float, strategy_name: str):
    # Buy & hold line
    first_price = float(results["Close"].iloc[0])
    bah_shares = initial_cash / first_price
    results["Buy & Hold"] = results["Close"] * bah_shares
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["Date"], y=results["Portfolio Value"],
        mode="lines", name=f"ML Strategy",
        line=dict(color="#63b3ed", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=results["Date"], y=results["Buy & Hold"],
        mode="lines", name="Buy & Hold",
        line=dict(color="#94a3b8", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"{ticker} — {strategy_name} vs Buy & Hold",
        template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig


def price_signals_chart(results: pd.DataFrame, ticker: str):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=results["Date"], y=results["Close"],
        mode="lines", name="Price",
        line=dict(color="#e2e8f0", width=1.5),
    ))
    
    # BUY signals
    buys = results[results["Action"].str.contains("BUY", na=False) & ~results["Action"].str.contains("final", na=False)]
    fig.add_trace(go.Scatter(
        x=buys["Date"], y=buys["Close"],
        mode="markers", name="BUY",
        marker=dict(color="#10b981", size=10, symbol="triangle-up"),
    ))
    
    # SELL signals
    sells = results[results["Action"].str.contains("SELL", na=False)]
    fig.add_trace(go.Scatter(
        x=sells["Date"], y=sells["Close"],
        mode="markers", name="SELL",
        marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
    ))
    
    fig.update_layout(
        title=f"{ticker} — Trade Signals",
        template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig


def display_kpis(results, initial_cash):
    final_value  = float(results["Portfolio Value"].iloc[-1])
    pct_return   = ((final_value - initial_cash) / initial_cash) * 100
    first_price  = float(results["Close"].iloc[0])
    last_price   = float(results["Close"].iloc[-1])
    bah_return   = (((initial_cash / first_price * last_price) - initial_cash) / initial_cash) * 100
    n_buys       = results["Action"].str.contains("BUY", na=False).sum() - results["Action"].str.contains("final", na=False).sum()
    n_sells      = results["Action"].str.contains("SELL", na=False).sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    for col, val, lbl, color in zip(
        [k1,k2,k3,k4,k5,k6],
        [f"${final_value:,.0f}", f"{'+' if pct_return>=0 else ''}{pct_return:.1f}%",
         f"{'+' if bah_return>=0 else ''}{bah_return:.1f}%",
         str(max(0, n_buys)), str(n_sells), f"${initial_cash:,.0f}"],
        ["Final Value","ML Return","Buy & Hold","BUY signals","SELL signals","Starting Capital"],
        ["#63b3ed",
         "#10b981" if pct_return>=0 else "#ef4444",
         "#10b981" if bah_return>=0 else "#ef4444",
         "#10b981","#ef4444","#94a3b8"],
    ):
        col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # Return comparison
    outperform = pct_return - bah_return
    if outperform > 0:
        st.success(f"✅ ML Strategy **outperformed** Buy & Hold by **{outperform:.1f}%**")
    elif outperform < 0:
        st.warning(f"⚠️ ML Strategy **underperformed** Buy & Hold by **{abs(outperform):.1f}%**")
    else:
        st.info("ML Strategy matched Buy & Hold performance")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## 🔁 Backtesting")
st.markdown('<p style="color:#64748b;margin-top:-8px;">Simulate how the model would have performed on historical data.</p>', unsafe_allow_html=True)

if not run_btn:
    st.markdown('''
    <div class="info">
        <strong>Two strategies available:</strong><br>
        • <strong>Binary:</strong> BUY on Rise prediction, SELL on Fall prediction<br>
        • <strong>Multi-Class:</strong> Position sizing based on prediction confidence (Big/Small movements)
    </div>
    ''', unsafe_allow_html=True)
    st.info("👈 Configure settings and click **Run Backtest**.")
    st.stop()

if not api_key:
    st.error("Please enter your SimFin API key.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Fetch data (extra buffer for rolling windows)
fetch_start = (start_date - timedelta(days=60)).strftime("%Y-%m-%d")
fetch_end   = end_date.strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker} historical data…"):
    try:
        client = PySimFin(api_key=api_key)
        df_raw = client.get_share_prices(ticker, start=fetch_start, end=fetch_end)
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

# ── TABS: Binary vs Multi-Class ────────────────────────────────────────────────
tab_binary, tab_multi = st.tabs(["📊 Binary Strategy", "📊 Multi-Class Strategy"])

# ── Binary Tab ─────────────────────────────────────────────────────────────────
with tab_binary:
    st.markdown('''
    <div class="info">
        <strong>Binary Strategy:</strong> Rise predicted → BUY 1 share | Fall predicted → SELL all shares
    </div>
    ''', unsafe_allow_html=True)
    
    try:
        pipeline_binary, features_binary = load_model(ticker, "binary")
        
        with st.spinner("Running binary simulation…"):
            results_binary = run_backtest_binary(df_raw.copy(), pipeline_binary, features_binary, float(initial_cash))
        
        # Trim to requested date range
        results_binary = results_binary[results_binary["Date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
        
        if results_binary.empty:
            st.error("No results in selected range after ETL. Try wider dates.")
        else:
            st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
            display_kpis(results_binary, initial_cash)
            
            st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
            st.plotly_chart(portfolio_chart(results_binary, ticker, initial_cash, "Binary Strategy"), use_container_width=True)
            
            st.markdown('<p class="sec">Trade Signals</p>', unsafe_allow_html=True)
            st.plotly_chart(price_signals_chart(results_binary, ticker), use_container_width=True)
            
            st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
            trade_log = results_binary[results_binary["Action"] != "HOLD"][
                ["Date","Close","Action","Shares","Cash","Portfolio Value"]
            ].copy()
            trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(trade_log, use_container_width=True, height=280)
            
    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))

# ── Multi-Class Tab ────────────────────────────────────────────────────────────
with tab_multi:
    st.markdown('''
    <div class="info">
        <strong>Multi-Class Strategy:</strong><br>
        • Big Fall (0) → SELL ALL<br>
        • Small Fall (1) → HOLD<br>
        • Small Rise (2) → BUY 1 share<br>
        • Big Rise (3) → BUY 2 shares
    </div>
    ''', unsafe_allow_html=True)
    
    try:
        pipeline_multi, features_multi = load_model(ticker, "multi")
        
        with st.spinner("Running multi-class simulation…"):
            results_multi = run_backtest_multi(df_raw.copy(), pipeline_multi, features_multi, float(initial_cash))
        
        # Trim to requested date range
        results_multi = results_multi[results_multi["Date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
        
        if results_multi.empty:
            st.error("No results in selected range after ETL. Try wider dates.")
        else:
            st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
            display_kpis(results_multi, initial_cash)
            
            st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
            st.plotly_chart(portfolio_chart(results_multi, ticker, initial_cash, "Multi-Class Strategy"), use_container_width=True)
            
            st.markdown('<p class="sec">Trade Signals</p>', unsafe_allow_html=True)
            st.plotly_chart(price_signals_chart(results_multi, ticker), use_container_width=True)
            
            # Prediction distribution
            st.markdown('<p class="sec">Prediction Distribution</p>', unsafe_allow_html=True)
            pred_counts = results_multi["Pred Name"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            class_colors = {"Big Fall": "#ef4444", "Small Fall": "#f97316", "Small Rise": "#22c55e", "Big Rise": "#10b981"}
            for col, name in zip([c1,c2,c3,c4], ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]):
                count = pred_counts.get(name, 0)
                pct = (count / len(results_multi)) * 100
                col.markdown(f'''
                <div class="kpi">
                    <div class="kpi-val" style="color:{class_colors[name]};">{count}</div>
                    <div class="kpi-lbl">{name} ({pct:.1f}%)</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
            trade_log = results_multi[results_multi["Action"] != "HOLD"][
                ["Date","Close","Pred Name","Action","Shares","Cash","Portfolio Value"]
            ].copy()
            trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(trade_log, use_container_width=True, height=280)
            
    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))

# Full data expander
with st.expander("📋 Full daily results"):
    st.info("Select a tab above to see the full results for that strategy.")