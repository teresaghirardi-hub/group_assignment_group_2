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

import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pysimfin import PySimFin, SimFinAPIError, SimFinNotFoundError, SimFinRateLimitError
from etl import get_api_key, add_technical_features, load_model

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


# ── Backtest simulation ────────────────────────────────────────────────────────

def run_backtest_binary(df: pd.DataFrame, pipeline, feature_cols: list, initial_cash: float):
    """Binary strategy: BUY on Rise (1), SELL on Fall (0)."""
    df = add_technical_features(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    col_map   = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    date_col  = col_map.get("date",  "date")

    results = []
    cash, shares = initial_cash, 0

    for i in range(len(df)):
        row   = df.iloc[i]
        X     = row[feature_cols].values.reshape(1, -1)
        pred  = int(pipeline.predict(X)[0])
        price = float(row[close_col])
        action = "HOLD"

        if pred == 1 and cash >= price:   # Rise → BUY
            shares += 1
            cash   -= price
            action  = "BUY"
        elif pred == 0 and shares > 0:    # Fall → SELL ALL
            cash  += shares * price
            action = f"SELL {shares}"
            shares = 0

        results.append({
            "Date":            row[date_col],
            "Close":           price,
            "Prediction":      pred,
            "Action":          action,
            "Shares":          shares,
            "Cash":            cash,
            "Portfolio Value": cash + shares * price,
        })

    # Final liquidation
    if shares > 0:
        final_price = float(df.iloc[-1][close_col])
        cash += shares * final_price
        results[-1].update({"Action": f"SELL {shares} (final)", "Shares": 0,
                             "Cash": cash, "Portfolio Value": cash})

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

    col_map    = {c.lower(): c for c in df.columns}
    close_col  = col_map.get("close", "close")
    date_col   = col_map.get("date",  "date")
    CLASS_NAMES = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]

    results = []
    cash, shares = initial_cash, 0

    for i in range(len(df)):
        row   = df.iloc[i]
        X     = row[feature_cols].values.reshape(1, -1)
        pred  = int(pipeline.predict(X)[0])
        price = float(row[close_col])
        action = "HOLD"

        if pred == 0 and shares > 0:         # Big Fall → SELL ALL
            cash  += shares * price
            action = f"SELL {shares}"
            shares = 0
        elif pred == 2 and cash >= price:    # Small Rise → BUY 1
            shares += 1
            cash   -= price
            action  = "BUY 1"
        elif pred == 3 and cash >= 2 * price: # Big Rise → BUY 2
            shares += 2
            cash   -= 2 * price
            action  = "BUY 2"
        elif pred == 3 and cash >= price:    # Big Rise but can only afford 1
            shares += 1
            cash   -= price
            action  = "BUY 1"

        results.append({
            "Date":            row[date_col],
            "Close":           price,
            "Prediction":      pred,
            "Pred Name":       CLASS_NAMES[pred],
            "Action":          action,
            "Shares":          shares,
            "Cash":            cash,
            "Portfolio Value": cash + shares * price,
        })

    # Final liquidation
    if shares > 0:
        final_price = float(df.iloc[-1][close_col])
        cash += shares * final_price
        results[-1].update({"Action": f"SELL {shares} (final)", "Shares": 0,
                             "Cash": cash, "Portfolio Value": cash})

    return pd.DataFrame(results)


# ── Charts ─────────────────────────────────────────────────────────────────────

def portfolio_chart(results: pd.DataFrame, ticker: str, initial_cash: float, strategy_name: str):
    first_price       = float(results["Close"].iloc[0])
    bah_shares        = initial_cash / first_price
    results           = results.copy()
    results["Buy & Hold"] = results["Close"] * bah_shares

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["Date"], y=results["Portfolio Value"],
        mode="lines", name="ML Strategy",
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
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig


def price_signals_chart(results: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["Date"], y=results["Close"],
        mode="lines", name="Price",
        line=dict(color="#e2e8f0", width=1.5),
    ))
    buys = results[
        results["Action"].str.contains("BUY", na=False) &
        ~results["Action"].str.contains("final", na=False)
    ]
    fig.add_trace(go.Scatter(
        x=buys["Date"], y=buys["Close"],
        mode="markers", name="BUY",
        marker=dict(color="#10b981", size=10, symbol="triangle-up"),
    ))
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
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig


def display_kpis(results: pd.DataFrame, initial_cash: float):
    final_value  = float(results["Portfolio Value"].iloc[-1])
    pct_return   = ((final_value - initial_cash) / initial_cash) * 100
    first_price  = float(results["Close"].iloc[0])
    last_price   = float(results["Close"].iloc[-1])
    bah_return   = (((initial_cash / first_price * last_price) - initial_cash) / initial_cash) * 100
    n_buys  = results["Action"].str.contains("BUY",  na=False).sum() - \
              results["Action"].str.contains("final", na=False).sum()
    n_sells = results["Action"].str.contains("SELL", na=False).sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for col, val, lbl, color in zip(
        [k1, k2, k3, k4, k5, k6],
        [f"${final_value:,.0f}",
         f"{'+' if pct_return >= 0 else ''}{pct_return:.1f}%",
         f"{'+' if bah_return >= 0 else ''}{bah_return:.1f}%",
         str(max(0, n_buys)), str(n_sells), f"${initial_cash:,.0f}"],
        ["Final Value", "ML Return", "Buy & Hold", "BUY signals", "SELL signals", "Starting Capital"],
        ["#63b3ed",
         "#10b981" if pct_return >= 0 else "#ef4444",
         "#10b981" if bah_return >= 0 else "#ef4444",
         "#10b981", "#ef4444", "#94a3b8"],
    ):
        col.markdown(
            f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

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

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Resolve API key from secrets / environment (stops with error if missing)
api_key = get_api_key()

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
                ["Date", "Close", "Action", "Shares", "Cash", "Portfolio Value"]
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
            st.markdown('<p class="sec">Prediction Distribution</p>', unsafe_allow_html=True)
            pred_counts  = results_multi["Pred Name"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            class_colors = {"Big Fall": "#ef4444", "Small Fall": "#f97316",
                            "Small Rise": "#22c55e", "Big Rise": "#10b981"}
            for col, name in zip([c1, c2, c3, c4], ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]):
                cnt = pred_counts.get(name, 0)
                pct = (cnt / len(results_multi)) * 100
                col.markdown(f'''
                <div class="kpi">
                    <div class="kpi-val" style="color:{class_colors[name]};">{cnt}</div>
                    <div class="kpi-lbl">{name} ({pct:.1f}%)</div>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
            trade_log = results_multi[results_multi["Action"] != "HOLD"][
                ["Date", "Close", "Pred Name", "Action", "Shares", "Cash", "Portfolio Value"]
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
