"""
pages/backtesting.py — Backtesting Page
========================================
Two strategy modes available:

SIMPLE MODE (original behaviour):
  Binary:     Rise → BUY 1 share  | Fall → SELL ALL
  Multi:      Big Fall → SELL ALL | Small Fall → HOLD
              Small Rise → BUY 1  | Big Rise → BUY 2

ADVANCED MODE:
  Trades execute at next-day OPEN (not today's close)
  Position size = configurable % of current cash
  Only trades when model confidence >= threshold
  Transaction costs: commission + slippage per trade
  Full risk metrics: Sharpe, max drawdown, win rate
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pysimfin import PySimFin, SimFinAPIError, SimFinNotFoundError, SimFinRateLimitError
from etl import get_api_key, add_technical_features, load_model
from team_photos import logo

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backtesting | AUGUR Analytics", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
header[data-testid="stHeader"] { background: #0a0e1a !important; border-bottom: none !important; }
[data-testid="stDecoration"] { display: none !important; }
/* Lighter sidebar so text is readable */
section[data-testid="stSidebar"] {
    background: #1a2744;
    border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] a:hover { color: #63b3ed !important; }

/* ── Sidebar widgets: all match sidebar bg with light text ── */

/* Selectbox */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1a2744 !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #e2e8f0 !important; }

/* Selectbox / date / number input popup list */
[data-baseweb="menu"] { background-color: #1e3258 !important; }
[data-baseweb="menu"] li { color: #e2e8f0 !important; background-color: #1e3258 !important; }
[data-baseweb="menu"] li:hover { background-color: #2a4070 !important; }
[data-baseweb="menu"] [aria-selected="true"] { background-color: rgba(37,99,235,0.35) !important; }

/* Text / date / number inputs */
section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: #1a2744 !important;
    border-color: rgba(255,255,255,0.2) !important;
}
section[data-testid="stSidebar"] [data-baseweb="base-input"] input {
    background-color: #1a2744 !important;
    color: #e2e8f0 !important;
}
/* Number input +/- buttons */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: #243b60 !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* Radio buttons */
section[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    background-color: #1a2744 !important;
    border-color: rgba(255,255,255,0.3) !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] {
    color: #e2e8f0 !important;
}

/* Slider track & thumb */
section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background-color: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #63b3ed !important;
    border-color: #63b3ed !important;
}

/* ══ DATE PICKER — nuclear override ══ */

[data-baseweb="base-input"] {
    background-color: #1a2744 !important;
    border-color: rgba(255,255,255,0.2) !important;
}
[data-baseweb="base-input"] input {
    background-color: #1a2744 !important;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    color-scheme: dark !important;
}

[data-baseweb="calendar"],
[data-baseweb="calendar"] div,
[data-baseweb="calendar"] span,
[data-baseweb="calendar"] p,
[data-baseweb="calendar"] button,
[data-baseweb="calendar"] [role="gridcell"],
[data-baseweb="calendar"] [role="columnheader"] {
    background-color: #1e2d4a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: rgba(255,255,255,0.1) !important;
}

[data-baseweb="calendar"] [role="columnheader"],
[data-baseweb="calendar"] [role="columnheader"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-baseweb="calendar"] button:hover,
[data-baseweb="calendar"] button:hover * {
    background-color: rgba(124,58,237,0.3) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-baseweb="calendar"] button[aria-selected="true"],
[data-baseweb="calendar"] button[aria-selected="true"] * {
    background-color: #7c3aed !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

[data-baseweb="calendar"] button[aria-label*="previous"],
[data-baseweb="calendar"] button[aria-label*="next"],
[data-baseweb="calendar"] button[aria-label*="Previous"],
[data-baseweb="calendar"] button[aria-label*="Next"] {
    background-color: #1e2d4a !important;
    color: #e2e8f0 !important;
}
[data-baseweb="calendar"] button[aria-label*="previous"] svg,
[data-baseweb="calendar"] button[aria-label*="next"] svg,
[data-baseweb="calendar"] button[aria-label*="Previous"] svg,
[data-baseweb="calendar"] button[aria-label*="Next"] svg {
    fill: #e2e8f0 !important;
    stroke: #e2e8f0 !important;
}

[data-baseweb="calendar"] [data-baseweb="select"] > div,
[data-baseweb="calendar"] select {
    background-color: #1e2d4a !important;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.15) !important;
}


/* Hide auto-generated Streamlit nav (we use our own) */
[data-testid="stSidebarNavItems"] { display: none; }
.kpi { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px; text-align: center; }
.kpi-val { font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.kpi-lbl { font-size: 0.78rem; color: #64748b; margin-top: 5px; }
.sec { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 24px 0 12px 0; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.info { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2); border-radius: 8px; padding: 10px 14px; font-size: 0.87rem; color: #94a3b8; margin-bottom: 8px; }
.warn { background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.2); border-radius: 8px; padding: 10px 14px; font-size: 0.87rem; color: #fbbf24; margin-bottom: 8px; }
.mode-badge-simple   { display:inline-block; background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.4); color:#63b3ed; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-family:'JetBrains Mono',monospace; }
.mode-badge-advanced { display:inline-block; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#10b981; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<img src="data:image/png;base64,{logo}" '
        f'style="width:100%;max-width:200px;margin:0 auto 4px;display:block;">',
        unsafe_allow_html=True
    )

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
    st.markdown("#### Strategy Mode")
    strategy_mode = st.radio(
        "Choose mode",
        ["🟦  Simple", "🟩  Advanced"],
        help=(
            "Simple: fixed share counts, trades at today's close, no costs.\n"
            "Advanced: portfolio % sizing, confidence filter, next-day open execution, "
            "transaction costs, full risk metrics."
        ),
    )
    is_advanced = strategy_mode == "🟩  Advanced"

    if is_advanced:
        st.markdown("#### Advanced Settings")
        confidence_threshold = st.slider(
            "Min. confidence to trade",
            min_value=0.50, max_value=0.90, value=0.55, step=0.05,
            help="Only trade when the model's predicted probability exceeds this value.",
        )
        alloc_pct = st.slider(
            "Capital per trade (%)",
            min_value=5, max_value=100, value=20, step=5,
            help="Fraction of current cash deployed on each BUY signal.",
        )
        commission = st.number_input(
            "Commission per trade ($)",
            min_value=0.0, max_value=50.0, value=1.0, step=0.5,
        )
        slippage_pct = st.slider(
            "Slippage (%)",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Execution price is worse than quoted open by this fraction.",
        )
    else:
        confidence_threshold = 0.50
        alloc_pct            = 20
        commission           = 0.0
        slippage_pct         = 0.0

    st.markdown("---")
    run_btn = st.button("🔁 Run Backtest", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE BACKTEST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_simple_binary(df, pipeline, feature_cols, initial_cash):
    """Rise → BUY 1 share | Fall → SELL ALL. Executes at today's close."""
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)
    col_map   = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    date_col  = col_map.get("date",  "date")

    results = []
    cash, shares = initial_cash, 0

    for i in range(len(df)):
        row    = df.iloc[i]
        X      = row[feature_cols].values.reshape(1, -1)
        pred   = int(pipeline.predict(X)[0])
        price  = float(row[close_col])
        action = "HOLD"

        if pred == 1 and cash >= price:
            shares += 1; cash -= price; action = "BUY"
        elif pred == 0 and shares > 0:
            cash += shares * price; action = f"SELL {shares}"; shares = 0

        results.append({"Date": row[date_col], "Close": price, "Prediction": pred,
                        "Action": action, "Shares": shares, "Cash": round(cash, 2),
                        "Portfolio Value": round(cash + shares * price, 2), "Trade PnL": np.nan})

    if shares > 0:
        fp = float(df.iloc[-1][close_col]); cash += shares * fp
        results[-1].update({"Action": f"SELL {shares} (final)", "Shares": 0,
                             "Cash": round(cash, 2), "Portfolio Value": round(cash, 2)})
    return pd.DataFrame(results)


def run_simple_multi(df, pipeline, feature_cols, initial_cash):
    """Big Fall → SELL ALL | Small Fall → HOLD | Small Rise → BUY 1 | Big Rise → BUY 2."""
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)
    col_map    = {c.lower(): c for c in df.columns}
    close_col  = col_map.get("close", "close")
    date_col   = col_map.get("date",  "date")
    CLASS_NAMES = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]

    results = []
    cash, shares = initial_cash, 0

    for i in range(len(df)):
        row    = df.iloc[i]
        X      = row[feature_cols].values.reshape(1, -1)
        pred   = int(pipeline.predict(X)[0])
        price  = float(row[close_col])
        action = "HOLD"

        if pred == 0 and shares > 0:
            cash += shares * price; action = f"SELL {shares}"; shares = 0
        elif pred == 2 and cash >= price:
            shares += 1; cash -= price; action = "BUY 1"
        elif pred == 3 and cash >= 2 * price:
            shares += 2; cash -= 2 * price; action = "BUY 2"
        elif pred == 3 and cash >= price:
            shares += 1; cash -= price; action = "BUY 1"

        results.append({"Date": row[date_col], "Close": price, "Prediction": pred,
                        "Pred Name": CLASS_NAMES[pred], "Action": action, "Shares": shares,
                        "Cash": round(cash, 2), "Portfolio Value": round(cash + shares * price, 2),
                        "Trade PnL": np.nan})

    if shares > 0:
        fp = float(df.iloc[-1][close_col]); cash += shares * fp
        results[-1].update({"Action": f"SELL {shares} (final)", "Shares": 0,
                             "Cash": round(cash, 2), "Portfolio Value": round(cash, 2)})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  ADVANCED BACKTEST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_costs(price, side, commission, slippage_pct):
    slip = price * slippage_pct / 100
    return (price + slip if side == "buy" else price - slip), commission


def run_advanced_binary(df, pipeline, feature_cols, initial_cash,
                        confidence_threshold, alloc_pct, commission, slippage_pct):
    """
    Rise, conf >= threshold → BUY (alloc_pct % of cash) at NEXT-DAY open.
    Fall, conf >= threshold → SELL ALL at NEXT-DAY open.
    """
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)
    col_map   = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    open_col  = col_map.get("open",  "open")
    date_col  = col_map.get("date",  "date")

    X_all  = df[feature_cols].values
    preds  = pipeline.predict(X_all)
    probas = pipeline.predict_proba(X_all)
    conf   = probas[np.arange(len(preds)), preds]

    results = []
    cash, shares, avg_cost = initial_cash, 0.0, 0.0

    for i in range(len(df) - 1):
        row        = df.iloc[i]
        next_row   = df.iloc[i + 1]
        pred       = int(preds[i])
        confidence = float(conf[i])
        exec_price = float(next_row[open_col])
        action     = "HOLD"
        trade_pnl  = np.nan

        if pred == 1 and confidence >= confidence_threshold and cash > 0:
            buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
            spend = cash * (alloc_pct / 100) - cost
            if spend > 0:
                new_sh   = spend / buy_price
                avg_cost = (avg_cost * shares + buy_price * new_sh) / (shares + new_sh)
                shares  += new_sh; cash -= (spend + cost); action = "BUY"

        elif pred == 0 and confidence >= confidence_threshold and shares > 0:
            sell_price, cost = apply_costs(exec_price, "sell", commission, slippage_pct)
            proceeds  = shares * sell_price - cost
            trade_pnl = proceeds - shares * avg_cost
            cash += proceeds; action = f"SELL {shares:.3f}"; shares = 0.0; avg_cost = 0.0

        results.append({"Date": row[date_col], "Close": float(row[close_col]),
                        "Exec Price": exec_price, "Prediction": pred,
                        "Confidence": round(confidence * 100, 1), "Action": action,
                        "Shares": round(shares, 4), "Cash": round(cash, 2),
                        "Portfolio Value": round(cash + shares * float(row[close_col]), 2),
                        "Trade PnL": round(trade_pnl, 2) if not np.isnan(trade_pnl) else np.nan})

    if shares > 0:
        lp = float(df.iloc[-2][close_col])
        sp, cost = apply_costs(lp, "sell", commission, slippage_pct)
        proceeds = shares * sp - cost; trade_pnl = proceeds - shares * avg_cost; cash += proceeds
        results[-1].update({"Action": f"SELL {shares:.3f} (final)", "Shares": 0.0,
                             "Cash": round(cash, 2), "Portfolio Value": round(cash, 2),
                             "Trade PnL": round(trade_pnl, 2)})
    return pd.DataFrame(results)


def run_advanced_multi(df, pipeline, feature_cols, initial_cash,
                       confidence_threshold, alloc_pct, commission, slippage_pct):
    """
    Big Fall,   conf >= threshold → SELL ALL at NEXT-DAY open.
    Small Fall                    → HOLD.
    Small Rise, conf >= threshold → BUY (alloc_pct % of cash).
    Big Rise,   conf >= threshold → BUY (2x alloc_pct, capped at cash).
    """
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)
    col_map    = {c.lower(): c for c in df.columns}
    close_col  = col_map.get("close", "close")
    open_col   = col_map.get("open",  "open")
    date_col   = col_map.get("date",  "date")
    CLASS_NAMES = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]

    X_all  = df[feature_cols].values
    preds  = pipeline.predict(X_all)
    probas = pipeline.predict_proba(X_all)
    conf   = probas[np.arange(len(preds)), preds]

    results = []
    cash, shares, avg_cost = initial_cash, 0.0, 0.0

    for i in range(len(df) - 1):
        row        = df.iloc[i]
        next_row   = df.iloc[i + 1]
        pred       = int(preds[i])
        confidence = float(conf[i])
        exec_price = float(next_row[open_col])
        action     = "HOLD"
        trade_pnl  = np.nan

        if confidence >= confidence_threshold:
            if pred == 0 and shares > 0:
                sell_price, cost = apply_costs(exec_price, "sell", commission, slippage_pct)
                proceeds  = shares * sell_price - cost
                trade_pnl = proceeds - shares * avg_cost
                cash += proceeds; action = f"SELL {shares:.3f}"; shares = 0.0; avg_cost = 0.0

            elif pred == 2 and cash > 0:
                buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
                spend = cash * (alloc_pct / 100) - cost
                if spend > 0:
                    new_sh   = spend / buy_price
                    avg_cost = (avg_cost * shares + buy_price * new_sh) / (shares + new_sh)
                    shares  += new_sh; cash -= (spend + cost); action = "BUY 1×"

            elif pred == 3 and cash > 0:
                buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
                spend = min(cash * (alloc_pct / 100) * 2, cash) - cost
                if spend > 0:
                    new_sh   = spend / buy_price
                    avg_cost = (avg_cost * shares + buy_price * new_sh) / (shares + new_sh)
                    shares  += new_sh; cash -= (spend + cost); action = "BUY 2×"

        results.append({"Date": row[date_col], "Close": float(row[close_col]),
                        "Exec Price": exec_price, "Prediction": pred,
                        "Pred Name": CLASS_NAMES[pred], "Confidence": round(confidence * 100, 1),
                        "Action": action, "Shares": round(shares, 4), "Cash": round(cash, 2),
                        "Portfolio Value": round(cash + shares * float(row[close_col]), 2),
                        "Trade PnL": round(trade_pnl, 2) if not np.isnan(trade_pnl) else np.nan})

    if shares > 0:
        lp = float(df.iloc[-2][close_col])
        sp, cost = apply_costs(lp, "sell", commission, slippage_pct)
        proceeds = shares * sp - cost; trade_pnl = proceeds - shares * avg_cost; cash += proceeds
        results[-1].update({"Action": f"SELL {shares:.3f} (final)", "Shares": 0.0,
                             "Cash": round(cash, 2), "Portfolio Value": round(cash, 2),
                             "Trade PnL": round(trade_pnl, 2)})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_metrics(results, initial_cash, rf_annual=0.05):
    pv      = results["Portfolio Value"].values
    returns = np.diff(pv) / pv[:-1]

    total_return = (pv[-1] - initial_cash) / initial_cash * 100
    n_days       = len(pv)
    ann_return   = ((pv[-1] / initial_cash) ** (252 / max(n_days, 1)) - 1) * 100

    rf_daily = rf_annual / 252
    excess   = returns - rf_daily
    sharpe   = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    peak   = np.maximum.accumulate(pv)
    max_dd = ((pv - peak) / peak).min() * 100

    trade_pnl = results["Trade PnL"].dropna()
    win_rate  = (trade_pnl > 0).mean() * 100 if len(trade_pnl) > 0 else 0.0
    n_trades  = len(trade_pnl)

    return {"total_return": total_return, "ann_return": ann_return,
            "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate, "n_trades": n_trades}


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_chart(results, ticker, initial_cash, strategy_name):
    bah = results["Close"] * (initial_cash / float(results["Close"].iloc[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Portfolio Value"],
                             mode="lines", name="ML Strategy", line=dict(color="#63b3ed", width=2)))
    fig.add_trace(go.Scatter(x=results["Date"], y=bah, mode="lines", name="Buy & Hold",
                             line=dict(color="#94a3b8", width=2, dash="dot")))
    fig.update_layout(title=f"{ticker} — {strategy_name} vs Buy & Hold",
                      template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                      xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=40,b=0), height=350)
    return fig


def drawdown_chart(results, ticker):
    pv  = results["Portfolio Value"].values
    dd  = (pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=dd, mode="lines",
                             line=dict(color="#ef4444", width=1.5),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
    fig.update_layout(title=f"{ticker} — Drawdown (%)", template="plotly_dark",
                      paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                      xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
                      margin=dict(l=0,r=0,t=40,b=0), height=220)
    return fig


def signals_chart(results, ticker):
    buys  = results[results["Action"].str.contains("BUY",  na=False) &
                    ~results["Action"].str.contains("final", na=False)]
    sells = results[results["Action"].str.contains("SELL", na=False)]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=results["Date"], y=results["Close"], mode="lines",
                             name="Price", line=dict(color="#e2e8f0", width=1.5)))
    fig.add_trace(go.Scatter(x=buys["Date"],  y=buys["Close"],  mode="markers", name="BUY",
                             marker=dict(color="#10b981", size=9, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"], mode="markers", name="SELL",
                             marker=dict(color="#ef4444", size=9, symbol="triangle-down")))
    fig.update_layout(title=f"{ticker} — Trade Signals", template="plotly_dark",
                      paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                      xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,t=40,b=0), height=320)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  KPI PANELS
# ══════════════════════════════════════════════════════════════════════════════

def _bah_return(results, initial_cash):
    return ((initial_cash / float(results["Close"].iloc[0]) * float(results["Close"].iloc[-1]))
            - initial_cash) / initial_cash * 100


def display_kpis_simple(results, initial_cash):
    final_value = float(results["Portfolio Value"].iloc[-1])
    pct_return  = (final_value - initial_cash) / initial_cash * 100
    bah         = _bah_return(results, initial_cash)
    n_buys      = results["Action"].str.contains("BUY",  na=False).sum()
    n_sells     = results["Action"].str.contains("SELL", na=False).sum()

    cols  = st.columns(5)
    items = [
        (f"${final_value:,.0f}", "#63b3ed", "Final Value"),
        (f"{'+' if pct_return>=0 else ''}{pct_return:.1f}%",
         "#10b981" if pct_return >= 0 else "#ef4444", "ML Return"),
        (f"{'+' if bah>=0 else ''}{bah:.1f}%",
         "#10b981" if bah >= 0 else "#ef4444", "Buy & Hold"),
        (str(n_buys),  "#10b981", "BUY signals"),
        (str(n_sells), "#ef4444", "SELL signals"),
    ]
    for col, (val, color, lbl) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div>'
                     f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    diff = pct_return - bah
    if diff > 0:   st.success(f"✅ ML Strategy **outperformed** Buy & Hold by **{diff:.1f}%**")
    elif diff < 0: st.warning(f"⚠️ ML Strategy **underperformed** Buy & Hold by **{abs(diff):.1f}%**")
    else:          st.info("ML Strategy matched Buy & Hold performance")


def display_kpis_advanced(results, initial_cash):
    m   = compute_risk_metrics(results, initial_cash)
    bah = _bah_return(results, initial_cash)
    fv  = float(results["Portfolio Value"].iloc[-1])

    cols  = st.columns(6)
    items = [
        (f"${fv:,.0f}", "#63b3ed", "Final Value"),
        (f"{'+' if m['total_return']>=0 else ''}{m['total_return']:.1f}%",
         "#10b981" if m["total_return"] >= 0 else "#ef4444", "ML Return"),
        (f"{'+' if bah>=0 else ''}{bah:.1f}%",
         "#10b981" if bah >= 0 else "#ef4444", "Buy & Hold"),
        (f"{m['sharpe']:.2f}",
         "#10b981" if m["sharpe"] >= 1 else "#f97316" if m["sharpe"] >= 0 else "#ef4444",
         "Sharpe Ratio"),
        (f"{m['max_dd']:.1f}%", "#ef4444", "Max Drawdown"),
        (f"{m['win_rate']:.0f}%  ({m['n_trades']})",
         "#10b981" if m["win_rate"] >= 50 else "#ef4444", "Win Rate (trades)"),
    ]
    for col, (val, color, lbl) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div>'
                     f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    diff = m["total_return"] - bah
    if diff > 0:   st.success(f"✅ ML Strategy **outperformed** Buy & Hold by **{diff:.1f}%**")
    elif diff < 0: st.warning(f"⚠️ ML Strategy **underperformed** Buy & Hold by **{abs(diff):.1f}%**")
    else:          st.info("ML Strategy matched Buy & Hold performance")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_results(results, ticker, initial_cash, strategy_name, is_advanced, show_pred_dist=False, prefix=""):
    results = results.copy()
    results["Date"] = pd.to_datetime(results["Date"])
    results = results[results["Date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

    if results.empty:
        st.error("No results in selected range after ETL. Try wider dates.")
        return

    st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
    if is_advanced:
        display_kpis_advanced(results, initial_cash)
    else:
        display_kpis_simple(results, initial_cash)

    st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
    st.plotly_chart(portfolio_chart(results, ticker, initial_cash, strategy_name), use_container_width=True, key=f"{prefix}_portfolio")

    if is_advanced:
        st.markdown('<p class="sec">Drawdown</p>', unsafe_allow_html=True)
        st.plotly_chart(drawdown_chart(results, ticker), use_container_width=True, key=f"{prefix}_drawdown")

    st.markdown('<p class="sec">Trade Signals</p>', unsafe_allow_html=True)
    st.plotly_chart(signals_chart(results, ticker), use_container_width=True, key=f"{prefix}_signals")

    if show_pred_dist and "Pred Name" in results.columns:
        st.markdown('<p class="sec">Prediction Distribution</p>', unsafe_allow_html=True)
        pred_counts  = results["Pred Name"].value_counts()
        class_colors = {"Big Fall": "#ef4444", "Small Fall": "#f97316",
                        "Small Rise": "#22c55e", "Big Rise": "#10b981"}
        pcols = st.columns(4)
        for col, name in zip(pcols, ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]):
            cnt = pred_counts.get(name, 0)
            pct = cnt / len(results) * 100
            col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{class_colors[name]};">'
                         f'{cnt}</div><div class="kpi-lbl">{name} ({pct:.1f}%)</div></div>',
                         unsafe_allow_html=True)

    st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
    if is_advanced:
        log_cols = ["Date", "Close", "Exec Price", "Confidence", "Action",
                    "Shares", "Cash", "Portfolio Value", "Trade PnL"]
    else:
        log_cols = ["Date", "Close", "Action", "Shares", "Cash", "Portfolio Value"]

    if show_pred_dist and "Pred Name" in results.columns:
        log_cols.insert(3 if is_advanced else 2, "Pred Name")

    trade_log = results[results["Action"] != "HOLD"][
    [c for c in log_cols if c in results.columns]
    ].copy()

    if trade_log.empty:
        st.markdown(
            '<div class="warn">⚠️ No trades were executed. '
            'In Advanced mode, try lowering the <strong>confidence threshold</strong> '
            'in the sidebar — the model may never have exceeded the current threshold '
            'during this period.</div>',
            unsafe_allow_html=True
        )
    else:
        trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(trade_log, use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔁 Backtesting")
st.markdown('<p style="color:#64748b;margin-top:-8px;">Simulate how the model would have performed on historical data.</p>', unsafe_allow_html=True)

if is_advanced:
    st.markdown('<span class="mode-badge-advanced">🟩 Advanced Mode</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="mode-badge-simple">🟦 Simple Mode</span>', unsafe_allow_html=True)

if not run_btn:
    if is_advanced:
        st.markdown('''
        <div class="info">
            <strong>Advanced mode active:</strong><br>
            • Trades execute at <strong>next-day open</strong> (not today's close)<br>
            • Position size scales as a <strong>% of current portfolio</strong><br>
            • Only trades when <strong>model confidence ≥ threshold</strong><br>
            • Commission + slippage deducted on every trade<br>
            • Full risk metrics: <strong>Sharpe ratio, max drawdown, win rate</strong>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="info">
            <strong>Simple mode active:</strong><br>
            • <strong>Binary:</strong> Rise → BUY 1 share | Fall → SELL all shares<br>
            • <strong>Multi-class:</strong> Big Fall → SELL ALL | Small Fall → HOLD | Small Rise → BUY 1 | Big Rise → BUY 2<br>
            • Trades at today's close — no confidence filter, no transaction costs
        </div>
        ''', unsafe_allow_html=True)
    st.info("👈 Configure settings and click **Run Backtest**.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

api_key = get_api_key()
fetch_start = (start_date - timedelta(days=60)).strftime("%Y-%m-%d")
fetch_end   = end_date.strftime("%Y-%m-%d")

with st.spinner(f"Fetching {ticker} historical data…"):
    try:
        client = PySimFin(api_key=api_key)
        df_raw = client.get_share_prices(ticker, start=fetch_start, end=fetch_end)
    except SimFinRateLimitError:
        st.error("Rate limit hit. Wait a moment and retry."); st.stop()
    except SimFinNotFoundError:
        st.error(f"Ticker '{ticker}' not found."); st.stop()
    except SimFinAPIError as e:
        st.error(f"SimFin API error: {e}"); st.stop()

if df_raw.empty:
    st.error("No data returned."); st.stop()

df_raw.columns = df_raw.columns.str.lower()

if is_advanced and "open" not in df_raw.columns:
    st.markdown('''<div class="warn">⚠️ No <code>open</code> column in SimFin data —
    falling back to <code>close</code> as execution price. Results will be slightly optimistic.</div>''',
    unsafe_allow_html=True)
    df_raw["open"] = df_raw["close"]

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_binary, tab_multi = st.tabs(["📊 Binary Strategy", "📊 Multi-Class Strategy"])

with tab_binary:
    if is_advanced:
        st.markdown(f'''<div class="info">
            <strong>Advanced Binary:</strong> Rise (conf ≥ {confidence_threshold:.0%}) → BUY {alloc_pct}% of cash at next-day open |
            Fall (conf ≥ {confidence_threshold:.0%}) → SELL ALL at next-day open
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="info">
            <strong>Simple Binary:</strong> Rise → BUY 1 share at today's close | Fall → SELL ALL at today's close
        </div>''', unsafe_allow_html=True)
    try:
        pipeline_binary, features_binary = load_model(ticker, "binary")
        with st.spinner("Running binary simulation…"):
            if is_advanced:
                results = run_advanced_binary(df_raw.copy(), pipeline_binary, features_binary,
                                              float(initial_cash), confidence_threshold,
                                              alloc_pct, commission, slippage_pct)
                label = f"Binary Advanced (conf≥{confidence_threshold:.0%}, {alloc_pct}% sizing)"
            else:
                results = run_simple_binary(df_raw.copy(), pipeline_binary, features_binary, float(initial_cash))
                label   = "Binary Simple (Buy 1 / Sell All)"
        render_results(results, ticker, initial_cash, label, is_advanced, show_pred_dist=False, prefix="binary")
    except FileNotFoundError as e: st.error(str(e))
    except ValueError as e:        st.error(str(e))

with tab_multi:
    if is_advanced:
        st.markdown(f'''<div class="info">
            <strong>Advanced Multi-Class:</strong> Big Fall (conf ≥ {confidence_threshold:.0%}) → SELL ALL |
            Small Fall → HOLD | Small Rise (conf ≥ {confidence_threshold:.0%}) → BUY {alloc_pct}% |
            Big Rise (conf ≥ {confidence_threshold:.0%}) → BUY {min(alloc_pct*2,100)}%
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="info">
            <strong>Simple Multi-Class:</strong> Big Fall → SELL ALL | Small Fall → HOLD |
            Small Rise → BUY 1 share | Big Rise → BUY 2 shares
        </div>''', unsafe_allow_html=True)
    try:
        pipeline_multi, features_multi = load_model(ticker, "multi")
        with st.spinner("Running multi-class simulation…"):
            if is_advanced:
                results = run_advanced_multi(df_raw.copy(), pipeline_multi, features_multi,
                                             float(initial_cash), confidence_threshold,
                                             alloc_pct, commission, slippage_pct)
                label = f"Multi-Class Advanced (conf≥{confidence_threshold:.0%}, {alloc_pct}% sizing)"
            else:
                results = run_simple_multi(df_raw.copy(), pipeline_multi, features_multi, float(initial_cash))
                label   = "Multi-Class Simple (Buy 1-2 / Sell All)"
        render_results(results, ticker, initial_cash, label, is_advanced, show_pred_dist=True, prefix="multi")
    except FileNotFoundError as e: st.error(str(e))
    except ValueError as e:        st.error(str(e))
