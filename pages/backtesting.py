"""
pages/backtesting.py — Backtesting Page (Improved)
====================================================
Key improvements over v1:
  1. Realistic execution price  — trades execute at next day's OPEN, not
     today's close (you can't trade after the close that generated the signal).
  2. Transaction costs          — configurable commission + slippage per trade.
  3. Confidence threshold       — only trade when model probability >= threshold,
     reducing low-conviction noise trades.
  4. Position sizing            — invest a fixed % of current portfolio per signal
     instead of always buying exactly 1 share.
  5. Full risk metrics          — Sharpe ratio, max drawdown, win rate, # trades.
  6. Vectorised predictions     — batch predict instead of row-by-row loop (~100x faster).

Binary Strategy:
  Prediction = 1 (Rise), confidence >= threshold → BUY (size = alloc_pct of cash)
  Prediction = 0 (Fall), confidence >= threshold → SELL all shares

Multi-Class Strategy:
  0 (Big Fall),   confidence >= threshold → SELL ALL
  1 (Small Fall)                          → HOLD
  2 (Small Rise), confidence >= threshold → BUY  (1× alloc_pct)
  3 (Big Rise),   confidence >= threshold → BUY  (2× alloc_pct, capped at cash)
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
.warn { background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.2); border-radius: 8px; padding: 10px 14px; font-size: 0.87rem; color: #fbbf24; }
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
        end_date = st.date_input("To", value=date.today() - timedelta(days=1))

    initial_cash = st.number_input("Starting Capital ($)", value=10_000, step=1_000)

    st.markdown("#### Strategy Settings")

    confidence_threshold = st.slider(
        "Min. confidence to trade",
        min_value=0.50, max_value=0.90, value=0.55, step=0.05,
        help="Only trade when the model's predicted probability exceeds this value. "
             "Higher = fewer but more confident trades.",
    )
    alloc_pct = st.slider(
        "Capital per trade (%)",
        min_value=5, max_value=100, value=20, step=5,
        help="Fraction of current cash to deploy on each BUY signal. "
             "Keeps position size proportional to portfolio value.",
    )
    commission = st.number_input(
        "Commission per trade ($)",
        min_value=0.0, max_value=50.0, value=1.0, step=0.5,
        help="Fixed cost charged on every buy or sell execution.",
    )
    slippage_pct = st.slider(
        "Slippage (%)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Execution price is worse than the quoted open by this fraction. "
             "Models the bid-ask spread and market impact.",
    )

    st.markdown("---")
    run_btn = st.button("🔁 Run Backtest", use_container_width=True, type="primary")


# ── Helper: apply transaction cost to an execution price ──────────────────────

def apply_costs(price: float, side: str, commission: float, slippage_pct: float) -> tuple[float, float]:
    """
    Return (effective_price, total_cost_overhead).

    side = 'buy'  → price is bumped UP   by slippage (you pay more)
    side = 'sell' → price is bumped DOWN by slippage (you receive less)
    """
    slip = price * slippage_pct / 100
    if side == "buy":
        effective = price + slip
    else:
        effective = price - slip
    return effective, commission


# ── Risk metrics ───────────────────────────────────────────────────────────────

def compute_risk_metrics(results: pd.DataFrame, initial_cash: float, rf_annual: float = 0.05) -> dict:
    """
    Compute key strategy risk/return metrics.

    rf_annual : risk-free annual rate (default 5 % — approximate US T-bill 2024)
    """
    pv = results["Portfolio Value"].values
    returns = np.diff(pv) / pv[:-1]            # daily portfolio returns

    total_return = (pv[-1] - initial_cash) / initial_cash * 100

    # Annualised return (assuming 252 trading days)
    n_days = len(pv)
    ann_return = ((pv[-1] / initial_cash) ** (252 / max(n_days, 1)) - 1) * 100

    # Sharpe ratio (annualised)
    rf_daily = rf_annual / 252
    excess   = returns - rf_daily
    sharpe   = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Max drawdown
    peak      = np.maximum.accumulate(pv)
    drawdown  = (pv - peak) / peak
    max_dd    = drawdown.min() * 100          # negative number

    # Win rate on closed trades
    trades = results[results["Action"] != "HOLD"].copy()
    trade_pnl = trades["Trade PnL"].dropna()
    win_rate  = (trade_pnl > 0).mean() * 100 if len(trade_pnl) > 0 else 0.0
    n_trades  = len(trade_pnl)

    return {
        "total_return": total_return,
        "ann_return":   ann_return,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
    }


# ── Backtest: Binary ───────────────────────────────────────────────────────────

def run_backtest_binary(
    df: pd.DataFrame,
    pipeline,
    feature_cols: list,
    initial_cash: float,
    confidence_threshold: float,
    alloc_pct: float,
    commission: float,
    slippage_pct: float,
) -> pd.DataFrame:
    """
    Binary strategy with realistic execution:
      - Signal generated from today's features (using today's close)
      - Trade executes at NEXT DAY's open (adjusted for slippage)
      - Position size = alloc_pct % of current cash
      - Trade only fires if model confidence >= confidence_threshold
    """
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)

    col_map   = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    open_col  = col_map.get("open",  "open")
    date_col  = col_map.get("date",  "date")

    # ── Vectorised batch prediction ──────────────────────────────────────────
    X_all   = df[feature_cols].values
    preds   = pipeline.predict(X_all)
    probas  = pipeline.predict_proba(X_all)          # shape (n, 2)
    conf    = probas[np.arange(len(preds)), preds]   # confidence of chosen class

    results = []
    cash, shares = initial_cash, 0.0
    avg_cost = 0.0   # average cost basis for current position

    for i in range(len(df) - 1):   # stop one row early — need next-day open
        row        = df.iloc[i]
        next_row   = df.iloc[i + 1]
        pred       = int(preds[i])
        confidence = float(conf[i])
        exec_price = float(next_row[open_col])   # NEXT DAY open
        action     = "HOLD"
        trade_pnl  = np.nan

        # ── BUY signal ──
        if pred == 1 and confidence >= confidence_threshold and cash > 0:
            buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
            spend      = cash * (alloc_pct / 100) - cost
            if spend > 0:
                new_shares = spend / buy_price
                avg_cost   = (avg_cost * shares + buy_price * new_shares) / (shares + new_shares)
                shares    += new_shares
                cash      -= (spend + cost)
                action     = "BUY"

        # ── SELL signal ──
        elif pred == 0 and confidence >= confidence_threshold and shares > 0:
            sell_price, cost = apply_costs(exec_price, "sell", commission, slippage_pct)
            proceeds   = shares * sell_price - cost
            trade_pnl  = proceeds - shares * avg_cost
            cash      += proceeds
            action     = f"SELL {shares:.4f}"
            shares     = 0.0
            avg_cost   = 0.0

        results.append({
            "Date":            row[date_col],
            "Close":           float(row[close_col]),
            "Exec Price":      exec_price,
            "Prediction":      pred,
            "Confidence":      round(confidence * 100, 1),
            "Action":          action,
            "Shares":          round(shares, 4),
            "Cash":            round(cash, 2),
            "Portfolio Value": round(cash + shares * float(row[close_col]), 2),
            "Trade PnL":       round(trade_pnl, 2) if not np.isnan(trade_pnl) else np.nan,
        })

    # Final liquidation at last available close
    if shares > 0:
        last_row   = df.iloc[-2]   # last row with a next-day open
        last_price = float(last_row[close_col])
        sell_price, cost = apply_costs(last_price, "sell", commission, slippage_pct)
        proceeds  = shares * sell_price - cost
        trade_pnl = proceeds - shares * avg_cost
        cash     += proceeds
        results[-1].update({
            "Action":          f"SELL {shares:.4f} (final)",
            "Shares":          0.0,
            "Cash":            round(cash, 2),
            "Portfolio Value": round(cash, 2),
            "Trade PnL":       round(trade_pnl, 2),
        })

    return pd.DataFrame(results)


# ── Backtest: Multi-Class ──────────────────────────────────────────────────────

def run_backtest_multi(
    df: pd.DataFrame,
    pipeline,
    feature_cols: list,
    initial_cash: float,
    confidence_threshold: float,
    alloc_pct: float,
    commission: float,
    slippage_pct: float,
) -> pd.DataFrame:
    """
    Multi-class strategy with realistic execution:
      0 (Big Fall),   conf >= threshold → SELL ALL
      1 (Small Fall)                    → HOLD
      2 (Small Rise), conf >= threshold → BUY (alloc_pct % of cash)
      3 (Big Rise),   conf >= threshold → BUY (2 × alloc_pct % of cash, capped)
    """
    df = add_technical_features(df).dropna(subset=feature_cols).reset_index(drop=True)

    col_map    = {c.lower(): c for c in df.columns}
    close_col  = col_map.get("close", "close")
    open_col   = col_map.get("open",  "open")
    date_col   = col_map.get("date",  "date")
    CLASS_NAMES = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]

    # ── Vectorised batch prediction ──────────────────────────────────────────
    X_all  = df[feature_cols].values
    preds  = pipeline.predict(X_all)
    probas = pipeline.predict_proba(X_all)
    conf   = probas[np.arange(len(preds)), preds]

    results = []
    cash, shares = initial_cash, 0.0
    avg_cost = 0.0

    for i in range(len(df) - 1):
        row        = df.iloc[i]
        next_row   = df.iloc[i + 1]
        pred       = int(preds[i])
        confidence = float(conf[i])
        exec_price = float(next_row[open_col])
        action     = "HOLD"
        trade_pnl  = np.nan

        if confidence >= confidence_threshold:

            # Big Fall → SELL ALL
            if pred == 0 and shares > 0:
                sell_price, cost = apply_costs(exec_price, "sell", commission, slippage_pct)
                proceeds  = shares * sell_price - cost
                trade_pnl = proceeds - shares * avg_cost
                cash     += proceeds
                action    = f"SELL {shares:.4f}"
                shares    = 0.0
                avg_cost  = 0.0

            # Small Rise → BUY (1 unit of alloc_pct)
            elif pred == 2 and cash > 0:
                buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
                spend      = cash * (alloc_pct / 100) - cost
                if spend > 0:
                    new_shares = spend / buy_price
                    avg_cost   = (avg_cost * shares + buy_price * new_shares) / (shares + new_shares)
                    shares    += new_shares
                    cash      -= (spend + cost)
                    action     = "BUY 1×"

            # Big Rise → BUY (2 units of alloc_pct, capped at cash)
            elif pred == 3 and cash > 0:
                buy_price, cost = apply_costs(exec_price, "buy", commission, slippage_pct)
                spend      = min(cash * (alloc_pct / 100) * 2, cash) - cost
                if spend > 0:
                    new_shares = spend / buy_price
                    avg_cost   = (avg_cost * shares + buy_price * new_shares) / (shares + new_shares)
                    shares    += new_shares
                    cash      -= (spend + cost)
                    action     = "BUY 2×"

        results.append({
            "Date":            row[date_col],
            "Close":           float(row[close_col]),
            "Exec Price":      exec_price,
            "Prediction":      pred,
            "Pred Name":       CLASS_NAMES[pred],
            "Confidence":      round(confidence * 100, 1),
            "Action":          action,
            "Shares":          round(shares, 4),
            "Cash":            round(cash, 2),
            "Portfolio Value": round(cash + shares * float(row[close_col]), 2),
            "Trade PnL":       round(trade_pnl, 2) if not np.isnan(trade_pnl) else np.nan,
        })

    # Final liquidation
    if shares > 0:
        last_row   = df.iloc[-2]
        last_price = float(last_row[close_col])
        sell_price, cost = apply_costs(last_price, "sell", commission, slippage_pct)
        proceeds  = shares * sell_price - cost
        trade_pnl = proceeds - shares * avg_cost
        cash     += proceeds
        results[-1].update({
            "Action":          f"SELL {shares:.4f} (final)",
            "Shares":          0.0,
            "Cash":            round(cash, 2),
            "Portfolio Value": round(cash, 2),
            "Trade PnL":       round(trade_pnl, 2),
        })

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


def drawdown_chart(results: pd.DataFrame, ticker: str):
    pv   = results["Portfolio Value"].values
    peak = np.maximum.accumulate(pv)
    dd   = (pv - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["Date"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
    ))
    fig.update_layout(
        title=f"{ticker} — Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
        margin=dict(l=0, r=0, t=40, b=0), height=220,
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
        marker=dict(color="#10b981", size=8, symbol="triangle-up"),
    ))
    sells = results[results["Action"].str.contains("SELL", na=False)]
    fig.add_trace(go.Scatter(
        x=sells["Date"], y=sells["Close"],
        mode="markers", name="SELL",
        marker=dict(color="#ef4444", size=8, symbol="triangle-down"),
    ))
    fig.update_layout(
        title=f"{ticker} — Trade Signals",
        template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0), height=320,
    )
    return fig


# ── KPI display ────────────────────────────────────────────────────────────────

def display_kpis(results: pd.DataFrame, initial_cash: float):
    metrics = compute_risk_metrics(results, initial_cash)

    first_price = float(results["Close"].iloc[0])
    last_price  = float(results["Close"].iloc[-1])
    bah_return  = ((initial_cash / first_price * last_price) - initial_cash) / initial_cash * 100
    final_value = float(results["Portfolio Value"].iloc[-1])

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    items = [
        (f"${final_value:,.0f}",
         "#63b3ed",
         "Final Value"),
        (f"{'+' if metrics['total_return'] >= 0 else ''}{metrics['total_return']:.1f}%",
         "#10b981" if metrics["total_return"] >= 0 else "#ef4444",
         "ML Return"),
        (f"{'+' if bah_return >= 0 else ''}{bah_return:.1f}%",
         "#10b981" if bah_return >= 0 else "#ef4444",
         "Buy & Hold"),
        (f"{metrics['sharpe']:.2f}",
         "#10b981" if metrics["sharpe"] >= 1 else "#f97316" if metrics["sharpe"] >= 0 else "#ef4444",
         "Sharpe Ratio"),
        (f"{metrics['max_dd']:.1f}%",
         "#ef4444",
         "Max Drawdown"),
        (f"{metrics['win_rate']:.0f}%  ({metrics['n_trades']})",
         "#10b981" if metrics["win_rate"] >= 50 else "#ef4444",
         "Win Rate (trades)"),
    ]
    for col, (val, color, lbl) in zip([k1, k2, k3, k4, k5, k6], items):
        col.markdown(
            f'<div class="kpi"><div class="kpi-val" style="color:{color};">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    outperform = metrics["total_return"] - bah_return
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
        <strong>Improvements in this version:</strong><br>
        • Trades execute at <strong>next-day open</strong> (not today's close)<br>
        • <strong>Commission + slippage</strong> deducted on every trade<br>
        • Only trades when <strong>model confidence ≥ threshold</strong><br>
        • <strong>Position sizing</strong> scales with portfolio value<br>
        • Full risk metrics: <strong>Sharpe ratio, max drawdown, win rate</strong>
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

# ── Warn if Open column missing (fall back to Close) ──────────────────────────
if "open" not in df_raw.columns:
    st.markdown('''
    <div class="warn">
        ⚠️ No <code>open</code> column found in SimFin data — using <code>close</code> as
        execution price. Results will be slightly optimistic. Check your SimFin plan
        for intraday OHLC data access.
    </div>
    ''', unsafe_allow_html=True)
    df_raw["open"] = df_raw["close"]

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_binary, tab_multi = st.tabs(["📊 Binary Strategy", "📊 Multi-Class Strategy"])

# ── Binary Tab ─────────────────────────────────────────────────────────────────
with tab_binary:
    st.markdown(f'''
    <div class="info">
        <strong>Binary Strategy:</strong>
        Rise (conf ≥ {confidence_threshold:.0%}) → BUY {alloc_pct}% of cash at next-day open |
        Fall (conf ≥ {confidence_threshold:.0%}) → SELL all at next-day open
        | Commission: ${commission} + {slippage_pct}% slippage per trade
    </div>
    ''', unsafe_allow_html=True)
    try:
        pipeline_binary, features_binary = load_model(ticker, "binary")
        with st.spinner("Running binary simulation…"):
            results_binary = run_backtest_binary(
                df_raw.copy(), pipeline_binary, features_binary,
                float(initial_cash), confidence_threshold,
                alloc_pct, commission, slippage_pct,
            )
        results_binary["Date"] = pd.to_datetime(results_binary["Date"])
        results_binary = results_binary[
            results_binary["Date"] >= pd.to_datetime(start_date)
        ].reset_index(drop=True)

        if results_binary.empty:
            st.error("No results in selected range after ETL. Try wider dates.")
        else:
            st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
            display_kpis(results_binary, initial_cash)

            st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
            st.plotly_chart(portfolio_chart(results_binary, ticker, initial_cash, "Binary Strategy"), use_container_width=True)

            st.markdown('<p class="sec">Drawdown</p>', unsafe_allow_html=True)
            st.plotly_chart(drawdown_chart(results_binary, ticker), use_container_width=True)

            st.markdown('<p class="sec">Trade Signals</p>', unsafe_allow_html=True)
            st.plotly_chart(price_signals_chart(results_binary, ticker), use_container_width=True)

            st.markdown('<p class="sec">Trade Log</p>', unsafe_allow_html=True)
            trade_log = results_binary[results_binary["Action"] != "HOLD"][[
                "Date", "Close", "Exec Price", "Confidence", "Action",
                "Shares", "Cash", "Portfolio Value", "Trade PnL"
            ]].copy()
            trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(trade_log, use_container_width=True, height=280)

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))

# ── Multi-Class Tab ────────────────────────────────────────────────────────────
with tab_multi:
    st.markdown(f'''
    <div class="info">
        <strong>Multi-Class Strategy:</strong><br>
        • Big Fall (conf ≥ {confidence_threshold:.0%}) → SELL ALL at next-day open<br>
        • Small Fall → HOLD<br>
        • Small Rise (conf ≥ {confidence_threshold:.0%}) → BUY {alloc_pct}% of cash<br>
        • Big Rise (conf ≥ {confidence_threshold:.0%}) → BUY {min(alloc_pct * 2, 100)}% of cash
        | Commission: ${commission} + {slippage_pct}% slippage per trade
    </div>
    ''', unsafe_allow_html=True)
    try:
        pipeline_multi, features_multi = load_model(ticker, "multi")
        with st.spinner("Running multi-class simulation…"):
            results_multi = run_backtest_multi(
                df_raw.copy(), pipeline_multi, features_multi,
                float(initial_cash), confidence_threshold,
                alloc_pct, commission, slippage_pct,
            )
        results_multi["Date"] = pd.to_datetime(results_multi["Date"])
        results_multi = results_multi[
            results_multi["Date"] >= pd.to_datetime(start_date)
        ].reset_index(drop=True)

        if results_multi.empty:
            st.error("No results in selected range after ETL. Try wider dates.")
        else:
            st.markdown('<p class="sec">Summary</p>', unsafe_allow_html=True)
            display_kpis(results_multi, initial_cash)

            st.markdown('<p class="sec">Portfolio Performance</p>', unsafe_allow_html=True)
            st.plotly_chart(portfolio_chart(results_multi, ticker, initial_cash, "Multi-Class Strategy"), use_container_width=True)

            st.markdown('<p class="sec">Drawdown</p>', unsafe_allow_html=True)
            st.plotly_chart(drawdown_chart(results_multi, ticker), use_container_width=True)

            st.markdown('<p class="sec">Trade Signals</p>', unsafe_allow_html=True)
            st.plotly_chart(price_signals_chart(results_multi, ticker), use_container_width=True)

            st.markdown('<p class="sec">Prediction Distribution</p>', unsafe_allow_html=True)
            pred_counts  = results_multi["Pred Name"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            class_colors = {
                "Big Fall": "#ef4444", "Small Fall": "#f97316",
                "Small Rise": "#22c55e", "Big Rise": "#10b981",
            }
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
            trade_log = results_multi[results_multi["Action"] != "HOLD"][[
                "Date", "Close", "Exec Price", "Pred Name", "Confidence",
                "Action", "Shares", "Cash", "Portfolio Value", "Trade PnL"
            ]].copy()
            trade_log["Date"] = trade_log["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(trade_log, use_container_width=True, height=280)

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))

with st.expander("📋 Full daily results"):
    st.info("Select a tab above and switch to that tab's expander to see full daily data.")
