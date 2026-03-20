"""
pages/go_live.py — Go Live Page
================================
Fetches live data from SimFin, applies ETL transformations,
loads both binary and multi-class models, and shows predictions.
"""

import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pysimfin import PySimFin, SimFinAPIError, SimFinNotFoundError, SimFinRateLimitError
from etl import get_api_key, add_technical_features, prepare_for_prediction, load_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Go Live | AlgoTrader", page_icon="⚡", layout="wide")

# ── Class definitions ──────────────────────────────────────────────────────────
CLASS_NAMES_BINARY   = ["Fall", "Rise"]
CLASS_COLORS_BINARY  = ["#ef4444", "#10b981"]
CLASS_ICONS_BINARY   = ["📉", "📈"]
CLASS_ACTIONS_BINARY = ["SELL", "BUY"]

CLASS_NAMES_MULTI   = ["Big Fall", "Small Fall", "Small Rise", "Big Rise"]
CLASS_COLORS_MULTI  = ["#ef4444", "#f97316", "#22c55e", "#10b981"]
CLASS_ICONS_MULTI   = ["📉📉", "📉", "📈", "📈📈"]
CLASS_ACTIONS_MULTI = ["STRONG SELL", "SELL", "BUY", "STRONG BUY"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(255,255,255,0.06); }
.stat { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 16px; text-align: center; }
.stat-val { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #63b3ed; }
.stat-lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; }
.sec { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin: 20px 0 10px 0; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.info { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2); border-radius: 8px; padding: 10px 14px; font-size: 0.87rem; color: #94a3b8; }
.prob-bar { background: #1e293b; border-radius: 4px; height: 8px; margin-top: 4px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; }
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
    TICKERS      = ["AMZN", "AAPL", "MSFT", "GOOG", "TSLA"]
    ticker       = st.selectbox("Select Ticker", TICKERS)
    days_history = st.slider("Days of history to show", 30, 365, 90)
    st.markdown("---")
    run_btn = st.button("⚡ Get Prediction", use_container_width=True, type="primary")


# ── Candlestick chart ──────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, ticker: str):
    col_map  = {c.lower(): c for c in df.columns}
    date_col = col_map.get("date", "date")

    fig = go.Figure(data=[go.Candlestick(
        x=df[date_col],
        open=df[col_map.get("open",   "open")],
        high=df[col_map.get("high",   "high")],
        low =df[col_map.get("low",    "low")],
        close=df[col_map.get("close", "close")],
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
    )])
    fig.update_layout(
        title=f"{ticker} Price History",
        template="plotly_dark",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        xaxis_rangeslider_visible=False,
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", tickprefix="$"),
        margin=dict(l=0, r=0, t=40, b=0), height=350,
    )
    return fig


# ── Display prediction ─────────────────────────────────────────────────────────

def display_prediction(pipeline, X_latest, model_type: str, ticker: str):
    """Show prediction results with confidence and probability bars."""
    if model_type == "binary":
        class_names   = CLASS_NAMES_BINARY
        class_colors  = CLASS_COLORS_BINARY
        class_icons   = CLASS_ICONS_BINARY
        class_actions = CLASS_ACTIONS_BINARY
    else:
        class_names   = CLASS_NAMES_MULTI
        class_colors  = CLASS_COLORS_MULTI
        class_icons   = CLASS_ICONS_MULTI
        class_actions = CLASS_ACTIONS_MULTI

    prediction    = int(pipeline.predict(X_latest)[0])
    probabilities = pipeline.predict_proba(X_latest)[0]
    confidence    = probabilities[prediction] * 100

    pred_color = class_colors[prediction]
    pred_icon  = class_icons[prediction]
    pred_name  = class_names[prediction]
    action     = class_actions[prediction]

    # Prediction box
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, {pred_color}20, {pred_color}08);
                border: 2px solid {pred_color}; border-radius: 14px; padding: 28px; text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 6px;">{pred_icon}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {pred_color};">{pred_name.upper()}</div>
        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 6px;">Confidence: {confidence:.1f}%</div>
    </div>
    ''', unsafe_allow_html=True)

    # Trading signal
    st.markdown('<p class="sec">Trading Signal</p>', unsafe_allow_html=True)
    st.markdown(f'''
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px; padding: 18px; text-align: center;">
        <div style="font-size: 2.2rem; font-weight: 700; color: {pred_color};
                    font-family: 'JetBrains Mono', monospace;">{action}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Probability bars
    st.markdown('<p class="sec">Class Probabilities</p>', unsafe_allow_html=True)
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        pct   = prob * 100
        color = class_colors[i]
        st.markdown(f'''
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                <span>{name}</span>
                <span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-fill" style="background: {color}; width: {pct}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Go Live")
st.markdown('<p style="color:#64748b;margin-top:-8px;">Real-time predictions powered by live SimFin data.</p>', unsafe_allow_html=True)

if not run_btn:
    st.markdown('''
    <div class="info">
        👈 Select a ticker and click <strong>Get Prediction</strong>.
    </div>
    ''', unsafe_allow_html=True)
    st.stop()

# Resolve API key from secrets / environment (stops with error if missing)
api_key = get_api_key()

# Fetch data from SimFin
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

# ── Display chart ──────────────────────────────────────────────────────────────
col_map    = {c.lower(): c for c in df_raw.columns}
close_col  = col_map.get("close",  "close")
open_col   = col_map.get("open",   "open")
volume_col = col_map.get("volume", "volume")
date_col   = col_map.get("date",   "date")

df_chart = df_raw.copy()
df_chart[date_col] = pd.to_datetime(df_chart[date_col])
df_chart = df_chart.tail(days_history)

st.plotly_chart(candlestick_chart(df_chart, ticker), use_container_width=True)

# ── Price stats ────────────────────────────────────────────────────────────────
latest     = df_raw.iloc[-1]
prev       = df_raw.iloc[-2]
last_close = float(latest[close_col])
day_change = last_close - float(prev[close_col])
day_pct    = (day_change / float(prev[close_col])) * 100

st.markdown('<p class="sec">Latest Data</p>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
arrow = "▲" if day_change >= 0 else "▼"
c_col = "#10b981" if day_change >= 0 else "#ef4444"

for col, val, lbl in zip(
    [s1, s2, s3, s4],
    [
        f"${last_close:.2f}",
        f'<span style="color:{c_col}">{arrow} {day_pct:.2f}%</span>',
        f"${float(latest[open_col]):.2f}" if open_col in df_raw.columns else "—",
        f"{int(float(latest[volume_col])):,}" if volume_col in df_raw.columns else "—",
    ],
    ["Last Close", "Day Change", "Open", "Volume"],
):
    col.markdown(f'''
    <div class="stat">
        <div class="stat-val">{val}</div>
        <div class="stat-lbl">{lbl}</div>
    </div>
    ''', unsafe_allow_html=True)

# ── TABS: Binary vs Multi-Class ────────────────────────────────────────────────
st.markdown("---")
tab_binary, tab_multi = st.tabs(["📊 Binary Model (Rise/Fall)", "📊 Multi-Class Model (4 Classes)"])

# ── Binary Tab ─────────────────────────────────────────────────────────────────
with tab_binary:
    try:
        pipeline_binary, features_binary = load_model(ticker, "binary")
        X_live_binary = prepare_for_prediction(df_raw.copy(), features_binary)

        if X_live_binary.empty:
            st.error("Not enough data after ETL. Try a wider date range.")
        else:
            X_latest_binary = X_live_binary.iloc[[-1]]
            col1, col2 = st.columns([1, 1])
            with col1:
                display_prediction(pipeline_binary, X_latest_binary, "binary", ticker)
            with col2:
                st.markdown('<p class="sec">Features Used</p>', unsafe_allow_html=True)
                feat_df = X_latest_binary.T.rename(columns={X_latest_binary.index[0]: "Value"})
                feat_df["Value"] = feat_df["Value"].round(6)
                st.dataframe(feat_df, use_container_width=True, height=min(35 * len(feat_df) + 38, 350))
                pred_date = (date.today() + timedelta(days=1)).strftime("%A, %d %B %Y")
                st.markdown(f'''
                <div class="info" style="margin-top:10px;">
                    📅 Predicting for: <strong>{pred_date}</strong><br>
                    Based on data up to: <strong>{latest[date_col]}</strong>
                </div>
                ''', unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(f"ETL error: {e}")

# ── Multi-Class Tab ────────────────────────────────────────────────────────────
with tab_multi:
    try:
        pipeline_multi, features_multi = load_model(ticker, "multi")
        X_live_multi = prepare_for_prediction(df_raw.copy(), features_multi)

        if X_live_multi.empty:
            st.error("Not enough data after ETL. Try a wider date range.")
        else:
            X_latest_multi = X_live_multi.iloc[[-1]]
            col1, col2 = st.columns([1, 1])
            with col1:
                display_prediction(pipeline_multi, X_latest_multi, "multi", ticker)
            with col2:
                st.markdown('<p class="sec">Features Used</p>', unsafe_allow_html=True)
                feat_df = X_latest_multi.T.rename(columns={X_latest_multi.index[0]: "Value"})
                feat_df["Value"] = feat_df["Value"].round(6)
                st.dataframe(feat_df, use_container_width=True, height=min(35 * len(feat_df) + 38, 350))
                st.markdown('''
                <div class="info" style="margin-top:10px;">
                    <strong>Multi-Class Targets:</strong><br>
                    • Big Fall: return &lt; -1%<br>
                    • Small Fall: -1% ≤ return &lt; 0%<br>
                    • Small Rise: 0% ≤ return &lt; +1%<br>
                    • Big Rise: return ≥ +1%
                </div>
                ''', unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(f"ETL error: {e}")

# Raw data expander
with st.expander("📋 Raw price data"):
    st.dataframe(df_raw.tail(30), use_container_width=True)
