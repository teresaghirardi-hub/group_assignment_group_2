"""
app.py — Home Page
==================
Streamlit entry point. Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AlgoTrader | Automated Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* Lighter sidebar so text is readable */
section[data-testid="stSidebar"] {
    background: #1a2744;
    border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] a:hover { color: #63b3ed !important; }

/* Sidebar selectbox — match sidebar bg, light text */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1a2744 !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #e2e8f0 !important; }
/* Dropdown popup list */
[data-baseweb="menu"] { background-color: #1e3258 !important; }
[data-baseweb="menu"] li { color: #e2e8f0 !important; background-color: #1e3258 !important; }
[data-baseweb="menu"] li:hover { background-color: #2a4070 !important; }
[data-baseweb="menu"] [aria-selected="true"] { background-color: rgba(37,99,235,0.35) !important; }

/* Hide auto-generated Streamlit nav (we use our own) */
[data-testid="stSidebarNavItems"] { display: none; }

.hero {
    background: linear-gradient(135deg, #0f1729 0%, #162040 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px; padding: 48px 40px; margin-bottom: 24px;
}
.hero-title { font-size: 2.8rem; font-weight: 700; color: #fff; margin: 0 0 8px 0; letter-spacing: -1px; }
.hero-title span { background: linear-gradient(90deg,#63b3ed,#4fd1c7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { font-size: 1.05rem; color: #94a3b8; max-width: 580px; margin: 0 0 20px 0; }
.badge { display: inline-block; background: rgba(66,153,225,0.12); border: 1px solid rgba(66,153,225,0.3); color: #63b3ed; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; margin-right: 6px; font-family: 'JetBrains Mono', monospace; }

.card { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 22px; height: 100%; }
.card-icon { font-size: 1.8rem; margin-bottom: 10px; }
.card-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
.card-desc { font-size: 0.88rem; color: #64748b; line-height: 1.6; }

.chip { background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.2); border-radius: 8px; padding: 12px 16px; text-align: center; }
.chip-val { font-size: 1.5rem; font-weight: 700; color: #63b3ed; font-family: 'JetBrains Mono', monospace; }
.chip-lbl { font-size: 0.75rem; color: #64748b; margin-top: 4px; }

.step { display: flex; align-items: flex-start; gap: 14px; padding: 14px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.step-num { background: linear-gradient(135deg,#2563eb,#0ea5e9); color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.82rem; flex-shrink: 0; font-family: 'JetBrains Mono', monospace; }
.step-title { color: #e2e8f0; font-weight: 600; display: block; margin-bottom: 3px; }
.step-desc { color: #64748b; font-size: 0.87rem; }

.team-card { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 18px; text-align: center; }
.team-av { width: 48px; height: 48px; background: linear-gradient(135deg,#2563eb,#0ea5e9); border-radius: 50%; margin: 0 auto 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; }
.team-name { font-weight: 600; color: #e2e8f0; font-size: 0.92rem; }
.team-role { color: #64748b; font-size: 0.8rem; margin-top: 3px; }

.sec-hdr { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; margin: 28px 0 14px 0; padding-bottom: 6px; border-bottom: 1px solid rgba(255,255,255,0.06); }

.model-status { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 14px 18px; margin-top: 8px; }
.model-ok { color: #10b981; }
.model-miss { color: #ef4444; }

.instr-box { background: #0f1e35; border: 1px solid rgba(99,179,237,0.2); border-radius: 12px; padding: 20px 24px; margin-bottom: 8px; }
.instr-box h4 { color: #63b3ed; font-size: 0.95rem; font-weight: 600; margin: 0 0 8px 0; }
.instr-box ul { color: #94a3b8; font-size: 0.87rem; line-height: 1.8; margin: 0; padding-left: 18px; }
.instr-box .tag { display: inline-block; background: rgba(99,179,237,0.12); border-radius: 4px; padding: 1px 7px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #63b3ed; }
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
    st.markdown('<p style="font-size:0.75rem;color:#94a3b8;">Powered by SimFin · Streamlit</p>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Algo<span>Trader</span></div>
    <p class="hero-sub">An automated daily trading system using machine learning to predict
    next-day stock price movements for 5 US companies.</p>
    <span class="badge">Python</span>
    <span class="badge">Scikit-learn</span>
    <span class="badge">XGBoost</span>
    <span class="badge">SimFin API</span>
    <span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ── Stats ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip([c1,c2,c3,c4],
    ["5", "24", "2", "Live"],
    ["Companies tracked", "Technical features", "Model types", "SimFin data feed"]):
    col.markdown(f'<div class="chip"><div class="chip-val">{val}</div><div class="chip-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── App Overview ───────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Overview</p>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
for col, icon, title, desc in zip(
    [fc1, fc2, fc3],
    ["🔬", "⚡", "📊"],
    ["ML Predictions", "Real-Time Data", "Strategy Backtesting"],
    [
        "Binary & multi-class models trained on 24 technical indicators predict next-day price movements with confidence scores.",
        "Connects live to SimFin API and applies the identical ETL pipeline used during training — no data leakage.",
        "Simulate BUY/SELL/HOLD strategies on historical data and review performance metrics, drawdowns, and equity curves.",
    ],
):
    col.markdown(f'<div class="card"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-desc">{desc}</div></div>', unsafe_allow_html=True)

# ── How to use ─────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">How to use this app</p>', unsafe_allow_html=True)
gi1, gi2, gi3 = st.columns(3)

with gi1:
    st.markdown("""
    <div class="instr-box">
        <h4>⚡ Go Live — get a prediction</h4>
        <ul>
            <li>Click <strong>Go Live</strong> in the left sidebar.</li>
            <li>Select a <strong>ticker</strong> (AMZN, AAPL, MSFT, GOOG, TSLA).</li>
            <li>Adjust <strong>days of history</strong> shown on the chart.</li>
            <li>Click <strong>Get Prediction</strong>.</li>
            <li>The app fetches live data, runs the ETL pipeline, and shows you a <strong>BUY / SELL / HOLD</strong> signal with class probabilities for both the binary and multi-class model.</li>
            <li>Expand <span class="tag">Features Used</span> to inspect the computed indicators and <span class="tag">Model Statistics</span> for classifier details.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with gi2:
    st.markdown("""
    <div class="instr-box">
        <h4>🔁 Backtesting — simulate a strategy</h4>
        <ul>
            <li>Click <strong>Backtesting</strong> in the left sidebar.</li>
            <li>Choose a ticker, date range, and strategy mode (<strong>Simple</strong> or <strong>Advanced</strong>).</li>
            <li><strong>Simple</strong> mode uses the binary model signal directly.</li>
            <li><strong>Advanced</strong> mode adds a confidence threshold, position sizing, and transaction costs.</li>
            <li>Review the equity curve, Sharpe ratio, max drawdown, and trade log.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with gi3:
    st.markdown("""
    <div class="instr-box">
        <h4>🛠️ Setup requirements</h4>
        <ul>
            <li>A valid <strong>SimFin API key</strong> must be set before using Go Live.</li>
            <li>Add it to a <span class="tag">.env</span> file at the project root:<br><span class="tag">SIMFIN_API_KEY=your_key</span></li>
            <li>Or on Streamlit Cloud, add it under <em>App settings → Secrets</em>.</li>
            <li>Models must be trained first. Run <span class="tag">ml_model_binary.ipynb</span> and <span class="tag">ml_model_multiclass.ipynb</span> to generate <span class="tag">.joblib</span> files in <span class="tag">models/</span>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">How it works</p>', unsafe_allow_html=True)
left, right = st.columns([1, 1])

with left:
    for num, title, desc in [
        ("01", "ETL Pipeline", "SimFin bulk data enriched with 24 technical indicators (SMAs, MACD, Bollinger Bands, RSI, etc). Creates binary and multi-class targets."),
        ("02", "Model Training", "4 models compared per ticker: Logistic Regression, Random Forest, Gradient Boosting, XGBoost. Best model selected by AUC-ROC (binary) or F1 Macro (multi)."),
        ("03", "Model Export", "Best model pipeline saved as .joblib. Feature list saved as .txt. Both loaded by the web app at runtime."),
        ("04", "Live Prediction", "Go Live page fetches fresh SimFin data, applies identical ETL, shows prediction with confidence scores."),
    ]:
        st.markdown(f'<div class="step"><div class="step-num">{num}</div><div><span class="step-title">{title}</span><span class="step-desc">{desc}</span></div></div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card" style="height:auto;">
        <div class="card-title" style="margin-bottom:14px;">📁 Project Structure</div>
        <pre style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#94a3b8;margin:0;line-height:1.8;">
trading-app/
├── app.py                     Home page
├── etl.py                     Shared ETL & model utilities
├── pysimfin.py                SimFin API wrapper
├── pages/
│   ├── go_live.py             Live predictions
│   └── backtesting.py         Strategy backtesting
├── notebooks/
│   ├── etl.ipynb              ETL pipeline
│   ├── ml_model_binary.ipynb  Binary classification
│   └── ml_model_multiclass.ipynb  Multi-class
├── models/
│   ├── model_AMZN_binary.joblib
│   ├── model_AMZN_multi.joblib
│   ├── features_AMZN_binary.txt
│   └── ...
├── data/
│   ├── raw/                   SimFin bulk downloads
│   └── processed/             ETL output CSVs
├── requirements.txt
└── README.md</pre>
    </div>
    """, unsafe_allow_html=True)

# ── Model Status Check ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Model Status</p>', unsafe_allow_html=True)

TICKERS = ["AMZN", "AAPL", "MSFT", "GOOG", "TSLA"]
MODELS_DIR = Path(__file__).parent / "models"

def _render_model_row(model_suffix: str) -> int:
    """Render one row of model status cards; return count of found models."""
    cols = st.columns(5)
    results = []
    for col, t in zip(cols, TICKERS):
        mp = MODELS_DIR / f"model_{t}_{model_suffix}.joblib"
        fp = MODELS_DIR / f"features_{t}_{model_suffix}.txt"
        ok = mp.exists() and fp.exists()
        results.append(ok)
        if ok:
            with open(fp) as fh:
                n = len([ln for ln in fh if ln.strip()])
            col.markdown(f'<div class="model-status"><div class="model-ok">✓ {t}</div>'
                         f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{n} features</div></div>',
                         unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="model-status"><div class="model-miss">✗ {t}</div>'
                         f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Not found</div></div>',
                         unsafe_allow_html=True)
    return sum(results)

st.markdown("**Binary Models** (Rise / Fall)")
binary_ok = _render_model_row("binary")

st.markdown("**Multi-Class Models** (Big Fall / Small Fall / Small Rise / Big Rise)")
multi_ok = _render_model_row("multi")

if binary_ok < 5 or multi_ok < 5:
    st.warning("⚠️ Some models missing. Run `notebooks/ml_model_binary.ipynb` and `notebooks/ml_model_multiclass.ipynb` first.")

# ── Covered Tickers ────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Covered companies</p>', unsafe_allow_html=True)
cols = st.columns(5)
for col, (ticker, name) in zip(cols, [("AMZN","Amazon"),("AAPL","Apple"),("MSFT","Microsoft"),("GOOG","Alphabet"),("TSLA","Tesla")]):
    col.markdown(f'<div class="chip"><div class="chip-val" style="font-size:1rem;">{ticker}</div><div class="chip-lbl">{name}</div></div>', unsafe_allow_html=True)

# ── Team ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">The team</p>', unsafe_allow_html=True)
team = [("👤","Team Member 1","ML & ETL"),("👤","Team Member 2","API Wrapper"),
        ("👤","Team Member 3","Streamlit App"),("👤","Team Member 4","Deployment")]
tcols = st.columns(4)
for col, (av, name, role) in zip(tcols, team):
    col.markdown(f'<div class="team-card"><div class="team-av">{av}</div><div class="team-name">{name}</div><div class="team-role">{role}</div></div>', unsafe_allow_html=True)
