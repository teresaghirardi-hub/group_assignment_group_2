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
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid rgba(255,255,255,0.06); }

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
    st.markdown('<p style="font-size:0.75rem;color:#475569;">Powered by SimFin · Streamlit</p>', unsafe_allow_html=True)

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
    ["Companies", "Technical features", "Model types", "SimFin feed"]):
    col.markdown(f'<div class="chip"><div class="chip-val">{val}</div><div class="chip-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── Feature cards ──────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">What this system does</p>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
for col, icon, title, desc in zip(
    [fc1, fc2, fc3],
    ["🔬", "⚡", "📊"],
    ["ML Predictions", "Real-Time Data", "Trading Signals"],
    [
        "Binary & Multi-class models trained on 24 technical indicators predict price movements.",
        "Connects live to SimFin API. Applies the same ETL pipeline used in training.",
        "Translates predictions into BUY / SELL / HOLD signals with confidence levels.",
    ],
):
    col.markdown(f'<div class="card"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-desc">{desc}</div></div>', unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">How it works</p>', unsafe_allow_html=True)
left, right = st.columns([1, 1])

with left:
    for num, title, desc in [
        ("01", "ETL Pipeline", "SimFin bulk data enriched with 24 technical indicators (SMAs, MACD, Bollinger Bands, RSI, etc). Creates binary and multi-class targets."),
        ("02", "Model Training", "4 models compared per ticker: Logistic Regression, Random Forest, Gradient Boosting, XGBoost. Best model selected by AUC-ROC (binary) or F1 Macro (multi)."),
        ("03", "Model Export", "Best model pipeline saved as .joblib. Feature list saved as .txt. Both loaded by web app at runtime."),
        ("04", "Live Prediction", "Go Live page fetches fresh SimFin data, applies identical ETL, shows prediction with confidence."),
    ]:
        st.markdown(f'<div class="step"><div class="step-num">{num}</div><div><span class="step-title">{title}</span><span class="step-desc">{desc}</span></div></div>', unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card" style="height:auto;">
        <div class="card-title" style="margin-bottom:14px;">📁 Project Structure</div>
        <pre style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#94a3b8;margin:0;line-height:1.8;">
trading-app/
├── app.py                     Home page
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

# Check both binary and multi models
st.markdown("**Binary Models** (Rise/Fall)")
cols = st.columns(5)
binary_ok = 0
for col, ticker in zip(cols, TICKERS):
    model_path = MODELS_DIR / f"model_{ticker}_binary.joblib"
    features_path = MODELS_DIR / f"features_{ticker}_binary.txt"
    
    if model_path.exists() and features_path.exists():
        with open(features_path) as f:
            n_features = len([line for line in f if line.strip()])
        col.markdown(f'''
        <div class="model-status">
            <div class="model-ok">✓ {ticker}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{n_features} features</div>
        </div>
        ''', unsafe_allow_html=True)
        binary_ok += 1
    else:
        col.markdown(f'''
        <div class="model-status">
            <div class="model-miss">✗ {ticker}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Not found</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("**Multi-Class Models** (Big Fall / Small Fall / Small Rise / Big Rise)")
cols = st.columns(5)
multi_ok = 0
for col, ticker in zip(cols, TICKERS):
    model_path = MODELS_DIR / f"model_{ticker}_multi.joblib"
    features_path = MODELS_DIR / f"features_{ticker}_multi.txt"
    
    if model_path.exists() and features_path.exists():
        with open(features_path) as f:
            n_features = len([line for line in f if line.strip()])
        col.markdown(f'''
        <div class="model-status">
            <div class="model-ok">✓ {ticker}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{n_features} features</div>
        </div>
        ''', unsafe_allow_html=True)
        multi_ok += 1
    else:
        col.markdown(f'''
        <div class="model-status">
            <div class="model-miss">✗ {ticker}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Not found</div>
        </div>
        ''', unsafe_allow_html=True)

if binary_ok < 5 or multi_ok < 5:
    st.warning("⚠️ Some models missing. Run `notebooks/ml_model_binary.ipynb` and `notebooks/ml_model_multiclass.ipynb` first.")

# ── Tickers ────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Covered companies</p>', unsafe_allow_html=True)
cols = st.columns(5)
for col, (ticker, name) in zip(cols, [("AMZN","Amazon"),("AAPL","Apple"),("MSFT","Microsoft"),("GOOG","Alphabet"),("TSLA","Tesla")]):
    col.markdown(f'<div class="chip"><div class="chip-val" style="font-size:1rem;">{ticker}</div><div class="chip-lbl">{name}</div></div>', unsafe_allow_html=True)

# ── Team ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">The team</p>', unsafe_allow_html=True)
# TODO: replace with actual team member names and roles
team = [("👤","Team Member 1","ML & ETL"),("👤","Team Member 2","API Wrapper"),
        ("👤","Team Member 3","Streamlit App"),("👤","Team Member 4","Deployment")]
tcols = st.columns(4)
for col, (av, name, role) in zip(tcols, team):
    col.markdown(f'<div class="team-card"><div class="team-av">{av}</div><div class="team-name">{name}</div><div class="team-role">{role}</div></div>', unsafe_allow_html=True)