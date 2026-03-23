"""
app.py — Home Page
==================
Streamlit entry point. Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AUGUR Analytics | Automated Trading System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
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

.card { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 22px; height: 100%; min-height: 225px; box-sizing: border-box; display: flex; flex-direction: column; }
.card-icon { font-size: 1.8rem; margin-bottom: 10px; }
.card-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
.card-desc { font-size: 0.88rem; color: #64748b; line-height: 1.6; }
.company-card { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 22px; box-sizing: border-box; }
.how-card { background: #111827; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 22px; box-sizing: border-box; }

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

.instr-box { background: #0f1e35; border: 1px solid rgba(99,179,237,0.2); border-radius: 12px; padding: 20px 24px; margin-bottom: 8px; height: 100%; min-height: 300px; box-sizing: border-box; }
.instr-box h4 { color: #63b3ed; font-size: 0.95rem; font-weight: 600; margin: 0 0 8px 0; }
.instr-box ul { color: #94a3b8; font-size: 0.87rem; line-height: 1.8; margin: 0; padding-left: 18px; }
.instr-box .tag { display: inline-block; background: rgba(99,179,237,0.12); border-radius: 4px; padding: 1px 7px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #63b3ed; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    from team_photos import logo
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
    st.markdown('<p style="font-size:0.75rem;color:#94a3b8;">Powered by SimFin · Streamlit</p>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">AUGUR<span> Analytics</span></div>
    <p class="hero-sub">
        An end-to-end automated daily trading system built for the
        <strong style="color:#e2e8f0;">Python for Data Analytics II</strong> group project at IE University.
        Machine learning models trained on 24 technical indicators predict next-day stock price movements
        for 5 major US companies — combining a live prediction engine with a historical strategy backtester.
    </p>
    <span class="badge">Python</span>
    <span class="badge">Scikit-learn</span>
    <span class="badge">XGBoost</span>
    <span class="badge">SimFin API</span>
    <span class="badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

# ── Top-level stats ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in zip(
    [c1, c2, c3, c4, c5],
    ["5", "24", "4", "2", "Daily"],
    ["Covered companies", "Technical features", "Algorithms compared", "Model types", "Prediction frequency"],
):
    col.markdown(f'<div class="chip"><div class="chip-val">{val}</div><div class="chip-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── Covered companies ──────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Covered companies</p>', unsafe_allow_html=True)
comp_cols = st.columns(5)
for col, (ticker, name, sector) in zip(comp_cols, [
    ("AMZN", "Amazon",    "E-commerce / Cloud"),
    ("AAPL", "Apple",     "Consumer Tech"),
    ("MSFT", "Microsoft", "Enterprise Software"),
    ("GOOG", "Alphabet",  "Search / Advertising"),
    ("TSLA", "Tesla",     "Electric Vehicles"),
]):
    col.markdown(
        f'<div class="company-card" style="text-align:center;padding:18px 12px;">'
        f'<div class="chip-val" style="font-size:1.15rem;">{ticker}</div>'
        f'<div style="font-size:0.88rem;color:#e2e8f0;font-weight:600;margin-top:6px;">{name}</div>'
        f'<div class="chip-lbl" style="margin-top:4px;">{sector}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── What this app does ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">What this app does</p>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
for col, icon, title, desc in zip(
    [fc1, fc2, fc3],
    ["🔬", "⚡", "📊"],
    ["ML-Powered Predictions", "Live Data Pipeline", "Strategy Backtesting"],
    [
        "Two classification models are trained per company — a binary model (Rise/Fall) and a multi-class model (Big Fall / Small Fall / Small Rise / Big Rise). Each company's best-performing model is selected by cross-validated AUC-ROC or F1 Macro.",
        "The Go Live page connects to the SimFin API, runs the same ETL pipeline used at training time, and returns a trading signal with class probabilities — no lookahead bias, no data leakage.",
        "Simulate a BUY/SELL/HOLD strategy over any historical date range. Choose a simple signal-based approach or an advanced mode with confidence thresholds, position sizing, and transaction costs.",
    ],
):
    col.markdown(
        f'<div class="card"><div class="card-icon">{icon}</div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">How it works</p>', unsafe_allow_html=True)
hw1, hw2, hw3, hw4 = st.columns(4)
for col, num, title, desc in zip(
    [hw1, hw2, hw3, hw4],
    ["01", "02", "03", "04"],
    ["ETL Pipeline", "Model Training", "Model Export", "Live Prediction"],
    [
        "SimFin OHLCV data is enriched with 24 technical indicators — SMAs, MACD, Bollinger Bands, RSI, ATR, OBV and more. Binary and multi-class targets are derived from the following day's return.",
        "Four classifiers are evaluated per ticker: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. The best is chosen by AUC-ROC (binary) or F1 Macro (multi-class) via time-series cross-validation.",
        "The winning pipeline — scaler + model — is saved to a .joblib file and the exact feature list to a .txt file, guaranteeing identical preprocessing between training and inference.",
        "The Go Live page fetches fresh data, applies the identical ETL pipeline, and feeds the result into the saved model — returning a directional signal and probability breakdown.",
    ],
):
    col.markdown(
        f'<div class="how-card">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        f'<div class="step-num">{num}</div>'
        f'<span class="card-title" style="margin:0;">{title}</span>'
        f'</div>'
        f'<div class="card-desc">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
# ── How to use ─────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">How to use this app</p>', unsafe_allow_html=True)
gi1, gi2, gi3 = st.columns(3, gap="medium")

with gi1:
    st.markdown("""
    <div style="height:100%;">
    <div class="instr-box">
        <h4>⚡ Go Live — get a prediction</h4>
        <ul>
            <li>Click <strong>Go Live</strong> in the sidebar.</li>
            <li>Select a <strong>company ticker</strong> from the dropdown.</li>
            <li>Adjust the <strong>days of history</strong> shown on the price chart.</li>
            <li>Click <strong>Get Prediction</strong>.</li>
            <li>View the <strong>BUY / SELL / HOLD</strong> signal, class probabilities, and model KPIs.</li>
            <li>Expand <span class="tag">Features Used</span> to inspect all computed indicators.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

with gi2:
    st.markdown("""
    <div style="height:100%;">
    <div class="instr-box">
        <h4>🔁 Backtesting — simulate a strategy</h4>
        <ul>
            <li>Click <strong>Backtesting</strong> in the sidebar.</li>
            <li>Choose a ticker and a <strong>date range</strong>.</li>
            <li>Pick <strong>Simple</strong> mode (direct signal) or <strong>Advanced</strong> mode (confidence threshold, position sizing, and transaction costs).</li>
            <li>Review the equity curve, Sharpe ratio, max drawdown, and trade log.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

with gi3:
    st.markdown("""
    <div style="height:100%;">
    <div class="instr-box">
        <h4>🛠 Requirements before running</h4>
        <ul>
            <li>The live app on Streamlit Cloud is ready to use — no setup needed.</li>
            <li>To run locally, create a <span class="tag">.env</span> file in the project root:<br>
                <span class="tag">SIMFIN_API_KEY=your_key</span><br>
                Get a free key at <strong>simfin.com</strong></li>
            <li>Train the models first by running <span class="tag">ml_model_binary.ipynb</span> and <span class="tag">ml_model_multiclass.ipynb</span> — this saves the <span class="tag">.joblib</span> files to <span class="tag">models/</span>.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── Model Status Check ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-hdr">Model Status</p>', unsafe_allow_html=True)

TICKERS = ["AMZN", "AAPL", "MSFT", "GOOG", "TSLA"]
MODELS_DIR = Path(__file__).parent / "models"

# Best model per ticker from notebook training runs
BINARY_MODELS = {
    "AMZN": ("Logistic Regression (Tuned)", 0.5314),
    "AAPL": ("Random Forest (Tuned)",       0.4768),
    "MSFT": ("Ensemble",                    0.5617),
    "GOOG": ("Logistic Regression (Tuned)", 0.5658),
    "TSLA": ("Logistic Regression (Tuned)", 0.5841),
}
MULTI_MODELS = {
    "AMZN": ("Random Forest (Tuned)", 0.2315),
    "AAPL": ("Random Forest",         0.2338),
    "MSFT": ("XGBoost",               0.2523),
    "GOOG": ("Ensemble",              0.2442),
    "TSLA": ("Logistic Regression",   0.2982),
}

def _render_model_row(model_suffix: str, model_meta: dict) -> int:
    """Render one row of model status cards; return count of found models."""
    cols = st.columns(5)
    results = []
    for col, t in zip(cols, TICKERS):
        mp = MODELS_DIR / f"model_{t}_{model_suffix}.joblib"
        fp = MODELS_DIR / f"features_{t}_{model_suffix}.txt"
        ok = mp.exists() and fp.exists()
        results.append(ok)
        algo, score = model_meta[t]
        metric = "AUC-ROC" if model_suffix == "binary" else "F1 Macro"
        if ok:
            with open(fp) as fh:
                n = len([ln for ln in fh if ln.strip()])
            col.markdown(
                f'<div class="model-status">'
                f'<div class="model-ok">✓ {t}</div>'
                f'<div style="font-size:0.78rem;color:#e2e8f0;margin-top:5px;font-weight:500;">{algo}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:3px;">{metric}: {score} · {n} features</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<div class="model-status"><div class="model-miss">✗ {t}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Not found</div></div>',
                unsafe_allow_html=True,
            )
    return sum(results)

st.markdown("**Binary Models** — Rise / Fall")
binary_ok = _render_model_row("binary", BINARY_MODELS)
st.markdown("**Multi-Class Models** — Big Fall / Small Fall / Small Rise / Big Rise")
multi_ok = _render_model_row("multi", MULTI_MODELS)

if binary_ok < 5 or multi_ok < 5:
    st.warning("⚠️ Some models are missing. Run `ml_model_binary.ipynb` and `ml_model_multiclass.ipynb` to generate them.")

# ── Team ───────────────────────────────────────────────────────────────────────
from team_photos import marcos, nuria, dan, siddharth, teresa

def team_card(col, name, b64, position="center top", zoom="100%"):
    col.markdown(
        f'<div class="team-card">'
        f'<div style="width:72px;height:72px;border-radius:50%;'
        f'border:2px solid #7c3aed;margin:0 auto 10px;'
        f'overflow:hidden;display:flex;align-items:center;justify-content:center;">'
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:{zoom};height:{zoom};'
        f'object-fit:cover;object-position:{position};'
        f'min-width:72px;min-height:72px;">'
        f'</div>'
        f'<div class="team-name">{name}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:0 0 8px 0;">🧠 ML Team — ETL pipeline, ML model & trading strategy</p>', unsafe_allow_html=True)
ml1, ml2, ml3, _, _ = st.columns(5)
team_card(ml1, "Marcos Ortiz",    marcos,    position="center 20%",  zoom="100%")
team_card(ml2, "Dan Tigu",        dan,       position="center 20%",  zoom="100%")
team_card(ml3, "Nuria Diaz",      nuria,     position="center 30%",  zoom="250%")

st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:16px 0 8px 0;">💻 DEV Team — API wrapper, Streamlit web app & cloud deployment</p>', unsafe_allow_html=True)
dev1, dev2, _, _, _ = st.columns(5)
team_card(dev1, "Siddharth Murali", siddharth, position="center 20%", zoom="100%")
team_card(dev2, "Teresa Ghirardi",  teresa,    position="center 30%", zoom="250%")