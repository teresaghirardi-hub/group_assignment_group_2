"""
etl.py — Shared ETL & model utilities
======================================
Used by both pages/go_live.py and pages/backtesting.py.

Exports
-------
get_api_key()                        -> str
add_technical_features(df)           -> pd.DataFrame
prepare_for_prediction(df, features) -> pd.DataFrame
fix_sklearn_compatibility(obj)       -> obj
load_model(ticker, model_type)       -> (pipeline, feature_cols)
MODELS_DIR                           -> Path
"""
from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import LogisticRegression


# ── Path shared by both pages ──────────────────────────────────────────────────
# __file__ is  <project_root>/etl.py  →  models/ sits alongside it
MODELS_DIR = Path(__file__).parent / "models"


# ── API key resolution ─────────────────────────────────────────────────────────

def get_api_key() -> str:
    """
    Return the SimFin API key, checking in priority order:

    1. st.secrets["SIMFIN_API_KEY"]  — Streamlit Cloud secrets (secrets.toml)
    2. OS environment variable        — local .env loaded via python-dotenv,
                                        or a variable exported in the shell

    Stops the app with a clear error message if the key is not found in
    either location, so the user knows exactly what to fix.
    """
    # 1. Streamlit secrets (works on Streamlit Cloud and locally via .streamlit/secrets.toml)
    try:
        key = st.secrets["SIMFIN_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    # 2. Environment variable (local .env / shell export)
    import os
    key = os.environ.get("SIMFIN_API_KEY", "")
    if key:
        return key

    # Neither found — stop with a helpful message
    st.error(
        "**SimFin API key not found.**\n\n"
        "• **Locally:** add `SIMFIN_API_KEY=your_key` to a `.env` file at the "
        "project root and load it with `python-dotenv`, or export it in your shell.\n\n"
        "• **Streamlit Cloud:** add `SIMFIN_API_KEY = \"your_key\"` under "
        "*App settings → Secrets*."
    )
    st.stop()


# ── Technical feature engineering ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 24 technical features. Must match etl.ipynb exactly.

    Accepts a DataFrame whose columns may be lowercase or title-case
    (SimFin returns lowercase; some local CSVs use title-case).
    """
    df = df.copy()

    col = {c.lower(): c for c in df.columns}
    close  = df[col.get("close",  "close")]
    high   = df[col.get("high",   "high")]
    low    = df[col.get("low",    "low")]
    volume = df[col.get("volume", "volume")]

    date_col = col.get("date", "date")
    df[date_col] = pd.to_datetime(df[date_col])

    # ── Original 8 ──────────────────────────────────────────────────────────
    df["Returns"]       = np.log(close / close.shift(1))
    df["SMA_5"]         = close.rolling(window=5).mean()
    df["SMA_20"]        = close.rolling(window=20).mean()
    df["Volatility_5"]  = df["Returns"].rolling(window=5).std()
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()
    df["Volume_Change"] = volume.pct_change()

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs       = avg_gain / avg_loss
    df["RSI_14"]      = 100 - (100 / (1 + rs))
    df["Price_Range"] = (high - low) / close

    # ── MACD ────────────────────────────────────────────────────────────────
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    sma_20   = close.rolling(20).mean()
    std_20   = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    df["BB_Width"]    = (bb_upper - bb_lower) / sma_20
    df["BB_Position"] = (close - bb_lower) / (bb_upper - bb_lower)

    # ── Momentum ────────────────────────────────────────────────────────────
    df["Momentum_10"] = close / close.shift(10) - 1
    df["Momentum_20"] = close / close.shift(20) - 1

    # ── ATR Ratio ───────────────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["ATR_Ratio"] = atr_14 / close

    # ── Lagged Returns ───────────────────────────────────────────────────────
    df["Return_Lag1"] = df["Returns"].shift(1)
    df["Return_Lag2"] = df["Returns"].shift(2)
    df["Return_Lag3"] = df["Returns"].shift(3)
    df["Return_Lag5"] = df["Returns"].shift(5)

    # ── Volume Ratio ─────────────────────────────────────────────────────────
    df["Volume_Ratio"] = volume / volume.rolling(20).mean()

    # ── Day of Week ──────────────────────────────────────────────────────────
    df["DayOfWeek"] = df[date_col].dt.dayofweek

    # ── Distance from SMAs ───────────────────────────────────────────────────
    df["Dist_SMA_5"]  = (close - df["SMA_5"])  / df["SMA_5"]
    df["Dist_SMA_20"] = (close - df["SMA_20"]) / df["SMA_20"]

    return df


def prepare_for_prediction(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Apply ETL and return only the columns the model needs, dropping NaN rows."""
    df = add_technical_features(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features after ETL: {missing}")
    return df[feature_cols].dropna()


# ── Sklearn compatibility fix ──────────────────────────────────────────────────

def fix_sklearn_compatibility(obj):
    """
    Fix compatibility issues with models saved under older sklearn versions.

    In sklearn 1.5+, the 'multi_class' parameter was removed from
    LogisticRegression.  Old pickled models may lack the attribute entirely,
    causing errors when the new code tries to read it.

    Strategy: walk the model object recursively and add the attribute (set to
    'deprecated') wherever it is missing.
    """
    if isinstance(obj, LogisticRegression):
        if not hasattr(obj, "multi_class"):
            object.__setattr__(obj, "multi_class", "deprecated")
        return obj

    # sklearn Pipeline
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            fix_sklearn_compatibility(step)

    # Fitted ensembles (VotingClassifier, etc.)
    if hasattr(obj, "estimators_"):
        for est in obj.estimators_:
            fix_sklearn_compatibility(est)

    # Unfitted ensembles — list of (name, estimator) tuples
    if hasattr(obj, "estimators"):
        for item in obj.estimators:
            fix_sklearn_compatibility(item[1] if isinstance(item, tuple) else item)

    # Named steps dict
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            fix_sklearn_compatibility(step)

    return obj


# ── Model loader ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(ticker: str, model_type: str):
    """
    Load a trained pipeline and its feature list from disk.

    Parameters
    ----------
    ticker     : e.g. 'AAPL'
    model_type : 'binary' or 'multi'

    Returns
    -------
    (pipeline, feature_cols)

    Raises
    ------
    FileNotFoundError  if the model or features file is missing.
    """
    model_path    = MODELS_DIR / f"model_{ticker}_{model_type}.joblib"
    features_path = MODELS_DIR / f"features_{ticker}_{model_type}.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    pipeline = joblib.load(model_path)
    pipeline = fix_sklearn_compatibility(pipeline)

    with open(features_path) as fh:
        features = [line.strip() for line in fh if line.strip()]

    return pipeline, features
