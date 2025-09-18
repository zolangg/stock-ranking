# stock_ranking_app.py
import os, math, time, json
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Dict

# ==============================================
# Page + styles
# ==============================================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .smallcap { color:#6b7280; font-size:12px; margin-top:-8px; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      div[role="radiogroup"] label p { font-size: 0.88rem; line-height:1.25rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
      .ok { color:#059669; }
      .warn { color:#d97706; }
      .bad { color:#b91c1c; }
      .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking (Python models)")

# ==============================================
# Session state
# ==============================================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if "MODEL_A" not in st.session_state: st.session_state.MODEL_A = None
if "MODEL_B" not in st.session_state: st.session_state.MODEL_B = None
if "A_FEATURES" not in st.session_state: st.session_state.A_FEATURES = []
if "B_FEATURES" not in st.session_state: st.session_state.B_FEATURES = []

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ==============================================
# Helpers
# ==============================================
def df_to_markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if not keep: return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy().fillna("")
    header = "| " + " | ".join(keep) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = []
        for c in keep:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

# ==============================================
# Qualitative criteria (your original)
# ==============================================
QUAL_CRITERIA = [
    {
        "name": "GapStruct",
        "question": "Gap & Trend Development:",
        "options": [
            "Gap fully reversed: price loses >80% of gap.",
            "Choppy reversal: price loses 50–80% of gap.",
            "Partial retracement: price loses 25–50% of gap.",
            "Sideways consolidation: gap holds, price within top 25% of gap.",
            "Uptrend with deep pullbacks (>30% retrace).",
            "Uptrend with moderate pullbacks (10–30% retrace).",
            "Clean uptrend, only minor pullbacks (<10%).",
        ],
        "weight": 0.15,
        "help": "How well the gap holds and trends.",
    },
    {
        "name": "LevelStruct",
        "question": "Key Price Levels:",
        "options": [
            "Fails at all major support/resistance; cannot hold any key level.",
            "Briefly holds/reclaims a level but loses it quickly; repeated failures.",
            "Holds one support but unable to break resistance; capped below a key level.",
            "Breaks above resistance but cannot stay; dips below reclaimed level.",
            "Breaks and holds one major level; most resistance remains above.",
            "Breaks and holds several major levels; clears most overhead resistance.",
            "Breaks and holds above all resistance; blue sky.",
        ],
        "weight": 0.15,
        "help": "Break/hold behavior at key levels.",
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "Sharp, accelerating downtrend; new lows repeatedly.",
            "Persistent downtrend; still lower lows.",
            "Downtrend losing momentum; flattening.",
            "Clear base; sideways consolidation.",
            "Bottom confirmed; higher low after base.",
            "Uptrend begins; breaks out of base.",
            "Sustained uptrend; higher highs, blue sky.",
        ],
        "weight": 0.10,
        "help": "Higher-timeframe bias.",
    },
]

# ==============================================
# Sidebar: weights & modifiers
# ==============================================
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}"
    )
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
                             help="Used for CI bands around predicted day volume.")

# ==============================================
# Numeric bucket scorers
# ==============================================
def pts_rvol(x: float) -> int:
    for th, p in [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)]:
        if x < th: return p
    return 7

def pts_atr(x: float) -> int:
    for th, p in [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)]:
        if x < th: return p
    return 7

def pts_si(x: float) -> int:
    for th, p in [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)]:
        if x < th: return p
    return 7

def pts_fr(pm_vol_m: float, float_m: float) -> int:
    if float_m <= 0: return 1
    rot = pm_vol_m / float_m
    for th, p in [(0.01,1),(0.03,2),(0.10,3),(0.25,4),(0.50,5),(1.00,6)]:
        if rot < th: return p
    return 7

def pts_float(float_m: float) -> int:
    if float_m <= 3: return 7
    for th, p in [(200,2),(100,3),(50,4),(35,5),(10,6)]:
        if float_m > th: return p
    return 7

def odds_label(score: float) -> str:
    if score >= 85: return "Very High Odds"
    elif score >= 70: return "High Odds"
    elif score >= 55: return "Moderate Odds"
    elif score >= 40: return "Low Odds"
    else: return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 85 else
            "A+"  if score_pct >= 80 else
            "A"   if score_pct >= 70 else
            "B"   if score_pct >= 60 else
            "C"   if score_pct >= 45 else "D")

# ==============================================
# Feature engineering (exactly like your R pipeline)
# ==============================================
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = df.copy()

    # ensure numeric columns exist
    for c in ["PMVolM","PMDolM","FloatM","GapPct","ATR","MCapM"]:
        if c not in out: out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # catalyst binary
    out["Catalyst"] = (out.get("Catalyst", 0).fillna(0).astype(float) != 0).astype(int)

    # engineered
    out["FR"]       = out["PMVolM"] / np.maximum(out["FloatM"], eps)
    out["ln_pm"]    = np.log(np.maximum(out["PMVolM"], eps))
    out["ln_pmdol"] = np.log(np.maximum(out["PMDolM"], eps))
    out["ln_fr"]    = np.log(np.maximum(out["FR"], eps))
    out["ln_gapf"]  = np.log(np.maximum(out["GapPct"], 0)/100.0 + eps)
    out["ln_atr"]   = np.log(np.maximum(out["ATR"], eps))
    out["ln_mcap"]  = np.log(np.maximum(out["MCapM"], eps))
    out["ln_pmdol_per_mcap"] = np.log(np.maximum(out["PMDolM"] / np.maximum(out["MCapM"], eps), eps))

    # optional targets if present
    if "DVolM" in out:
        out["ln_DVol"] = np.log(np.maximum(pd.to_numeric(out["DVolM"], errors="coerce"), eps))
    if "FT" in out:
        out["FT_fac"] = np.where(out["FT"].astype(str).str.strip().str.lower().isin(["ft","1","true","yes","y"]), 1, 0)
    return out

# ==============================================
# Model training (Python only; RF default, XGB optional)
# ==============================================
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_model_A(df_all: pd.DataFrame, feature_cols: List[str], use_xgb: bool=False):
    dfA = df_all.dropna(subset=["ln_DVol"] + feature_cols).copy()
    if len(dfA) < 30:
        raise RuntimeError("Not enough rows to train Model A (need ≥30).")

    X = dfA[feature_cols].values
    y = dfA["ln_DVol"].values

    if use_xgb:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=700, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            learning_rate=0.05, reg_lambda=1.0, n_jobs=4, random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=900, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=4
        )

    k = 5 if len(dfA) >= 60 else 3
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_cv = []
    for tr, te in cv.split(X):
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])
        r2_cv.append(r2_score(y[te], yhat))
    model.fit(X, y)
    yhat_in = model.predict(X)
    r2_in = r2_score(y, yhat_in)

    return {
        "model": model,
        "features": feature_cols,
        "n": int(len(dfA)),
        "r2_cv_mean": float(np.mean(r2_cv)),
        "r2_cv_sd": float(np.std(r2_cv)),
        "r2_in": float(r2_in),
    }

def train_model_B(df_all: pd.DataFrame, feature_cols_core: List[str], modelA: Dict, use_xgb: bool=False):
    df = df_all.copy()
    A_feats = modelA["features"]

    # PredVol from Model A
    rows = df.dropna(subset=A_feats).index
    df.loc[rows, "PredVol_M"] = np.exp(modelA["model"].predict(df.loc[rows, A_feats].values))

    preds_B = feature_cols_core + ["PredVol_M"]
    dfB = df.dropna(subset=preds_B + ["FT_fac"]).copy()
    if len(dfB) < 30 or dfB["FT_fac"].nunique() < 2:
        raise RuntimeError("Not enough labeled rows or only one class for Model B.")

    X = dfB[preds_B].values
    y = dfB["FT_fac"].astype(int).values

    if use_xgb:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=700, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            learning_rate=0.05, reg_lambda=1.0, n_jobs=4, random_state=42,
            eval_metric="logloss"
        )
    else:
        model = RandomForestClassifier(
            n_estimators=900, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=4
        )

    k = 5 if len(dfB) >= 60 else 3
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
    model.fit(X, y)
    auc_in = roc_auc_score(y, model.predict_proba(X)[:,1])

    return {
        "model": model,
        "features": preds_B,
        "n": int(len(dfB)),
        "auc_cv_mean": float(np.mean(aucs)),
        "auc_cv_sd": float(np.std(aucs)),
        "auc_in": float(auc_in),
    }

# ==============================================
# Persistence (models/)
# ==============================================
from pathlib import Path
import joblib

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

A_PATH = MODELS_DIR / "model_A.pkl"
B_PATH = MODELS_DIR / "model_B.pkl"
META_PATH = MODELS_DIR / "model_meta.json"

def save_models(A_bundle: Dict, B_bundle: Dict):
    joblib.dump(A_bundle, A_PATH)
    joblib.dump(B_bundle, B_PATH)
    meta = {
        "ts": time.time(),
        "A_features": A_bundle["features"],
        "B_features": B_bundle["features"],
        "A_n": A_bundle["n"],
        "B_n": B_bundle["n"],
        "A_r2_cv_mean": A_bundle["r2_cv_mean"],
        "B_auc_cv_mean": B_bundle["auc_cv_mean"],
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

def try_load_models():
    if A_PATH.exists() and B_PATH.exists():
        A = joblib.load(A_PATH)
        B = joblib.load(B_PATH)
        st.session_state.MODEL_A = A
        st.session_state.MODEL_B = B
        st.session_state.A_FEATURES = A["features"]
        st.session_state.B_FEATURES = B["features"]
        return True
    return False

_ = try_load_models()  # load if present on startup

# ==============================================
# TRAINING UI
# ==============================================
st.markdown('<div class="section-title">Model Training (Python)</div>', unsafe_allow_html=True)
use_xgb = st.checkbox("Use XGBoost (if available)", value=False, help="If not installed, leave off (RandomForest used).")
up = st.file_uploader("Upload your PMH BO Merged (xlsx or csv) to (re)train models", type=["xlsx","csv"])
train_btn = st.button("Train / Retrain models")

# Feature sets (mirror your R/BART predictors)
A_FEATURES_DEFAULT = ["ln_pm","ln_pmdol","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst"]
B_FEATURES_CORE    = ["ln_pm","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst","ln_pmdol","ln_pmdol_per_mcap"]

if train_btn:
    if up is None:
        st.error("Upload a sheet first.")
        st.stop()
    if up.name.lower().endswith(".xlsx"):
        raw = pd.read_excel(up)
    else:
        raw = pd.read_csv(up)

    df_all = featurize(raw)

    with st.spinner("Training Model A (ln DVol)…"):
        A_bundle = train_model_A(df_all, A_FEATURES_DEFAULT, use_xgb=use_xgb)
    with st.spinner("Training Model B (FT classifier)…"):
        B_bundle = train_model_B(df_all, B_FEATURES_CORE, A_bundle, use_xgb=use_xgb)

    save_models(A_bundle, B_bundle)
    st.session_state.MODEL_A = A_bundle
    st.session_state.MODEL_B = B_bundle
    st.session_state.A_FEATURES = A_bundle["features"]
    st.session_state.B_FEATURES = B_bundle["features"]

    st.success("Models trained & saved to ./models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model A (ln Daily Volume)**")
        st.json({
            "n_rows": A_bundle["n"],
            "features": A_bundle["features"],
            "R² (CV mean)": round(A_bundle["r2_cv_mean"], 3),
            "R² (in-sample)": round(A_bundle["r2_in"], 3)
        })
    with c2:
        st.markdown("**Model B (FT classifier)**")
        st.json({
            "n_rows": B_bundle["n"],
            "features": B_bundle["features"],
            "AUC (CV mean)": round(B_bundle["auc_cv_mean"], 3),
            "AUC (in-sample)": round(B_bundle["auc_in"], 3)
        })

st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)

# ==============================================
# SCORING UI (Add Stock) — uses trained models
# ==============================================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Numeric Context</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.25, 1.25, 1.0])

        # LEFT
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # MIDDLE (order you requested)
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

        # RIGHT
        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Qualitative Context</div>', unsafe_allow_html=True)

        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None),
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Check models
        if not (st.session_state.MODEL_A and st.session_state.MODEL_B):
            st.error("Train or load models first (top of page).")
            st.stop()

        # compute PM $Vol (Millions)
        pm_dol_m = pm_vol_m * pm_vwap

        # feature row (same engineering as training)
        row = pd.DataFrame([{
            "PMVolM": pm_vol_m,
            "PMDolM": pm_dol_m,
            "FloatM": float_m,
            "GapPct": gap_pct,
            "ATR":    atr_usd,
            "MCapM":  mc_m,
            "Catalyst": 1 if catalyst_points > 0 else 0,
        }])
        f = featurize(row)

        # ---- Model A: Predicted Daily Volume (millions)
        A = st.session_state.MODEL_A
        featsA = A["features"]
        pred_ln = float(A["model"].predict(f[featsA].values)[0])
        pred_vol_m = math.exp(pred_ln)

        # CI bands (lognormal assumption with sidebar σ)
        ci68_l = pred_vol_m * math.exp(-1.0 * sigma_ln)
        ci68_u = pred_vol_m * math.exp(+1.0 * sigma_ln)
        ci95_l = pred_vol_m * math.exp(-1.96 * sigma_ln)
        ci95_u = pred_vol_m * math.exp(+1.96 * sigma_ln)

        # ---- Model B: FT probability (requires PredVol_M)
        f["PredVol_M"] = pred_vol_m
        B = st.session_state.MODEL_B
        featsB = B["features"]
        ft_prob = float(B["model"].predict_proba(f[featsB].values)[0,1])
        ft_label = ("FT likely" if ft_prob >= 0.60 else
                    "Toss-up"   if ft_prob >= 0.40 else
                    "FT unlikely")

        # ---- Numeric points ----
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # ---- Qualitative (weighted 1..7) ----
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            raw = st.session_state.get(f"qual_{crit['name']}", (1,))
            sel = raw[0] if isinstance(raw, tuple) else raw
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # ---- Combine + modifiers ----
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(
            combo_pct + news_weight*(1 if catalyst_points>0 else 0)*10 + dilution_weight*(dilution_points)*10, 2
        )
        final_score = max(0.0, min(100.0, final_score))

        # ---- Diagnostics for display ----
        pm_float_rot_x  = (pm_vol_m / float_m) if float_m > 0 else 0.0
        pm_pct_of_pred  = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        row_out = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            # Predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            "FT_Prob": round(ft_prob, 3),
            "FT_Label": ft_label,

            # Compact diagnostics
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),

            # raw inputs (debug)
            "_MCap_M": mc_m, "_ATR_$": atr_usd, "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m,
            "_Float_M": float_m, "_Gap_%": gap_pct, "_Catalyst": 1 if catalyst_points>0 else 0,
        }

        st.session_state.rows.append(row_out)
        st.session_state.last = row_out
        st.session_state.flash = (
            f"Saved {ticker} — {ft_label} ({ft_prob*100:.1f}%) · "
            f"Odds {row_out['Odds']} (Score {row_out['FinalScore']})"
        )
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Pred Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        cC.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        cD.metric("FT Probability", f"{l.get('FT_Prob',0)*100:.1f}%")
        cD.caption(l.get("FT_Label","—"))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d1.caption("Premarket volume ÷ predicted day volume × 100.")
        d2.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("PM dollar volume ÷ market cap × 100.")
        d3.metric("Numeric Block", f"{l.get('Numeric_%',0):.1f}%")
        d3.caption("Weighted buckets: RVOL / ATR / SI / FR / Float.")
        d4.metric("Qual Block", f"{l.get('Qual_%',0):.1f}%")
        d4.caption("Weighted radios: structure, levels, higher TF.")

# ==============================================
# RANKING TAB
# ==============================================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","FT_Label","FT_Prob",
            "Numeric_%","Qual_%","FinalScore","Level",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
            "PM_%_of_Pred","PM$ / MC_%"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","FT_Label","Level") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "FT_Label": st.column_config.TextColumn("FT Label"),
                "FT_Prob": st.column_config.NumberColumn("FT Prob", format="%.3f"),
                "Numeric_%": st.column_config.NumberColumn("Numeric %", format="%.1f"),
                "Qual_%": st.column_config.NumberColumn("Qual %", format="%.1f"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "Level": st.column_config.TextColumn("Level"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
            }
        )

        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, c2, _ = st.columns([0.25, 0.25, 0.5])
        with c1:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "ranking.csv",
                "text/csv",
                use_container_width=True
            )
        with c2:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
