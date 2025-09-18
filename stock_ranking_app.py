import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import numpy as np
import pandas as pd
import streamlit as st

# BART (pure Python via PyMC)
import pymc as pm
import pymc_bart as pmb
from scipy.special import expit

# ---------- Page + light CSS ----------
st.set_page_config(page_title="Premarket Stock Ranking (Python BART)", layout="wide")
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking (Python-only BART)")

# ---------- helpers ----------
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
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

# ---------- session ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- qualitative ----------
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

# ---------- sidebar ----------
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
for k in q_weights: q_weights[k] = q_weights[k] / qual_sum

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider(
    "Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
    help="Used for CI bands around predicted day volume."
)

# ---------- numeric scorers ----------
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

# ---------- data mapping + features ----------
def _pick_col(nms, patterns):
    lnms = [x.lower() for x in nms]
    for pat in patterns:
        for i, nm in enumerate(lnms):
            if pd.Series(nm).str.contains(pat, regex=True).iloc[0]:
                return i
    return None

def read_excel_dynamic(file, sheet="PMH BO Merged") -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")
    nms = list(df.columns)

    def _pick_col(nms, patterns):
        lnms = [x.lower() for x in nms]
        for pat in patterns:
            for i, nm in enumerate(lnms):
                if pd.Series(nm).str.contains(pat, regex=True).iloc[0]:
                    return i
        return None

    col_PMVolM   = _pick_col(nms, [r"^pm\s*\$?\s*vol.*\(m\)$", r"^pm\s*vol.*m", r"^pm\s*vol(ume)?$"])
    col_PMDolM   = _pick_col(nms, [r"^pm.*\$\s*vol.*\(m\)$", r"dollar.*pm.*\(m\)$", r"^pm\s*\$?\s*vol.*\(m\)$"])
    col_FloatM   = _pick_col(nms, [r"^float.*\(m\)$", r"^float.*m", r"^float$", r"public.*float"])
    col_GapPct   = _pick_col(nms, [r"^gap.*%$", r"^gap_?pct$", r"\bgap\b"])
    col_ATR      = _pick_col(nms, [r"^daily\s*atr$", r"^atr(\s*\$)?$", r"\batr\b"])
    col_MCapM    = _pick_col(nms, [r"^market.?cap.*\(m\)$", r"mcap.*\(m\)", r"^market.?cap", r"^mcap\s*m?$"])
    col_DVolM    = _pick_col(nms, [r"^daily\s*vol(ume)?\s*\(m\)$", r"^daily\s*vol(ume)?$", r"^d.*vol.*\(m\)$", r"^daily\s*vol.*m$"])
    col_Catalyst = _pick_col(nms, [r"^catalyst(flag|_?flag)?$", r"^catalyst$", r"^news$", r"catalyst.*score"])
    col_FT       = _pick_col(nms, [r"^ft$", r"follow.?through", r"^ft_?label$", r"^label$"])

    def _num(col):
        if col is None: return pd.Series([np.nan]*len(df))
        s = df.iloc[:, col]
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
        s = s.mask(s.eq("") | s.str.upper().isin(["NA","N/A","NULL","-"]))
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame({
        "PMVolM":   _num(col_PMVolM),
        "PMDolM":   _num(col_PMDolM),
        "FloatM":   _num(col_FloatM),
        "GapPct":   _num(col_GapPct),
        "ATR":      _num(col_ATR),
        "MCapM":    _num(col_MCapM),
        "DVolM":    _num(col_DVolM),
        "Catalyst": df.iloc[:, col_Catalyst] if col_Catalyst is not None else 0,
        "FT_raw":   df.iloc[:, col_FT] if col_FT is not None else np.nan,
    })

    # Catalyst → {0,1}
    if not pd.api.types.is_numeric_dtype(out["Catalyst"]):
        c = out["Catalyst"].astype(str).str.lower().str.strip()
        out["Catalyst"] = np.where(c.isin(["yes","y","true","1","ft","hot"]), 1, 0).astype(int)
    else:
        out["Catalyst"] = (pd.to_numeric(out["Catalyst"], errors="coerce").fillna(0).astype(float) != 0).astype(int)

    # FT label → categorical; keep NaN if no label present
    f_raw = out["FT_raw"]
    fl = f_raw.astype(str).str.lower().str.strip()
    is_ft = fl.isin(["ft","1","yes","y","true"])
    # If FT_raw is NaN -> keep NaN; else label FT/Fail
    ft_vals = np.where(f_raw.isna(), np.nan, np.where(is_ft, "FT", "Fail"))
    out["FT_fac"] = pd.Series(ft_vals, index=out.index).astype("category")

    return out

def featurize(raw: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    out = raw.copy()
    out["FR"]       = out["PMVolM"] / np.maximum(out["FloatM"].replace(0, np.nan), eps)
    out["ln_DVol"]  = np.log(np.maximum(out["DVolM"], eps))
    out["ln_pm"]    = np.log(np.maximum(out["PMVolM"], eps))
    out["ln_pmdol"] = np.log(np.maximum(out["PMDolM"], eps))
    out["ln_fr"]    = np.log(np.maximum(out["FR"], eps))
    out["ln_gapf"]  = np.log(np.maximum(out["GapPct"], 0)/100 + eps)
    out["ln_atr"]   = np.log(np.maximum(out["ATR"], eps))
    out["ln_mcap"]  = np.log(np.maximum(out["MCapM"], eps))
    out["ln_pmdol_per_mcap"] = np.log(np.maximum(out["PMDolM"] / np.maximum(out["MCapM"], eps), eps))
    out["Catalyst"] = (pd.to_numeric(out.get("Catalyst", 0), errors="coerce").fillna(0).astype(float) != 0).astype(int)
    out["FT_fac"]   = out["FT_fac"].astype("category")
    return out

# ---- Predictor sets
A_FEATURES_DEFAULT = ["ln_pm","ln_pmdol","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst"]
B_FEATURES_CORE    = ["ln_pm","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst","ln_pmdol","PredVol_M"]

# ---- Small helpers
def _assert_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))
        raise ValueError(f"{name} contains non-finite values at indices {bad}.")

def _sample(draws, tune, chains, seed):
    # Light but stable defaults for speed
    return pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,         # keep 1 unless you really need R-hat
        cores=1,               # Streamlit Cloud: 1 core prevents oversubscription
        target_accept=0.85,    # a bit lower => fewer divergences, faster
        progressbar=True,      # show progress so it doesn't feel "stuck"
        random_seed=seed,
        discard_tuned_samples=True,
        jitter_max_retries=0,
        compute_convergence_checks=False,
    )

# ---------- BART training ----------
def train_model_A(df_feats: pd.DataFrame, predictors: list[str], draws=800, tune=800, trees=200, seed=42):
    dfA = df_feats.dropna(subset=["ln_DVol"] + predictors).copy()
    if len(dfA) < 30:
        raise RuntimeError("Not enough rows to train Model A (need ≥30).")

    X = dfA[predictors].to_numpy(dtype=float)
    y = dfA["ln_DVol"].to_numpy(dtype=float)

    with pm.Model() as mA:
        X_A = pm.MutableData("X_A", X)             # <— key: data container
        f = pmb.BART("f", X_A, y, m=trees)         # latent (log volume)
        sigma = pm.Exponential("sigma", 1.0)
        pm.Normal("y_obs", mu=f, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          target_accept=0.9, random_seed=seed, progressbar=False)
    # store the data name for later set_data()
    return {"model": mA, "trace": trace, "predictors": predictors, "x_name": "X_A"}

def predict_model_A(bundle, Xnew_df: pd.DataFrame) -> np.ndarray:
    X = Xnew_df[bundle["predictors"]].to_numpy(dtype=float)
    with bundle["model"]:
        pm.set_data({bundle["x_name"]: X})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"],
            var_names=["y_obs"],
            return_inferencedata=False,
            progressbar=False
        )

    arr = np.asarray(ppc["y_obs"])
    # Expected shapes: (chains, draws, n) OR (draws, n) OR (n,)
    if arr.ndim == 3:
        # chains × draws × n_obs  -> mean over chains & draws
        ln_mean = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        # draws × n_obs -> mean over draws
        ln_mean = arr.mean(axis=0)
    elif arr.ndim == 1:
        # already n_obs
        ln_mean = arr
    else:
        raise ValueError(f"Unexpected PPC shape for y_obs: {arr.shape}")

    if ln_mean.shape[0] != X.shape[0]:
        raise ValueError(f"predict_model_A length mismatch: {ln_mean.shape[0]} vs {X.shape[0]}")

    return np.exp(ln_mean)

def train_model_B(df_feats_with_predvol: pd.DataFrame, predictors_core: list[str],
                  draws=400, tune=400, trees=75, chains=1, seed=123):
    """
    df_feats_with_predvol MUST already contain a finite 'PredVol_M' column.
    """
    dfB2 = df_feats_with_predvol.dropna(subset=["FT_fac","PredVol_M"]).copy()
    if len(dfB2) < 30 or dfB2["FT_fac"].nunique() < 2:
        raise RuntimeError("Not enough labeled rows or only one class for Model B.")

    preds_B = predictors_core.copy()
    if "PredVol_M" not in preds_B:
        preds_B = preds_B + ["PredVol_M"]

    X = dfB2[preds_B].to_numpy(dtype=float)
    y = (dfB2["FT_fac"].astype(str) == "FT").to_numpy(dtype=int)

    _assert_finite("X_B", X)
    _assert_finite("y_B", y)

    with pm.Model() as mB:
        X_B = pm.MutableData("X_B", X)            # <— data container
        f = pmb.BART("f", X_B, y, m=trees)        # latent logit
        p = pm.Deterministic("p", pm.math.sigmoid(f))
        pm.Bernoulli("y_obs", p=p, observed=y)
        trace = _sample(draws, tune, chains, seed)

    return {"model": mB, "trace": trace, "predictors": preds_B, "x_name": "X_B"}

def predict_model_B(bundle, Xnew_df: pd.DataFrame) -> np.ndarray:
    X = Xnew_df[bundle["predictors"]].to_numpy(dtype=float)
    with bundle["model"]:
        pm.set_data({bundle["x_name"]: X})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"],
            var_names=["y_obs"],    # Bernoulli draws 0/1
            return_inferencedata=False,
            progressbar=False
        )

    arr = np.asarray(ppc["y_obs"])
    # shapes: (chains, draws, n) or (draws, n) or (n,)
    if arr.ndim == 3:
        probs = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        probs = arr.mean(axis=0)
    elif arr.ndim == 1:
        probs = arr.astype(float)
    else:
        raise ValueError(f"Unexpected PPC shape for y_obs: {arr.shape}")

    if probs.shape[0] != X.shape[0]:
        raise ValueError(f"predict_model_B length mismatch: {probs.shape[0]} vs {X.shape[0]}")

    return probs.astype(float)

def predict_model_B(bundle, Xnew: pd.DataFrame) -> np.ndarray:
    X = Xnew[bundle["predictors"]].to_numpy(dtype=float)
    with bundle["model"]:
        f_new = pmb.predict(bundle["trace"], X, kind="mean")  # latent
    return expit(f_new)  # probability

def _row_cap(df: pd.DataFrame, n: int, y_col: str | None = None):
    if n <= 0 or len(df) <= n: 
        return df
    if y_col is None or y_col not in df.columns:
        return df.sample(n=n, random_state=42)
    # For classification (Model B): keep class balance
    pos = df[df[y_col] == 1]
    neg = df[df[y_col] == 0]
    n_pos = max(1, int(n * len(pos) / max(1, len(df))))
    n_neg = max(1, n - n_pos)
    return pd.concat([
        pos.sample(n=min(n_pos, len(pos)), random_state=42),
        neg.sample(n=min(n_neg, len(neg)), random_state=42)
    ], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

def _hash_df_for_cache(df: pd.DataFrame) -> int:
    try:
        return int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        return len(df)  # fallback

# ---------- TRAINING PANEL ----------
with st.expander("⚙️ Train / Load BART models (Python-only)"):
    st.write("Upload your **PMH Database.xlsx** (sheet: _PMH BO Merged_) to retrain.")
    up = st.file_uploader("Upload Excel", type=["xlsx"], accept_multiple_files=False)

    col_a, col_b = st.columns(2)
    with col_a:
        fast_mode = st.toggle("Fast training mode", value=True, help="Cuts trees/draws and caps rows.")
        trees = st.slider("BART trees per model", 25, 400, 200 if not fast_mode else 75, 25)
        draws = st.slider("MCMC draws", 200, 1500, 800 if not fast_mode else 250, 50)
    with col_b:
        tune  = st.slider("MCMC tune", 200, 1500, 800 if not fast_mode else 250, 50)
        chains = 1 if fast_mode else st.slider("Chains", 1, 2, 2)
        seed  = st.number_input("Random seed", value=42, step=1)

    # Row caps (apply after feature engineering)
    capA = st.number_input("Max rows for Model A", min_value=0, value=1200 if fast_mode else 0,
                           help="0 = no cap")
    capB = st.number_input("Max rows for Model B (after PredVol_M)", min_value=0, value=600 if fast_mode else 0,
                           help="0 = no cap")

    if st.button("Train models", use_container_width=True, type="primary"):
        if not up:
            st.error("Upload the Excel file first.")
        else:
            with st.spinner("Training Model A and Model B with BART…"):
                raw = read_excel_dynamic(up)
                df_all = featurize(raw)

                # Optional cap for A
                dfA = df_all.dropna(subset=["ln_DVol"] + A_FEATURES_DEFAULT)
                dfA = _row_cap(dfA, int(capA)) if capA else dfA

                # Train A
                A_bundle = train_model_A(
                    dfA, A_FEATURES_DEFAULT,
                    draws=draws, tune=tune, trees=trees, seed=seed
                )

                # --- Use A to generate PredVol_M across eligible rows
                pred_cols = A_bundle["predictors"]
                
                # Build X on the exact rows that have all predictors present
                okA = df_all[pred_cols].dropna().index
                if len(okA) == 0:
                    st.error("No rows pass Model A predictors; cannot proceed to Model B.")
                    st.stop()
                
                Xnew = df_all.loc[okA]  # rows ready for prediction
                preds = predict_model_A(A_bundle, Xnew)
                
                # Safety: ensure shape matches
                preds = np.asarray(preds).reshape(-1)
                if preds.shape[0] != len(okA):
                    st.error(f"Internal mismatch: got {preds.shape[0]} preds for {len(okA)} rows.")
                    st.stop()
                
                # Assign predictions back into df_all_with_pred
                df_all_with_pred = df_all.copy()
                df_all_with_pred.loc[okA, "PredVol_M"] = pd.Series(preds, index=okA, dtype="float64")

                # Prep B data, cap with class balance
                dfB = df_all_with_pred.dropna(subset=["FT_fac", "PredVol_M"]).copy()
                # Convert FT_fac → 1/0 once here for the cap helper
                dfB["_y"] = (dfB["FT_fac"].astype(str) == "FT").astype(int)
                dfB_cap = _row_cap(dfB, int(capB), y_col="_y") if capB else dfB

                # Train B
                B_bundle = train_model_B(
                    dfB_cap.drop(columns=["_y"]),
                    B_FEATURES_CORE,
                    draws=draws if not fast_mode else max(200, draws),
                    tune=tune if not fast_mode else max(200, tune),
                    trees= max(50, trees // (2 if not fast_mode else 3)),
                    chains=chains,
                    seed=seed+1
                )

                # Save + small cache key to avoid accidental retrain
                st.session_state["A_bundle"] = A_bundle
                st.session_state["B_bundle"] = B_bundle
                st.session_state["data_hash"] = _hash_df_for_cache(df_all)
                st.success(f"Trained (rows A: {len(dfA)}, rows B: {len(dfB_cap)}).")

# ---------- UI: Add / Ranking ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

def require_models():
    if "A_bundle" not in st.session_state or "B_bundle" not in st.session_state:
        st.warning("Train BART models first (see expander above).")
        st.stop()

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

        # MIDDLE
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
                # value is (idx, text); we use idx later (1..7)
                st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None),
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        require_models()
        A_bundle = st.session_state["A_bundle"]
        B_bundle = st.session_state["B_bundle"]

        # Compute PM $Vol (Millions)
        pm_dol_m = pm_vol_m * pm_vwap

        # Build a single-row feature frame (matching training transforms)
        row = pd.DataFrame([{
            "PMVolM": pm_vol_m, "PMDolM": pm_dol_m, "FloatM": float_m, "GapPct": gap_pct,
            "ATR": atr_usd, "MCapM": mc_m, "Catalyst": 1 if catalyst_points > 0 else 0,
            "DVolM": np.nan, "FT_fac": "Fail",
        }])
        feats = featurize(row)

        # Model A prediction (millions)
        pred_vol_m = float(predict_model_A(A_bundle, feats)[0])

        # Model B needs PredVol_M
        featsB = feats.copy()
        featsB["PredVol_M"] = pred_vol_m
        ft_prob = float(predict_model_B(B_bundle, featsB)[0])

        # CI bands (log space)
        ci68_l = pred_vol_m * math.exp(-1.0 * sigma_ln)
        ci68_u = pred_vol_m * math.exp(+1.0 * sigma_ln)
        ci95_l = pred_vol_m * math.exp(-1.96 * sigma_ln)
        ci95_u = pred_vol_m * math.exp(+1.96 * sigma_ln)

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

        ft_label = ("FT likely" if ft_prob >= 0.60 else
                    "Toss-up"   if ft_prob >= 0.40 else
                    "FT unlikely")

        row_out = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            "FT_Prob": round(ft_prob, 3),
            "FT_Label": ft_label,

            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),

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
