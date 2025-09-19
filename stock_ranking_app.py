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

# ---------- BART training / prediction ----------
def train_model_A(df_feats: pd.DataFrame, predictors: list[str],
                  draws=600, tune=400, trees=150, seed=42, chains=1):
    dfA = df_feats.dropna(subset=["ln_DVol"] + predictors).copy()
    if len(dfA) < 30:
        raise RuntimeError("Not enough rows to train Model A (need ≥30).")

    X = np.ascontiguousarray(dfA[predictors].to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(dfA["ln_DVol"].to_numpy(dtype=np.float64))

    with pm.Model() as mA:
        X_A = pm.MutableData("X_A", X)
        f = pmb.BART("f", X_A, y, m=trees)      # latent mean (log-volume)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=f, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            cores=1,
            random_seed=seed,
            target_accept=0.9,
            init="adapt_diag",
            progressbar=True,
            discard_tuned_samples=True,
            compute_convergence_checks=False,
        )

    return {"model": mA, "trace": trace, "predictors": predictors, "x_name": "X_A"}

def predict_model_A(bundle, Xnew_df: pd.DataFrame) -> np.ndarray:
    cols = bundle["predictors"]
    missing = [c for c in cols if c not in Xnew_df.columns]
    if missing:
        raise ValueError(f"Model A: missing predictors {missing}. Available: {list(Xnew_df.columns)}")
    X = Xnew_df[cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        bad = [c for i, c in enumerate(cols) if not np.all(np.isfinite(X[:, i]))]
        raise ValueError(f"Model A: non-finite values in {bad}")

    with bundle["model"]:
        pm.set_data({bundle["x_name"]: X})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"],
            var_names=["f"],   # latent mean evaluated on new X
            return_inferencedata=False,
            progressbar=False,
        )
    arr = np.asarray(ppc["f"])
    if arr.ndim == 3:
        ln_mean = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        ln_mean = arr.mean(axis=0)
    else:
        raise ValueError(f"Model A: unexpected PPC shape for 'f': {arr.shape}")
    if ln_mean.shape[0] != X.shape[0]:
        raise ValueError(f"Model A: length mismatch — PPC len {ln_mean.shape[0]} vs X rows {X.shape[0]}")
    return np.exp(ln_mean)

def train_model_B(df_feats_with_predvol: pd.DataFrame,
                  predictors_core: list[str],
                  draws: int = 400,
                  tune: int = 300,
                  trees: int = 60,
                  seed: int = 123):
    preds_B = list(predictors_core)
    if "PredVol_M" not in preds_B:
        preds_B.append("PredVol_M")

    dfB = df_feats_with_predvol.dropna(subset=preds_B + ["FT_fac"]).copy()
    if len(dfB) < 30:
        raise RuntimeError(f"Not enough rows for Model B (have {len(dfB)}, need ≥30).")

    X = np.ascontiguousarray(dfB[preds_B].to_numpy(dtype=np.float64))
    y = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(np.int8).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Model B only has one class present. Need both FT and Fail.")

    with pm.Model() as mB:
        X_B = pm.MutableData("X_B", X)
        f = pmb.BART("f", X_B, y, m=trees)
        pm.Bernoulli("y_obs", logit_p=f, observed=y)

        try:
            trace = pm.sample(
                draws=draws, tune=tune, chains=1, cores=1,
                random_seed=seed, init="adapt_diag",
                progressbar=True, discard_tuned_samples=True,
                compute_convergence_checks=False,
            )
        except Exception:
            trace = pm.sample(
                draws=draws, tune=tune, chains=1, cores=1,
                step=[pmb.PGBART()], random_seed=seed, init="adapt_diag",
                progressbar=True, discard_tuned_samples=True,
                compute_convergence_checks=False,
            )

    return {"model": mB, "trace": trace, "predictors": preds_B, "x_name": "X_B"}

def predict_model_B(bundle, Xnew_df: pd.DataFrame) -> np.ndarray:
    cols = bundle["predictors"]
    missing = [c for c in cols if c not in Xnew_df.columns]
    if missing:
        raise ValueError(f"Model B: missing predictors {missing}. Available: {list(Xnew_df.columns)}")
    X = Xnew_df[cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        bad = [c for i, c in enumerate(cols) if not np.all(np.isfinite(X[:, i]))]
        raise ValueError(f"Model B: non-finite values in {bad}")

    with bundle["model"]:
        pm.set_data({bundle["x_name"]: X})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"],
            var_names=["f"],   # latent logits on new X
            return_inferencedata=False,
            progressbar=False,
        )
    arr = np.asarray(ppc["f"])
    if arr.ndim == 3:
        logits_mean = arr.mean(axis=(0, 1))
    elif arr.ndim == 2:
        logits_mean = arr.mean(axis=0)
    else:
        raise ValueError(f"Model B: unexpected PPC shape for 'f': {arr.shape}")
    if logits_mean.shape[0] != X.shape[0]:
        raise ValueError(f"Model B: length mismatch — PPC len {logits_mean.shape[0]} vs X rows {X.shape[0]}")
    return expit(logits_mean).astype(float)

def _row_cap(df: pd.DataFrame, n: int, y_col: str | None = None):
    if n <= 0 or len(df) <= n:
        return df
    if y_col is None or y_col not in df.columns:
        return df.sample(n=n, random_state=42)
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
        return len(df)

# ---------- TOY TRAIN (for sanity checks) ----------
def _make_toy_df(n=120):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "PMVolM":   rng.lognormal(mean=0.0, sigma=0.8, size=n),
        "PMDolM":   rng.lognormal(mean=1.0, sigma=0.8, size=n),
        "FloatM":   rng.uniform(2, 60, size=n),
        "GapPct":   rng.uniform(2, 60, size=n),
        "ATR":      rng.uniform(0.05, 1.0, size=n),
        "MCapM":    rng.lognormal(mean=2.0, sigma=0.9, size=n),
        "DVolM":    rng.lognormal(mean=1.5, sigma=0.7, size=n),
        "Catalyst": rng.integers(0, 2, size=n),
    })
    # FT label driven a bit by PMDolM/MCap and GapPct
    score = 2.0*np.log(df["PMDolM"]/np.maximum(df["MCapM"],1e-6)+1e-6) + 0.01*df["GapPct"] + 0.5*df["Catalyst"]
    p = 1/(1+np.exp(-(score - score.mean())))
    y = (rng.uniform(0,1,size=n) < p).astype(int)
    df["FT_fac"] = pd.Series(np.where(y==1, "FT", "Fail")).astype("category")
    return featurize(df)

# ---------- MODEL STATUS (always visible) ----------
st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
A_ok = "A_bundle" in st.session_state and st.session_state.get("A_bundle") is not None
B_ok = "B_bundle" in st.session_state and st.session_state.get("B_bundle") is not None
c1.metric("Model A trained?", "Yes" if A_ok else "No")
c2.metric("Model B trained?", "Yes" if B_ok else "No")
if A_ok:
    c3.metric("A predictors", len(st.session_state["A_bundle"]["predictors"]))
else:
    c3.metric("A predictors", 0)

with st.expander("📄 Data preview & mapping (preflight)", expanded=False):
    up_inspect = st.file_uploader("Upload Excel just to inspect", type=["xlsx"], key="inspect_xlsx")
    sheet_name = st.text_input("Sheet name", "PMH BO Merged")
    if up_inspect:
        try:
            raw_ins = read_excel_dynamic(up_inspect, sheet=sheet_name)
            df_all_ins = featurize(raw_ins)
            st.write("Columns detected:", list(df_all_ins.columns))
            need_A = ["ln_DVol"] + A_FEATURES_DEFAULT
            missing_A = [c for c in need_A if c not in df_all_ins.columns]
            n_pass_A = df_all_ins.dropna(subset=need_A).shape[0] if not missing_A else 0
            st.write("Required for A:", need_A)
            st.write("Missing for A:", missing_A)
            st.write("Rows passing A dropna:", int(n_pass_A))
            if "FT_fac" in df_all_ins.columns:
                st.write("FT_fac counts:", df_all_ins["FT_fac"].astype(str).value_counts(dropna=False))
            st.write("Top NaN counts:")
            st.write(df_all_ins.isna().sum().sort_values(ascending=False).head(20))
            st.dataframe(df_all_ins.head(10), use_container_width=True)
        except Exception as e:
            st.exception(e)

# ---------- TRAINING PANEL ----------
with st.expander("⚙️ Train / Load BART models (Python-only)"):
    st.write("Upload your **PMH Database.xlsx** to retrain.")
    up = st.file_uploader("Upload Excel (for training)", type=["xlsx"], accept_multiple_files=False, key="train_xlsx")
    sheet_train = st.text_input("Sheet name for training", "PMH BO Merged", key="train_sheet")

    col_a, col_b = st.columns(2)
    with col_a:
        fast_mode = st.toggle("Fast training mode", value=True, help="Cuts trees/draws and caps rows.")
        trees = st.slider("BART trees per model", 25, 400, 200 if not fast_mode else 75, 25)
        draws = st.slider("MCMC draws", 200, 1500, 800 if not fast_mode else 250, 50)
    with col_b:
        tune  = st.slider("MCMC tune", 200, 1500, 800 if not fast_mode else 250, 50)
        chains = 1 if fast_mode else st.slider("Chains", 1, 2, 2)
        seed  = st.number_input("Random seed", value=42, step=1)

    capA = st.number_input("Max rows for Model A", min_value=0, value=1200 if fast_mode else 0, help="0 = no cap")
    capB = st.number_input("Max rows for Model B (after PredVol_M)", min_value=0, value=600 if fast_mode else 0, help="0 = no cap")

    cta1, cta2 = st.columns(2)
    train_clicked = cta1.button("Train models", use_container_width=True, type="primary")
    toy_clicked   = cta2.button("Train on toy data (sanity check)", use_container_width=True)

    if toy_clicked:
        with st.spinner("Training on toy data…"):
            df_all = _make_toy_df(n=160)

            # A
            dfA = df_all.dropna(subset=["ln_DVol"] + A_FEATURES_DEFAULT).copy()
            A_bundle = train_model_A(
                dfA, A_FEATURES_DEFAULT,
                draws=250 if fast_mode else 600,
                tune=250 if fast_mode else 400,
                trees=75 if fast_mode else 150,
                seed=seed, chains=1
            )

            # PredVol for all rows
            okA = df_all[A_bundle["predictors"]].dropna().index
            preds = predict_model_A(A_bundle, df_all.loc[okA])
            df_all_with_pred = df_all.copy()
            df_all_with_pred.loc[okA, "PredVol_M"] = preds

            # B
            need_B = list(B_FEATURES_CORE) + ["FT_fac"]
            dfB = df_all_with_pred.dropna(subset=need_B).copy()
            dfB["_y"] = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int)
            B_bundle = None
            if len(dfB) >= 30 and dfB["_y"].nunique() == 2:
                B_bundle = train_model_B(
                    df_all_with_pred,
                    B_FEATURES_CORE,
                    draws=250 if fast_mode else 400,
                    tune=250 if fast_mode else 300,
                    trees=60,
                    seed=seed+1,
                )

            # Save to session
            st.session_state["A_bundle"] = A_bundle
            st.session_state["B_bundle"] = B_bundle
            st.session_state["dfA_train"] = dfA.copy()
            st.session_state["dfB_train"] = (dfB.copy() if B_bundle is not None else pd.DataFrame())
            st.session_state["A_predictors"] = A_bundle["predictors"]
            st.session_state["B_predictors"] = (B_bundle["predictors"] if B_bundle else B_FEATURES_CORE)
            st.success(f"Toy training complete. A rows: {len(dfA)}, B rows: {len(dfB)}")
            do_rerun()

    if train_clicked:
        if not up:
            st.error("Upload the Excel file in this training uploader (not the preview one).")
        else:
            try:
                with st.spinner("Training Model A and Model B with BART…"):
                    raw = read_excel_dynamic(up, sheet=sheet_train)
                    df_all = featurize(raw)

                    # -------- Preflight A --------
                    need_A = ["ln_DVol"] + A_FEATURES_DEFAULT
                    missing_A_cols = [c for c in need_A if c not in df_all.columns]
                    if missing_A_cols:
                        st.error(f"Model A cannot train. Missing columns: {missing_A_cols}")
                        st.stop()
                    n_pass_A = len(df_all.dropna(subset=need_A))
                    if n_pass_A < 30:
                        st.error(f"Model A cannot train: only {n_pass_A} rows pass required columns (need ≥30).")
                        st.stop()

                    # A data
                    dfA = df_all.dropna(subset=need_A).copy()
                    dfA = _row_cap(dfA, int(capA)) if capA else dfA

                    # Train A
                    A_bundle = train_model_A(
                        dfA, A_FEATURES_DEFAULT,
                        draws=draws, tune=tune, trees=trees, seed=seed, chains=1
                    )

                    # Predictions for PredVol_M (index-aligned)
                    pred_cols = A_bundle["predictors"]
                    okA = df_all[pred_cols].dropna().index
                    if len(okA) == 0:
                        st.error("No rows pass Model A predictors; cannot proceed to Model B.")
                        st.stop()
                    preds = predict_model_A(A_bundle, df_all.loc[okA])
                    preds = np.asarray(preds).reshape(-1)
                    if preds.shape[0] != len(okA):
                        st.error(f"Internal mismatch: got {preds.shape[0]} preds for {len(okA)} rows.")
                        st.stop()

                    df_all_with_pred = df_all.copy()
                    df_all_with_pred.loc[okA, "PredVol_M"] = pd.Series(preds, index=okA, dtype="float64")

                    # -------- Preflight B --------
                    preds_B_needed = list(B_FEATURES_CORE)
                    if "PredVol_M" not in preds_B_needed:
                        preds_B_needed.append("PredVol_M")
                    need_B = preds_B_needed + ["FT_fac"]
                    missing_B_cols = [c for c in need_B if c not in df_all_with_pred.columns]
                    if missing_B_cols:
                        st.warning(f"Model B skipped (missing columns): {missing_B_cols}")

                    dfB = df_all_with_pred.dropna(subset=need_B).copy() if not missing_B_cols else pd.DataFrame()
                    B_bundle = None
                    if not dfB.empty:
                        y_arr = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int)
                        if len(dfB) >= 30 and y_arr.nunique() == 2:
                            B_bundle = train_model_B(
                                df_all_with_pred,
                                B_FEATURES_CORE,
                                draws=draws if not fast_mode else max(200, draws),
                                tune=tune if not fast_mode else max(200, tune),
                                trees=max(50, trees // (2 if not fast_mode else 3)),
                                seed=seed+1,
                            )
                        else:
                            st.warning(f"Model B skipped: eligible rows {len(dfB)}, classes {y_arr.nunique()} (need ≥30 rows and both classes).")
                    else:
                        st.warning("Model B skipped: no rows eligible after dropna on B predictors and FT_fac.")

                    # Save bundles
                    st.session_state["A_bundle"] = A_bundle
                    st.session_state["B_bundle"] = B_bundle
                    st.session_state["data_hash"] = _hash_df_for_cache(df_all)

                    # Save training frames and predictor lists
                    st.session_state["dfA_train"] = dfA.copy()
                    st.session_state["dfB_train"] = dfB.copy() if B_bundle is not None else pd.DataFrame()
                    st.session_state["A_predictors"] = A_bundle["predictors"]
                    st.session_state["B_predictors"] = (B_bundle["predictors"] if B_bundle else B_FEATURES_CORE)

                    st.success(f"Trained. A rows: {len(dfA)}. B rows: {0 if B_bundle is None else len(dfB)}.")
                    do_rerun()
            except Exception as e:
                st.exception(e)

# ---------- Metrics & Importance ----------
def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2); ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)
def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))
def _accuracy(y_true, y_pred_labels):
    y_true = np.asarray(y_true, dtype=int); y_pred_labels = np.asarray(y_pred_labels, dtype=int)
    return float(np.mean(y_true == y_pred_labels))
def _log_loss(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int); p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1-1e-12)
    return float(-np.mean(y_true*np.log(p) + (1-y_true)*np.log(1-p)))
def _brier(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float); p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y_true)**2))
def _roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int); y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return float("nan")
    cmp = pos[:, None] - neg[None, :]
    greater = np.sum(cmp > 0); equal = np.sum(cmp == 0)
    return float((greater + 0.5*equal) / (n1*n0))
def _fmt(x, nd=3):
    try: return f"{x:.{nd}f}"
    except Exception: return "—"

st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
with st.expander("🔎 Model diagnostics"):
    A_bundle = st.session_state.get("A_bundle")
    B_bundle = st.session_state.get("B_bundle")
    dfA_tr   = st.session_state.get("dfA_train")
    dfB_tr   = st.session_state.get("dfB_train")

    if A_bundle is not None and isinstance(dfA_tr, pd.DataFrame) and not dfA_tr.empty:
        try:
            y_true_log = dfA_tr["ln_DVol"].to_numpy(float)
            y_true_lvl = np.exp(y_true_log)
            y_pred_lvl = predict_model_A(A_bundle, dfA_tr)
            y_pred_log = np.log(np.maximum(y_pred_lvl, 1e-9))

            r2_log   = _r2(y_true_log, y_pred_log)
            rmse_log = _rmse(y_true_log, y_pred_log)
            r2_lvl   = _r2(y_true_lvl, y_pred_lvl)
            rmse_lvl = _rmse(y_true_lvl, y_pred_lvl)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("A — R² (log)", _fmt(r2_log))
            c2.metric("A — RMSE (log)", _fmt(rmse_log))
            c3.metric("A — R² (level)", _fmt(r2_lvl))
            c4.metric("A — RMSE (Millions)", _fmt(rmse_lvl))
        except Exception as e:
            st.warning(f"Model A metrics unavailable: {e}")
    else:
        st.info("Train Model A to see R²/RMSE.")

    if B_bundle is not None and isinstance(dfB_tr, pd.DataFrame) and not dfB_tr.empty:
        try:
            y_true = (dfB_tr["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int).to_numpy()
            proba  = predict_model_B(B_bundle, dfB_tr)
            pred   = (proba >= 0.5).astype(int)

            auc   = _roc_auc(y_true, proba)
            acc   = _accuracy(y_true, pred)
            ll    = _log_loss(y_true, proba)
            brier = _brier(y_true, proba)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("B — ROC AUC", _fmt(auc))
            d2.metric("B — Accuracy (0.5)", _fmt(acc))
            d3.metric("B — Log loss", _fmt(ll))
            d4.metric("B — Brier score", _fmt(brier))
        except Exception as e:
            st.warning(f"Model B metrics unavailable: {e}")
    else:
        st.info("Train Model B to see classification metrics.")

# ---------- Feature Importance (Permutation) ----------
@st.cache_data(show_spinner=False)
def _perm_importance_A(_bundle, df_eval, y_log, features, n_repeats=5, seed=42):
    rng = np.random.default_rng(seed)
    base = _r2(y_log, np.log(np.maximum(predict_model_A(_bundle, df_eval), 1e-9)))
    drops = []
    for ftr in features:
        scores = []
        for _ in range(n_repeats):
            df_s = df_eval.copy()
            df_s[ftr] = rng.permutation(df_s[ftr].values)
            y_pred = np.log(np.maximum(predict_model_A(_bundle, df_s), 1e-9))
            scores.append(_r2(y_log, y_pred))
        drops.append(base - np.mean(scores))
    out = pd.DataFrame({"feature": features, "R2_drop": drops})
    return out.sort_values("R2_drop", ascending=False, ignore_index=True)

@st.cache_data(show_spinner=False)
def _perm_importance_B(_bundle, df_eval, y_true, features, n_repeats=5, seed=42):
    rng = np.random.default_rng(seed)
    base = _roc_auc(y_true, predict_model_B(_bundle, df_eval))
    drops = []
    for ftr in features:
        scores = []
        for _ in range(n_repeats):
            df_s = df_eval.copy()
            df_s[ftr] = rng.permutation(df_s[ftr].values)
            scores.append(_roc_auc(y_true, predict_model_B(_bundle, df_s)))
        drops.append(base - np.mean(scores))
    out = pd.DataFrame({"feature": features, "AUC_drop": drops})
    return out.sort_values("AUC_drop", ascending=False, ignore_index=True)

with st.expander("🧠 Feature importance (permutation)"):
    A_bundle = st.session_state.get("A_bundle")
    B_bundle = st.session_state.get("B_bundle")
    dfA_tr   = st.session_state.get("dfA_train")
    dfB_tr   = st.session_state.get("dfB_train")

    featsA = (st.session_state.get("A_predictors")
              or (A_bundle.get("predictors") if isinstance(A_bundle, dict) else [])
              or [])
    featsB = (st.session_state.get("B_predictors")
              or (B_bundle.get("predictors") if isinstance(B_bundle, dict) else [])
              or [])

    # ---- Model A importance ----
    if A_bundle is not None and isinstance(dfA_tr, pd.DataFrame) and not dfA_tr.empty and len(featsA) > 0:
        try:
            df_eval_A = dfA_tr[featsA]
            y_log_A   = dfA_tr["ln_DVol"].to_numpy(float)
            fiA = _perm_importance_A(A_bundle, df_eval_A, y_log_A, featsA, n_repeats=5)
            st.markdown("**Model A — R² drop when shuffling each feature**")
            st.dataframe(fiA, hide_index=True, use_container_width=True)
        except Exception as e:
            st.info(f"A importance not available: {e}")
    else:
        st.caption("Train Model A to see importance.")
        st.caption(f"• Debug — A_bundle: {A_bundle is not None}, "
                   f"dfA_tr rows: {0 if not isinstance(dfA_tr, pd.DataFrame) else len(dfA_tr)}, "
                   f"featsA: {len(featsA)}")

    # ---- Model B importance ----
    if B_bundle is not None and isinstance(dfB_tr, pd.DataFrame) and not dfB_tr.empty and len(featsB) > 0:
        try:
            y_true_B = (dfB_tr["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int).to_numpy()
            df_eval_B = dfB_tr[featsB]
            fiB = _perm_importance_B(B_bundle, df_eval_B, y_true_B, featsB, n_repeats=5)
            st.markdown("**Model B — AUC drop when shuffling each feature**")
            st.dataframe(fiB, hide_index=True, use_container_width=True)
        except Exception as e:
            st.info(f"B importance not available: {e}")
    else:
        st.caption("Train Model B to see importance.")
        st.caption(f"• Debug — B_bundle: {B_bundle is not None}, "
                   f"dfB_tr rows: {0 if not isinstance(dfB_tr, pd.DataFrame) else len(dfB_tr)}, "
                   f"featsB: {len(featsB)}")

# ---------- UI: Add / Ranking ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

def require_models():
    if "A_bundle" not in st.session_state or st.session_state.get("A_bundle") is None:
        st.warning("Train Model A first (see expander above).")
        st.stop()

with tab_add:
    st.markdown('<div class="section-title">Numeric Context</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.25, 1.25, 1.0])

        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

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
        require_models()
        A_bundle = st.session_state["A_bundle"]
        B_bundle = st.session_state.get("B_bundle")

        pm_dol_m = pm_vol_m * pm_vwap

        row = pd.DataFrame([{
            "PMVolM": pm_vol_m, "PMDolM": pm_dol_m, "FloatM": float_m, "GapPct": gap_pct,
            "ATR": atr_usd, "MCapM": mc_m, "Catalyst": 1 if catalyst_points > 0 else 0,
            "DVolM": np.nan, "FT_fac": "Fail",
        }])
        feats = featurize(row)

        pred_vol_m = float(predict_model_A(A_bundle, feats)[0])

        if B_bundle is not None:
            featsB = feats.copy()
            featsB["PredVol_M"] = pred_vol_m
            ft_prob = float(predict_model_B(B_bundle, featsB)[0])
        else:
            ft_prob = np.nan

        ci68_l = pred_vol_m * math.exp(-1.0 * sigma_ln)
        ci68_u = pred_vol_m * math.exp(+1.0 * sigma_ln)
        ci95_l = pred_vol_m * math.exp(-1.96 * sigma_ln)
        ci95_u = pred_vol_m * math.exp(+1.96 * sigma_ln)

        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            raw_choice = st.session_state.get(f"qual_{crit['name']}", (1,))
            sel = raw_choice[0] if isinstance(raw_choice, tuple) else raw_choice
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(
            combo_pct + news_weight*(1 if catalyst_points>0 else 0)*10 + dilution_weight*(dilution_points)*10, 2
        )
        final_score = max(0.0, min(100.0, final_score))

        pm_float_rot_x  = (pm_vol_m / float_m) if float_m > 0 else 0.0
        pm_pct_of_pred  = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        ft_label = ("FT likely" if (not np.isnan(ft_prob) and ft_prob >= 0.60) else
                    "Toss-up"   if (not np.isnan(ft_prob) and ft_prob >= 0.40) else
                    ("FT unlikely" if not np.isnan(ft_prob) else "B model not trained"))

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

            "FT_Prob": (float(f"{ft_prob:.3f}") if not np.isnan(ft_prob) else np.nan),
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
            f"Saved {ticker} — {ft_label}"
            + (f" ({ft_prob*100:.1f}%)" if not np.isnan(ft_prob) else "")
            + f" · Odds {row_out['Odds']} (Score {row_out['FinalScore']})"
        )
        do_rerun()

# ---------- Ranking tab ----------
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
                df[c] = "" if c in ("Ticker","FT_Label","Level") else np.nan
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
