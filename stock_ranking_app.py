import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import numpy as np
import pandas as pd
import streamlit as st

# PyMC + BART
import pymc as pm
import pymc_bart as pmb
from scipy.special import expit

# ---------- Page + light CSS ----------
st.set_page_config(page_title="Premarket Stock Ranking — BART (fast)", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking — PyMC BART (fast)")

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

def _hash_df_for_cache(df: pd.DataFrame) -> int:
    try:
        return int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        return len(df)

# ---------- session ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- sidebar ----------
st.sidebar.header("Training profile (BART)")
fast_mode = st.sidebar.toggle("Fast profile", value=True, help="Reduces trees/draws and caps rows.")
trees = st.sidebar.slider("BART trees", 25, 300, 70 if fast_mode else 160, 5)
draws = st.sidebar.slider("MCMC draws", 100, 1500, 220 if fast_mode else 700, 20)
tune  = st.sidebar.slider("MCMC tune",  100, 1500, 220 if fast_mode else 700, 20)
seed  = st.sidebar.number_input("Random seed", value=42, step=1)
capA  = st.sidebar.number_input("Max rows for Model A", min_value=0, value=(1200 if fast_mode else 0), help="0 = no cap")
capB  = st.sidebar.number_input("Max rows for Model B", min_value=0, value=(700 if fast_mode else 0),   help="0 = no cap")

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider(
    "Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
    help="Used for CI bands around predicted day volume."
)

# Optional (slow)
compute_importance = st.sidebar.toggle("Compute permutation importance (slow)", value=False)

# ---------- data mapping + features ----------
def read_excel_dynamic(file, sheet="PMH BO Merged") -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")
    nms = list(df.columns)

    def _pick_col(nms, patterns):
        lnms = [x.lower() for x in nms]
        for pat in patterns:
            for i, nm in enumerate(lnms):
                if pd.Series([nm]).str.contains(pat, regex=True, na=False).iloc[0]:
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

    # FT label → categorical (keep NaN if absent)
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

# ---- Predictors
A_FEATURES_DEFAULT = ["ln_pm","ln_pmdol","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst"]
B_FEATURES_CORE    = ["ln_pm","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst","ln_pmdol","PredVol_M"]

# ---------- helper to robustly average over draws/chains ----------
def _mean_over_draws(arr: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(arr)
    for axis in range(arr.ndim):
        if arr.shape[axis] == n_rows:
            axes = tuple(i for i in range(arr.ndim) if i != axis)
            mu = arr.mean(axis=axes) if axes else arr
            mu = np.asarray(mu)
            if mu.ndim == 0:
                return np.full(n_rows, float(mu))
            return mu
    if n_rows == 1 and arr.ndim >= 1:
        return np.array([float(arr.mean())], dtype=float)
    scalar = float(arr.mean())
    return np.full(n_rows, scalar, dtype=float)

# ---------- BART training ----------
def train_model_A(df_feats: pd.DataFrame, predictors: list[str],
                  draws=220, tune=220, trees=70, seed=42):
    dfA = df_feats.dropna(subset=["ln_DVol"] + predictors).copy()
    if len(dfA) < 30:
        raise RuntimeError("Not enough rows to train Model A (need ≥30).")

    X = np.ascontiguousarray(dfA[predictors].to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(dfA["ln_DVol"].to_numpy(dtype=np.float64))

    with pm.Model() as mA:
        X_A = pm.MutableData("X_A", X)
        f = pmb.BART("f", X_A, y, m=trees)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=f, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=draws, tune=tune, chains=1, cores=1,
            random_seed=seed,
            init="adapt_diag",
            progressbar=True,
            discard_tuned_samples=True,
            compute_convergence_checks=False,
        )

    return {"model": mA, "trace": trace, "predictors": predictors, "x_name": "X_A", "n_train": X.shape[0]}

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
        pm.set_data({bundle["x_name"]: np.ascontiguousarray(X, dtype=np.float64)})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"], var_names=["f"], return_inferencedata=False, progressbar=False
        )

    arr = np.asarray(ppc["f"])
    ln_mean = _mean_over_draws(arr, X.shape[0])
    if ln_mean.shape[0] != X.shape[0]:
        raise ValueError(
            f"Model A: length mismatch — PPC len {ln_mean.shape[0]} vs X rows {X.shape[0]} "
            f"(train_n={bundle.get('n_train','?')})."
        )
    return np.exp(ln_mean)

def train_model_B(df_feats_with_predvol: pd.DataFrame,
                  predictors_core: list[str],
                  draws: int = 220,
                  tune: int = 220,
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

        trace = pm.sample(
            draws=draws, tune=tune, chains=1, cores=1,
            random_seed=seed,
            init="adapt_diag",
            progressbar=True,
            discard_tuned_samples=True,
            compute_convergence_checks=False,
        )

    return {"model": mB, "trace": trace, "predictors": preds_B, "x_name": "X_B", "n_train": X.shape[0]}

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
        pm.set_data({bundle["x_name"]: np.ascontiguousarray(X, dtype=np.float64)})
        ppc = pm.sample_posterior_predictive(
            bundle["trace"], var_names=["f"], return_inferencedata=False, progressbar=False
        )
    arr = np.asarray(ppc["f"])
    logits_mean = _mean_over_draws(arr, X.shape[0])
    if logits_mean.shape[0] != X.shape[0]:
        raise ValueError(
            f"Model B: length mismatch — PPC len {logits_mean.shape[0]} vs X rows {X.shape[0]} "
            f"(train_n={bundle.get('n_train','?')})."
        )
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

# ---------- TRAINING PANEL ----------
with st.expander("⚙️ Train / Load BART models"):
    st.write("Upload your **PMH Database.xlsx** (sheet: _PMH BO Merged_) to retrain.")
    up = st.file_uploader("Upload Excel", type=["xlsx"], accept_multiple_files=False, key="train_xlsx")
    sheet_train = st.text_input("Sheet name", "PMH BO Merged", key="train_sheet")

    if st.button("Train models", use_container_width=True, type="primary"):
        if not up:
            st.error("Upload the Excel file first.")
        else:
            try:
                with st.spinner("Training Model A and Model B with BART (fast)…"):
                    raw = read_excel_dynamic(up, sheet=sheet_train)
                    df_all = featurize(raw)

                    # --- A data ---
                    need_A = ["ln_DVol"] + A_FEATURES_DEFAULT
                    dfA = df_all.dropna(subset=need_A).copy()
                    dfA = _row_cap(dfA, int(capA)) if capA else dfA
                    if len(dfA) < 30:
                        st.error(f"Not enough rows for A after filtering: {len(dfA)}")
                        st.stop()

                    # Train A
                    A_bundle = train_model_A(dfA, A_FEATURES_DEFAULT, draws=draws, tune=tune, trees=trees, seed=seed)

                    # Predictions for PredVol_M (index-aligned)
                    pred_cols = A_bundle["predictors"]
                    okA = df_all[pred_cols].dropna().index
                    if len(okA) == 0:
                        st.error("No rows pass Model A predictors; cannot proceed to Model B.")
                        st.stop()
                    preds = predict_model_A(A_bundle, df_all.loc[okA, pred_cols])
                    preds = np.asarray(preds).reshape(-1)
                    df_all_with_pred = df_all.copy()
                    df_all_with_pred.loc[okA, "PredVol_M"] = pd.Series(preds, index=okA, dtype="float64")

                    # --- B data ---
                    preds_B_needed = list(B_FEATURES_CORE)
                    if "PredVol_M" not in preds_B_needed:
                        preds_B_needed.append("PredVol_M")
                    need_B = preds_B_needed + ["FT_fac"]
                    dfB = df_all_with_pred.dropna(subset=need_B).copy()
                    if len(dfB) < 30:
                        st.warning(f"Model B skipped: only {len(dfB)} eligible rows (need ≥30).")
                        B_bundle = None
                        dfB_cap = pd.DataFrame()
                    else:
                        dfB["_y"] = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int)
                        dfB_cap = _row_cap(dfB, int(capB), y_col="_y") if capB else dfB
                        B_bundle = train_model_B(
                            df_all_with_pred, B_FEATURES_CORE,
                            draws=draws, tune=tune, trees=max(50, trees - 10), seed=seed+1
                        )

                    # Save bundles + frames
                    st.session_state["A_bundle"] = A_bundle
                    st.session_state["B_bundle"] = B_bundle
                    st.session_state["data_hash"] = _hash_df_for_cache(df_all)
                    st.session_state["dfA_train"] = dfA.copy()
                    st.session_state["dfB_train"] = dfB_cap.copy()
                    st.session_state["A_predictors"] = A_bundle["predictors"]
                    st.session_state["B_predictors"] = (B_bundle["predictors"] if B_bundle else B_FEATURES_CORE)

                    st.success(f"Trained A (rows: {len(dfA)}). " + ("Trained B." if B_bundle else "B skipped."))

            except Exception as e:
                st.exception(e)

# ---------- Metrics (lightweight) ----------
def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def _roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return float("nan")
    cmp = pos[:, None] - neg[None, :]
    greater = np.sum(cmp > 0); equal = np.sum(cmp == 0)
    return float((greater + 0.5*equal) / (n1*n0))

st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
with st.expander("🔎 Quick diagnostics"):
    run_diag = st.checkbox("Run diagnostics now", value=False, help="Avoids heavy re-compute on every rerun.")

    A_bundle = st.session_state.get("A_bundle")
    B_bundle = st.session_state.get("B_bundle")
    dfA_tr   = st.session_state.get("dfA_train")
    dfB_tr   = st.session_state.get("dfB_train")

    if not run_diag:
        st.caption("Diagnostics are idle. Tick the box to compute.")
    else:
        if A_bundle is not None and isinstance(dfA_tr, pd.DataFrame) and not dfA_tr.empty:
            try:
                y_true_log = dfA_tr["ln_DVol"].to_numpy(float)
                y_pred_lvl = predict_model_A(A_bundle, dfA_tr)
                y_pred_log = np.log(np.maximum(y_pred_lvl, 1e-9))
                st.metric("Model A — R² (log)", f"{_r2(y_true_log, y_pred_log):.3f}")
                st.metric("Model A — RMSE (log)", f"{_rmse(y_true_log, y_pred_log):.3f}")
            except Exception as e:
                st.warning(f"Model A metrics unavailable: {e}")
        else:
            st.info("Train Model A to see diagnostics.")

        if B_bundle is not None and isinstance(dfB_tr, pd.DataFrame) and not dfB_tr.empty:
            try:
                y_true = (dfB_tr["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int).to_numpy()
                proba  = predict_model_B(B_bundle, dfB_tr)
                st.metric("Model B — ROC AUC", f"{_roc_auc(y_true, proba):.3f}")
            except Exception as e:
                st.warning(f"Model B metrics unavailable: {e}")
        elif st.session_state.get("B_bundle") is None:
            st.info("Model B skipped (insufficient rows).")

# ---------- Feature Importance (Permutation) — gated (slow) ----------
@st.cache_data(show_spinner=False)
def _perm_importance_A(_bundle, df_eval, y_log, features, n_repeats=3, seed=42):
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
def _perm_importance_B(_bundle, df_eval
