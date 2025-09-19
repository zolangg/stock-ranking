# Premarket Stock Ranking — ln(DVol) regression (Elastic Net w/ fallback) + NumPy 2.x-safe FT label
# -------------------------------------------------------------------------------------------------
# - Primary model: ElasticNetCV on ln(DVol) with conformal CI (requires scikit-learn)
# - Fallback model: custom RidgeCV (NumPy) with same UI/outputs if scikit-learn isn't installed
# - Fixes DTypePromotionError by building FT_fac via pandas (no mixed float/string in np.where)

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import numpy as np
import pandas as pd
import streamlit as st

# ----- Try scikit-learn; if missing, we use NumPy fallback -----
SKLEARN_AVAILABLE = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import KFold, cross_val_predict
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Page + light CSS ----------
st.set_page_config(page_title="Premarket Stock Ranking — ln(DVol) regression", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking — ln(DVol) regression")

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
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _hash_df_for_cache(df: pd.DataFrame) -> int:
    try:
        return int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        return len(df)

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

    # Catalyst → {0,1} (robust to dtype quirks)
    out["Catalyst"] = pd.to_numeric(out.get("Catalyst", 0), errors="coerce").fillna(0).ne(0).astype(int)

    # FT label → categorical (NumPy 2.x safe; no mixing NaN + strings)
    f_raw = out["FT_raw"]
    fl = f_raw.astype(str).str.lower().str.strip()
    is_ft = fl.isin(["ft","1","yes","y","true"])

    ft = pd.Series(np.where(is_ft, "FT", "Fail"), index=out.index, dtype="string")
    ft = ft.mask(f_raw.isna())  # preserve original missingness
    out["FT_fac"] = ft.astype("category")

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
    out["Catalyst"] = pd.to_numeric(out.get("Catalyst", 0), errors="coerce").fillna(0).ne(0).astype(int)
    out["FT_fac"]   = out["FT_fac"].astype("category")
    return out

A_FEATURES_DEFAULT = ["ln_pm","ln_pmdol","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst"]

# ---------- Metrics helpers ----------
def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2); ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def _mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# ---------- Fallback: tiny Standardizer + RidgeCV (NumPy) ----------
class _Std:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class _RidgeCV:
    def __init__(self, alphas, cv_splits=5, seed=42):
        self.alphas = np.array(sorted(set(alphas)), dtype=float)
        self.cv_splits = cv_splits
        self.seed = seed
    def _fit_ridge(self, X, y, alpha):
        XtX = X.T @ X
        nfeat = XtX.shape[0]
        A = XtX + alpha * np.eye(nfeat)
        coef = np.linalg.solve(A, X.T @ y)
        return coef
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.std_ = _Std().fit(X)
        Xs = self.std_.transform(X)
        y_mean = y.mean()
        ys = y - y_mean

        k = max(2, min(self.cv_splits, len(y)//10 if len(y)>=20 else 2))
        idx = np.arange(len(y))
        rng.shuffle(idx)
        folds = np.array_split(idx, k)

        mse_per_alpha = []
        for a in self.alphas:
            mses = []
            for i in range(k):
                val = folds[i]
                tr = np.concatenate([folds[j] for j in range(k) if j!=i])
                coef = self._fit_ridge(Xs[tr], ys[tr], a)
                yhat = Xs[val] @ coef
                mses.append(np.mean((yhat - ys[val])**2))
            mse_per_alpha.append(np.mean(mses))
        best_idx = int(np.argmin(mse_per_alpha))
        self.alpha_ = float(self.alphas[best_idx])

        # Fit on full data
        self.coef_ = self._fit_ridge(Xs, ys, self.alpha_)
        self.y_mean_ = y_mean
        return self
    def predict(self, X):
        Xs = self.std_.transform(X)
        return (Xs @ self.coef_) + self.y_mean_

# ---------- Model A trainers ----------
def _prep_A(df_feats: pd.DataFrame, predictors):
    dfA = df_feats.dropna(subset=["ln_DVol"] + predictors).copy()
    X = dfA[predictors].to_numpy(float)
    y = dfA["ln_DVol"].to_numpy(float)
    return dfA, X, y

def train_model_A(df_feats: pd.DataFrame, predictors, seed: int = 42):
    dfA, X, y = _prep_A(df_feats, predictors)
    if len(dfA) < 20:
        raise RuntimeError(f"Not enough rows to train model (have {len[dfA]}, need ≥20).")

    if SKLEARN_AVAILABLE:
        cv_splits = min(5, max(2, len(dfA)//10))
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        pipe = Pipeline([
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("enet", ElasticNetCV(
                l1_ratio=[0.05,0.25,0.5,0.75,0.95],
                alphas=None, cv=cv, n_alphas=200, random_state=seed, max_iter=20000
            ))
        ])
        pipe.fit(X, y)
        y_cv = cross_val_predict(pipe, X, y, cv=cv, method="predict")
        resid = y - y_cv
        coefs = getattr(pipe.named_steps["enet"], "coef_", None)
        alpha_ = getattr(pipe.named_steps["enet"], "alpha_", None)
        l1_ratio_ = getattr(pipe.named_steps["enet"], "l1_ratio_", None)
        model = pipe
        trainer = "ElasticNetCV (scikit-learn)"
        splits = cv_splits
    else:
        alphas = np.logspace(-3, 3, 41)
        ridge = _RidgeCV(alphas=alphas, cv_splits=5, seed=seed).fit(X, y)
        # manual CV predictions for conformal
        k = 5 if len(dfA) >= 25 else 2
        idx = np.arange(len(y))
        np.random.default_rng(seed).shuffle(idx)
        folds = np.array_split(idx, k)
        y_cv = np.zeros_like(y, dtype=float)
        for i in range(k):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j!=i])
            rcv = _RidgeCV(alphas=alphas, cv_splits=k, seed=seed)
            rcv.fit(X[tr], y[tr])
            y_cv[val] = rcv.predict(X[val])
        resid = y - y_cv
        coefs = ridge.coef_
        alpha_ = ridge.alpha_
        l1_ratio_ = 0.0
        model = ridge
        trainer = "RidgeCV (NumPy fallback)"
        splits = k

    return {
        "model": model,
        "predictors": predictors,
        "resid": resid,
        "n_train": len(dfA),
        "cv_splits": splits,
        "coefs": coefs,
        "alpha_": alpha_,
        "l1_ratio_": l1_ratio_,
        "trainer": trainer,
    }

def predict_model_A(bundle, Xnew_df: pd.DataFrame, central_coverage: float = 0.68):
    cols = bundle["predictors"]
    missing = [c for c in cols if c not in Xnew_df.columns]
    if missing:
        raise ValueError(f"Model A: missing predictors {missing}. Available: {list(Xnew_df.columns)}")
    X = Xnew_df[cols].to_numpy(dtype=float)

    ln_pred = bundle["model"].predict(X)

    # Conformal interval on log scale (symmetric abs residual quantile)
    alpha = max(0.0, min(1.0, 1.0 - central_coverage))
    r = np.abs(np.asarray(bundle["resid"], dtype=float))
    q = float(np.quantile(r, 1 - alpha)) if len(r) > 0 else 0.0

    pred = np.exp(ln_pred)
    lo = np.exp(ln_pred - q)
    hi = np.exp(ln_pred + q)
    return pred, lo, hi

# ---------- session ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- SIDEBAR ----------
st.sidebar.header("Training")
cov = st.sidebar.slider("Central CI coverage", 0.50, 0.95, 0.68, 0.01,
                        help="Conformal central coverage for predicted day volume (log-scale, then back-transformed).")
seed  = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.header("Feature set")
use_defaults = st.sidebar.toggle("Use default features", value=True)
custom_feats = st.sidebar.text_input(
    "Custom features (comma sep)",
    "ln_pm, ln_pmdol, ln_fr, ln_gapf, ln_atr, ln_mcap, Catalyst"
)
if use_defaults:
    A_predictors = A_FEATURES_DEFAULT
else:
    A_predictors = [c.strip() for c in custom_feats.split(",") if c.strip()]

# ---------- TRAINING PANEL ----------
with st.expander("⚙️ Train / Load ln(DVol) model"):
    if not SKLEARN_AVAILABLE:
        st.info("scikit-learn not found — falling back to built-in RidgeCV (L2). "
                "To enable ElasticNet, add `scikit-learn==1.5.1` to requirements.txt.")

    st.write("Upload your **PMH Database.xlsx** (sheet: _PMH BO Merged_) to retrain.")
    up = st.file_uploader("Upload Excel", type=["xlsx"], accept_multiple_files=False, key="train_xlsx")
    sheet_train = st.text_input("Sheet name", "PMH BO Merged", key="train_sheet")

    if st.button("Train model", use_container_width=True, type="primary"):
        if not up:
            st.error("Upload the Excel file first.")
        else:
            try:
                with st.spinner("Training ln(DVol) model…"):
                    raw = read_excel_dynamic(up, sheet=sheet_train)
                    df_all = featurize(raw)

                    need_A = ["ln_DVol"] + A_predictors
                    dfA = df_all.dropna(subset=need_A).copy()
                    if len(dfA) < 20:
                        st.error(f"Not enough rows after filtering: {len(dfA)} (need ≥20).")
                        st.stop()

                    A_bundle = train_model_A(df_all, A_predictors, seed=seed)

                    # Predictions for PredVol_M (index-aligned)
                    okA = df_all[A_bundle["predictors"]].dropna().index
                    if len(okA) == 0:
                        st.error("No rows pass Model A predictors; cannot compute predictions.")
                        st.stop()

                    preds, ci_l, ci_u = predict_model_A(A_bundle, df_all.loc[okA, A_bundle["predictors"]],
                                                        central_coverage=cov)
                    df_all_with_pred = df_all.copy()
                    df_all_with_pred.loc[okA, "PredVol_M"]    = pd.Series(preds, index=okA, dtype="float64")
                    df_all_with_pred.loc[okA, "PredVol_CI_L"] = pd.Series(ci_l,   index=okA, dtype="float64")
                    df_all_with_pred.loc[okA, "PredVol_CI_U"] = pd.Series(ci_u,   index=okA, dtype="float64")

                    # Save bundles + frames
                    st.session_state["A_bundle"]   = A_bundle
                    st.session_state["data_hash"]  = _hash_df_for_cache(df_all)
                    st.session_state["dfA_train"]  = dfA.copy()
                    st.session_state["df_all_full"]= df_all_with_pred.copy()
                    st.session_state["A_predictors"] = A_bundle["predictors"]

                    # Basic diagnostics (train fit on log-scale)
                    y_true_log = dfA["ln_DVol"].to_numpy(float)
                    X_train = dfA[A_bundle["predictors"]].to_numpy(float)
                    y_pred_log = A_bundle["model"].predict(X_train)

                    st.success(f"Trained ({A_bundle['trainer']}). Rows: {len(dfA)} • CV splits: {A_bundle['cv_splits']}")
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Train R² (log)",  f"{_r2(y_true_log, y_pred_log):.3f}")
                    with c2: st.metric("Train RMSE (log)", f"{_rmse(y_true_log, y_pred_log):.3f}")
                    with c3: st.metric("Train MAE (log)",  f"{_mae(y_true_log, y_pred_log):.3f}")

                    # Coeff summary
                    coefs = A_bundle.get("coefs", None)
                    if coefs is not None:
                        coef_df = pd.DataFrame({"feature": A_bundle["predictors"], "coef": coefs})
                        coef_df = coef_df.sort_values("coef", ascending=False)
                        st.caption("Model coefficients (ln-scale):")
                        st.dataframe(coef_df, hide_index=True, use_container_width=True)

            except Exception as e:
                st.exception(e)

# ---------- UI: Add / Ranking ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

def require_A():
    if "A_bundle" not in st.session_state or st.session_state.get("A_bundle") is None:
        st.warning("Train the model first (see expander above).")
        st.stop()

with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        left, right = st.columns([1.0, 2.0])

        with left:
            ticker = st.text_input("Ticker", "").strip().upper()
            catalyst_yes = st.toggle("Catalyst present?", value=False,
                                     help="News/pr catalyst flag used as binary feature.")

        with right:
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                float_m  = st.number_input("Public Float (Millions)",   min_value=0.0, value=0.0, step=0.01, format="%.2f")
                gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            with rcol2:
                atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                pm_vol_m = st.number_input("Premarket Volume (Millions shares)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                pm_dol_m = st.number_input("Premarket Dollar Volume (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Add / Predict", use_container_width=True)

    if submitted and ticker:
        require_A()
        A_bundle = st.session_state["A_bundle"]

        row = pd.DataFrame([{
            "PMVolM": pm_vol_m, "PMDolM": pm_dol_m, "FloatM": float_m, "GapPct": gap_pct,
            "ATR": atr_usd, "MCapM": mc_m, "Catalyst": 1 if catalyst_yes else 0,
            "DVolM": np.nan, "FT_fac": "Fail",
        }])
        feats = featurize(row)

        try:
            pred_vol_m, ci_l, ci_u = predict_model_A(
                A_bundle, feats[A_bundle["predictors"]], central_coverage=cov
            )
            pred_vol_m = float(pred_vol_m[0]); ci_l = float(ci_l[0]); ci_u = float(ci_u[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        pm_float_rot_x  = (pm_vol_m / float_m) if float_m > 0 else 0.0
        pm_pct_of_pred  = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        row_out = {
            "Ticker": ticker,
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI_L": round(ci_l, 2),
            "PredVol_CI_U": round(ci_u, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "_MCap_M": mc_m, "_ATR_$": atr_usd, "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m,
            "_Float_M": float_m, "_Gap_%": gap_pct, "_Catalyst": 1 if catalyst_yes else 0,
        }

        st.session_state.rows.append(row_out)
        st.session_state.last = row_out
        st.session_state.flash = f"Saved {ticker} — Predicted daily volume: {pred_vol_m:.2f}M (CI {ci_l:.2f}–{ci_u:.2f})"

with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "PredVol_M" in df.columns:
            df = df.sort_values("PredVol_M", ascending=False)
        df = df.reset_index(drop=True)

        cols_to_show = [
            "Ticker",
            "PredVol_M","PredVol_CI_L","PredVol_CI_U",
            "PM_%_of_Pred","PM$ / MC_%","PM_FloatRot_x"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c == "Ticker" else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI_L": st.column_config.NumberColumn("Pred Vol CI Low (M)",  format="%.2f"),
                "PredVol_CI_U": st.column_config.NumberColumn("Pred Vol CI High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation ×", format="%.3f"),
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
