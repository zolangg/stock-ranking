# Premarket Stock Ranking — Model A (ln DVol regression) + Model B (FT classification)
# ------------------------------------------------------------------------------------
# A: ElasticNetCV on ln(DVol) with conformal CI (fallback to RidgeCV if sklearn missing)
# B: Logistic Regression (Elastic Net) with Platt calibration (requires scikit-learn)
# NumPy 2.x-safe handling for FT_fac to avoid DTypePromotionError.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import numpy as np
import pandas as pd
import streamlit as st

# ----- Try scikit-learn; if missing, we keep Model A via custom RidgeCV and disable Model B -----
SKLEARN_AVAILABLE = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import ElasticNetCV, LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Page + CSS ----------
st.set_page_config(page_title="Premarket Stock Ranking — A: ln(DVol) • B: FT", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking — A: ln(DVol) regression • B: FT probability")

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

    # Catalyst → {0,1}
    out["Catalyst"] = pd.to_numeric(out.get("Catalyst", 0), errors="coerce").fillna(0).ne(0).astype(int)

    # FT label → categorical (NumPy 2.x safe)
    f_raw = out["FT_raw"]
    fl = f_raw.astype(str).str.lower().str.strip()
    is_ft = fl.isin(["ft","1","yes","y","true"])
    ft = pd.Series(np.where(is_ft, "FT", "Fail"), index=out.index, dtype="string")
    ft = ft.mask(f_raw.isna())  # preserve missing
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

# ---- Default feature sets ----
A_FEATURES_DEFAULT = ["ln_pm","ln_pmdol","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst"]
B_FEATURES_CORE    = ["ln_pm","ln_fr","ln_gapf","ln_atr","ln_mcap","Catalyst","ln_pmdol","PredVol_M"]

# ---------- Metrics ----------
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

def _roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int); y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return float("nan")
    cmp = pos[:, None] - neg[None, :]
    greater = np.sum(cmp > 0); equal = np.sum(cmp == 0)
    return float((greater + 0.5*equal) / (n1*n0))

def _accuracy(y_true, y_score, thr=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_hat = (np.asarray(y_score, dtype=float) >= thr).astype(int)
    return float((y_hat == y_true).mean()) if len(y_true) else float("nan")

def _logloss(y_true, y_score, eps=1e-12):
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true*np.log(p) + (1 - y_true)*np.log(1 - p))) if len(y_true) else float("nan")

# ---------- Fallback classes (Model A if sklearn missing) ----------
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
        self.coef_ = self._fit_ridge(Xs, ys, self.alpha_)
        self.y_mean_ = y_mean
        return self
    def predict(self, X):
        Xs = self.std_.transform(X)
        return (Xs @ self.coef_) + self.y_mean_

# ---------- Model A (train/predict) ----------
def _prep_A(df_feats: pd.DataFrame, predictors):
    dfA = df_feats.dropna(subset=["ln_DVol"] + predictors).copy()
    X = dfA[predictors].to_numpy(float)
    y = dfA["ln_DVol"].to_numpy(float)
    return dfA, X, y

def train_model_A(df_feats: pd.DataFrame, predictors, seed: int = 42):
    dfA, X, y = _prep_A(df_feats, predictors)
    if len(dfA) < 20:
        raise RuntimeError(f"Not enough rows to train Model A (have {len(dfA)}, need ≥20).")

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
        # manual CV preds for diagnostics
        y_hat = pipe.predict(X)
        resid = y - y_hat
        coefs = getattr(pipe.named_steps["enet"], "coef_", None)
        alpha_ = getattr(pipe.named_steps["enet"], "alpha_", None)
        l1_ratio_ = getattr(pipe.named_steps["enet"], "l1_ratio_", None)
        model = pipe
        trainer = "ElasticNetCV (sklearn)"
        splits = cv_splits
    else:
        alphas = np.logspace(-3, 3, 41)
        ridge = _RidgeCV(alphas=alphas, cv_splits=5, seed=seed).fit(X, y)
        # conformal residuals via simple in-sample proxy (keeps logic compact here)
        resid = y - ridge.predict(X)
        coefs = ridge.coef_
        alpha_ = ridge.alpha_
        l1_ratio_ = 0.0
        model = ridge
        trainer = "RidgeCV (NumPy fallback)"
        splits = 5

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

# ---------- Model B (train/predict) ----------
def _prep_B(df_feats_with_pred: pd.DataFrame, predictors):
    need = predictors + ["FT_fac"]
    dfB = df_feats_with_pred.dropna(subset=need).copy()
    y = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int).to_numpy()
    X = dfB[predictors].to_numpy(float)
    return dfB, X, y

def train_model_B(df_feats_with_pred: pd.DataFrame, predictors, seed: int = 123):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Model B requires scikit-learn. Install scikit-learn to enable FT classification.")

    need = predictors + ["FT_fac"]
    dfB = df_feats_with_pred.dropna(subset=need).copy()
    y = (dfB["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int).to_numpy()
    X = dfB[predictors].to_numpy(float)
    if len(dfB) < 20 or len(np.unique(y)) < 2:
        raise RuntimeError(f"Not enough labeled rows/classes for Model B (rows={len(dfB)}).")

    skf = StratifiedKFold(n_splits=min(5, max(2, len(dfB)//10)), shuffle=True, random_state=seed)
    base = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            class_weight="balanced", max_iter=20000, random_state=seed
        ))
    ])

    cal = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    cal.fit(X, y)

    proba_cv = cross_val_predict(cal, X, y, cv=skf, method="predict_proba")[:, 1]
    auc = _roc_auc(y, proba_cv)
    acc = _accuracy(y, proba_cv, 0.5)
    ll  = _logloss(y, proba_cv)

    return {"model": cal, "predictors": predictors, "cv_auc": auc, "cv_acc": acc, "cv_logloss": ll, "n_train": len(dfB)}

def predict_model_B(bundle, Xnew_df: pd.DataFrame):
    cols = bundle["predictors"]
    missing = [c for c in cols if c not in Xnew_df.columns]
    if missing:
        raise ValueError(f"Model B: missing predictors {missing}. Available: {list(Xnew_df.columns)}")
    X = Xnew_df[cols].to_numpy(dtype=float)
    return bundle["model"].predict_proba(X)[:, 1]

# ---------- session ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- SIDEBAR ----------
st.sidebar.header("Training options")
cov = st.sidebar.slider("Model A: central CI coverage", 0.50, 0.95, 0.68, 0.01,
                        help="Conformal central coverage for predicted day volume (log-scale, then back-transformed).")
seedA  = st.sidebar.number_input("Seed — Model A", value=42, step=1)
seedB  = st.sidebar.number_input("Seed — Model B", value=123, step=1)

st.sidebar.header("Feature sets")
use_defaults_A = st.sidebar.toggle("Use default A features", value=True)
custom_A = st.sidebar.text_input(
    "Custom A features (comma sep)",
    "ln_pm, ln_pmdol, ln_fr, ln_gapf, ln_atr, ln_mcap, Catalyst"
)
A_predictors = A_FEATURES_DEFAULT if use_defaults_A else [c.strip() for c in custom_A.split(",") if c.strip()]

use_defaults_B = st.sidebar.toggle("Use default B features", value=True)
custom_B = st.sidebar.text_input(
    "Custom B features (comma sep)",
    "ln_pm, ln_fr, ln_gapf, ln_atr, ln_mcap, Catalyst, ln_pmdol, PredVol_M"
)
B_predictors = B_FEATURES_CORE if use_defaults_B else [c.strip() for c in custom_B.split(",") if c.strip()]

# ---------- TRAINING PANEL ----------
with st.expander("⚙️ Train / Load Models"):
    if not SKLEARN_AVAILABLE:
        st.info("scikit-learn not found — Model A uses RidgeCV fallback; **Model B is disabled**. "
                "Add `scikit-learn==1.5.1` to requirements.txt to enable ElasticNet (A) and Logistic (B).")

    st.write("Upload your **PMH Database.xlsx** (sheet: _PMH BO Merged_) to (re)train.")
    up = st.file_uploader("Upload Excel", type=["xlsx"], accept_multiple_files=False, key="train_xlsx")
    sheet_train = st.text_input("Sheet name", "PMH BO Merged", key="train_sheet")

    if st.button("Train A & B", use_container_width=True, type="primary"):
        if not up:
            st.error("Upload the Excel file first.")
        else:
            try:
                with st.spinner("Training models…"):
                    raw = read_excel_dynamic(up, sheet=sheet_train)
                    df_all = featurize(raw)

                    # ==== Model A ====
                    need_A = ["ln_DVol"] + A_predictors
                    dfA = df_all.dropna(subset=need_A).copy()
                    if len(dfA) < 20:
                        st.error(f"Not enough rows for Model A after filtering: {len(dfA)} (need ≥20).")
                        st.stop()
                    A_bundle = train_model_A(df_all, A_predictors, seed=seedA)

                    # Attach A predictions (PredVol_M) for B
                    okA = df_all[A_bundle["predictors"]].dropna().index
                    predsA, ci_l, ci_u = predict_model_A(A_bundle, df_all.loc[okA, A_bundle["predictors"]],
                                                         central_coverage=cov)
                    df_all_with_pred = df_all.copy()
                    df_all_with_pred.loc[okA, "PredVol_M"]    = pd.Series(predsA, index=okA, dtype="float64")
                    df_all_with_pred.loc[okA, "PredVol_CI_L"] = pd.Series(ci_l,  index=okA, dtype="float64")
                    df_all_with_pred.loc[okA, "PredVol_CI_U"] = pd.Series(ci_u,  index=okA, dtype="float64")

                    # Save A artifacts
                    st.session_state["A_bundle"] = A_bundle
                    st.session_state["df_all_full"] = df_all_with_pred.copy()
                    st.session_state["dfA_train"] = dfA.copy()
                    st.session_state["A_predictors"] = A_bundle["predictors"]
                    st.session_state["data_hash"] = _hash_df_for_cache(df_all)

                    # Basic A diagnostics (train fit, log-scale)
                    y_true_log = dfA["ln_DVol"].to_numpy(float)
                    y_pred_log = A_bundle["model"].predict(dfA[A_bundle["predictors"]].to_numpy(float))
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("A: Train R² (log)",  f"{_r2(y_true_log, y_pred_log):.3f}")
                    with c2: st.metric("A: RMSE (log)",      f"{_rmse(y_true_log, y_pred_log):.3f}")
                    with c3: st.metric("A: MAE (log)",       f"{_mae(y_true_log, y_pred_log):.3f}")

                    # Coefs
                    coefs = A_bundle.get("coefs", None)
                    if coefs is not None:
                        coef_df = pd.DataFrame({"feature": A_bundle["predictors"], "coef": coefs})
                        coef_df = coef_df.sort_values("coef", ascending=False)
                        st.caption("A: Elastic Net coefficients (ln-scale):")
                        st.dataframe(coef_df, hide_index=True, use_container_width=True)

                    # ==== Model B (optional) ====
                    B_bundle = None
                    if SKLEARN_AVAILABLE:
                        need_B = B_predictors + ["FT_fac"]
                        dfB = df_all_with_pred.dropna(subset=need_B).copy()
                        if len(dfB) >= 20 and dfB["FT_fac"].nunique(dropna=True) >= 2:
                            # Diagnostics for B eligibility
                            dfB_check = df_all_with_pred[need_B].copy()
                            missing_cols = [c for c in need_B if c not in df_all_with_pred.columns]
                            st.caption(f"B features required: {need_B}")
                            if missing_cols:
                                st.warning(f"Model B skip reason: missing columns {missing_cols}")

                            dfB_nonan = dfB_check.dropna()
                            st.caption(f"B: non-missing rows after filtering: {len(dfB_nonan)}")

                            if "FT_fac" in df_all_with_pred.columns:
                                label_counts = df_all_with_pred["FT_fac"].value_counts(dropna=False)
                                st.caption(f"B: label distribution (including NaN):\n{label_counts.to_dict()}")

                            if len(dfB_nonan) > 0:
                                y_dbg = (dfB_nonan["FT_fac"].astype(str).str.lower().isin(["ft","1","yes","y","true"])).astype(int)
                                st.caption(f"B: eligible rows by class: {{0: {(y_dbg==0).sum()}, 1: {(y_dbg==1).sum()}}}")
                            try:
                                B_bundle = train_model_B(df_all_with_pred, B_predictors, seed=seedB)
                                st.session_state["B_bundle"] = B_bundle
                                st.session_state["B_predictors"] = B_bundle["predictors"]
                                c1, c2, c3 = st.columns(3)
                                with c1: st.metric("B: CV ROC-AUC",  f"{B_bundle['cv_auc']:.3f}")
                                with c2: st.metric("B: CV Accuracy", f"{B_bundle['cv_acc']:.3f}")
                                with c3: st.metric("B: CV LogLoss",  f"{B_bundle['cv_logloss']:.3f}")
                                st.success(f"Model B trained on {B_bundle['n_train']} rows.")
                            except Exception as e:
                                st.warning(f"Model B skipped: {e}")
                                st.session_state["B_bundle"] = None
                        else:
                            st.info("Model B skipped: insufficient labeled rows or only one class.")
                            st.session_state["B_bundle"] = None
                    else:
                        st.info("Model B disabled (scikit-learn not available).")

                    st.success(f"✅ Trained A ({A_bundle['trainer']}, rows={A_bundle['n_train']})"
                               + (" • Trained B" if st.session_state.get("B_bundle") else " • B skipped"))

            except Exception as e:
                st.exception(e)

# ---------- UI: Add / Ranking ----------
tab_add, tab_rank =
