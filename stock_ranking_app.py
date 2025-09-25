# app.py — Premarket FT Ranking with LASSO feature selection (no day-volume)
# -----------------------------------------------------------------------------
# Pipeline:
#   1) Read workbook (Legend-aware).
#   2) Build FT features: ln_gapf, ln_atr, ln_mcap, ln1p_rvol, ln1p_pmvol,
#      ln1p_pmmc, ln1p_si, catalyst, ln_float  (float yes; no rotation; no pm-fraction).
#   3) Standardize on train.
#   4) Logistic LASSO (coordinate descent) with 5-fold CV to pick λ (and features).
#   5) Refit weighted L2 logistic on the selected features for stability.
#   6) Predict & grade: only Ticker, Grade, FT Probability % are displayed/exported.
#
# Notes:
#   • No sklearn; custom L1 path with CV.
#   • Catalyst input UI sits under Premarket $ Volume.
#   • Deletable rows, CSV & Markdown export.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math, re, random
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============ Page ============
st.set_page_config(page_title="Premarket FT Ranking — LASSO", layout="wide")
st.title("Premarket FT Ranking — LASSO-selected Features")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============ Session ============
if "FT" not in st.session_state: st.session_state.FT = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if "TRAIN_PREDS" not in st.session_state: st.session_state.TRAIN_PREDS = np.array([])

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ============ Helpers ============
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip().replace(" ", "").replace("’","").replace("'","")
    if s == "": return None
    if "," in s and "." not in s: s = s.replace(",", ".")
    else: s = s.replace(",", "")
    try: return float(s)
    except: return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"
    s = st.text_input(label, fmt.format(value), key=key, help=help)
    v = _parse_local_float(s)
    if v is None: return float(value)
    v = max(min_value, v)
    if max_value is not None: v = min(max_value, v)
    return float(v)

def _nz(x, fallback=0.0):
    try:
        xx = float(x)
        return xx if np.isfinite(xx) else float(fallback)
    except:
        return float(fallback)

def _safe_log(x: float, eps: float = 1e-8) -> float:
    return math.log(max(_nz(x, 0.0), eps))

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
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ============ Legend mapping ============
_DEF = {
    "FT":     ["FT"],
    "GAP":    ["Gap %","Gap"],
    "ATR":    ["Daily ATR","ATR $","ATR","ATR (USD)","ATR$"],
    "RVOL":   ["RVOL @ BO","RVOL","Relative Volume"],
    "PMVOL":  ["PM Vol (M)","Premarket Vol (M)","PM Volume (M)"],
    "PM$":    ["PM $Vol (M)","PM Dollar Vol (M)","PM $ Volume (M)"],
    "FLOAT":  ["Float M Shares","Public Float (M)","Float (M)","Float"],
    "MCAP":   ["MarketCap M","Market Cap (M)","MCap M"],
    "SI":     ["Short Interest %","Short Float %","Short Interest (Float) %"],
    "CAT":    ["Catalyst","News","PR"],
}
def _norm(s: str) -> str:
    s = re.sub(r"\s+"," ", str(s).strip().lower())
    return s.replace("$","").replace("%","").replace("’","").replace("'","")
def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns); nm = {c:_norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n: return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None
def _parse_catalyst_col(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    pos = {"1","true","yes","y","t"}
    neg = {"0","false","no","n","f",""}
    out = []
    for v in s.fillna(""):
        if v in pos: out.append(1.0)
        elif v in neg: out.append(0.0)
        elif "pr" in v or "news" in v or "catalyst" in v: out.append(1.0)
        else:
            try: out.append(float(v))
            except: out.append(0.0)
    return pd.Series(out, dtype=float).clip(0,1)

# ============ Logistic helpers ============
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0/(1.0 + np.exp(-z))

def _logloss(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    p = np.clip(p, 1e-9, 1-1e-9)
    if w is None:
        return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))
    sw = float(np.sum(w)) if np.sum(w) > 0 else 1.0
    return float(-np.sum(w*(y*np.log(p) + (1-y)*np.log(1-p))) / sw)

def _soft(x: float, thr: float) -> float:
    if x > thr:  return x - thr
    if x < -thr: return x + thr
    return 0.0

def lasso_logistic_cd(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                      lam: float, l2: float = 1e-6, max_iter: int = 150, tol: float = 1e-6
                     ) -> Tuple[np.ndarray, float]:
    """
    Coordinate descent for L1-penalized logistic:
      minimize  -sum w_i [y_i log p_i + (1-y_i) log(1-p_i)] + lam * ||beta||_1 + (l2/2)||beta||^2
    Returns (beta[k], intercept).
    """
    n, k = X.shape
    beta = np.zeros(k, dtype=float)
    b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))  # logit prior
    for _ in range(max_iter):
        z = b + X @ beta
        p = _sigmoid(z)
        W = p*(1-p)*sample_w
        # intercept update (Newton)
        g0 = np.sum(sample_w*(y - p))
        h0 = np.sum(W) + 1e-9
        b_new = b + g0 / h0
        # coordinates
        beta_new = beta.copy()
        for j in range(k):
            xj = X[:, j]
            # gradient & hessian (diagonal) for j
            gj = np.sum(sample_w * xj * (y - p)) - l2 * beta[j]
            hj = np.sum(W * xj * xj) + l2 + 1e-9
            zj = beta[j] + gj / hj
            beta_new[j] = _soft(zj, lam / hj)
        # check convergence
        if np.linalg.norm(beta_new - beta) < tol and abs(b_new - b) < tol:
            beta, b = beta_new, b_new
            break
        beta, b = beta_new, b_new
    return beta, b

def lasso_cv(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray, nfolds: int = 5
            ) -> Tuple[float, np.ndarray, float]:
    """
    Pick λ by K-fold CV on logloss. Grid from λ_max → 0.01*λ_max (log-spaced).
    Returns (lambda_best, beta, intercept) fitted on full data.
    """
    n, k = X.shape
    # λ_max: max |grad| at beta=0 (with intercept only)
    b0 = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    p0 = _sigmoid(b0 + np.zeros(n))
    grad = X.T @ (sample_w * (y - p0))
    lam_max = float(np.max(np.abs(grad)) + 1e-9)
    lam_grid = np.exp(np.linspace(np.log(lam_max), np.log(max(lam_max*0.01, 1e-6)), 30))

    # build folds
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    folds = np.array_split(idx, nfolds)

    best_lam, best_loss = None, float("inf")
    for lam in lam_grid:
        losses = []
        for f in range(nfolds):
            val_idx = folds[f]
            tr_idx  = np.setdiff1d(idx, val_idx, assume_unique=True)
            Xtr, ytr, wtr = X[tr_idx], y[tr_idx], sample_w[tr_idx]
            Xvl, yvl, wvl = X[val_idx], y[val_idx], sample_w[val_idx]
            beta, b = lasso_logistic_cd(Xtr, ytr, wtr, lam=lam, l2=1e-6, max_iter=160, tol=1e-6)
            p_val = _sigmoid(b + Xvl @ beta)
            losses.append(_logloss(yvl, p_val, wvl))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss, best_lam = mean_loss, lam

    beta_full, b_full = lasso_logistic_cd(X, y, sample_w, lam=best_lam, l2=1e-6, max_iter=200, tol=1e-6)
    return best_lam, beta_full, b_full

def logit_fit_weighted_L2(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                          l2: float = 1.0, max_iter: int = 160, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Stable final fit on selected features."""
    n, k = X.shape
    w = np.zeros(k+1)  # [intercept, beta...]
    R = np.eye(k+1); R[0,0] = 0.0; R *= l2
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    for _ in range(max_iter):
        z = Xb @ w
        p = _sigmoid(z)
        W = p*(1-p)*sample_w
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H  = Xb.T @ WX + R
        g  = Xb.T @ ((y - p) * sample_w)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w + delta
        if np.linalg.norm(delta) < tol:
            w = w_new
            break
        w = w_new
    return w[1:].astype(float), float(w[0])

# ============ Grades ============
def _inv_warp_p(sw: float, alpha: float = 2.6) -> float:
    sw = float(np.clip(sw, 0.0, 1.0))
    return float(1.0 - (1.0 - sw) ** (1.0 / max(1e-9, alpha)))

def make_grade_cuts() -> dict:
    A_cut, Ap_cut, App_cut = 0.90, 0.97, 0.99
    p_cal = st.session_state.get("TRAIN_PREDS", np.array([]))
    if p_cal.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.80, "C": 0.60}
    sub = np.asarray(p_cal[(p_cal < A_cut) & np.isfinite(p_cal)])
    if sub.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.80, "C": 0.60}
    sub.sort()
    alpha = 2.6
    b_pos = _inv_warp_p(0.80, alpha)  # skewed higher
    c_pos = _inv_warp_p(0.50, alpha)  # median-ish
    B_cut = float(np.quantile(sub, b_pos))
    C_cut = float(np.quantile(sub, c_pos))
    B_cut = min(B_cut, 0.89)
    if not np.isfinite(B_cut): B_cut = 0.80
    if not np.isfinite(C_cut): C_cut = 0.60
    if C_cut >= B_cut: C_cut = max(0.10, B_cut - 0.02)
    return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": B_cut, "C": C_cut}

def _prob_to_grade(p: float, cuts: Dict[str,float]) -> str:
    if p >= cuts.get("App",0.99): return "A++"
    if p >= cuts.get("Ap",0.97):  return "A+"
    if p >= cuts.get("A",0.90):   return "A"
    if p >= cuts.get("B",0.80):   return "B"
    if p >= cuts.get("C",0.60):   return "C"
    return "D"

# ============ Upload & Learn ============
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn (LASSO select + final fit)", use_container_width=True)

def _load_and_learn(xls: pd.ExcelFile, sheet: str) -> None:
    raw = pd.read_excel(xls, sheet)

    # Map columns (Legend)
    col = {}
    for key, cands in _DEF.items():
        pick = _pick(raw, cands)
        if pick: col[key] = pick
    if "FT" not in col:
        st.error("No 'FT' column found in sheet.")
        return

    df = pd.DataFrame()
    ft_series = pd.to_numeric(raw[col["FT"]], errors="coerce")
    y = (ft_series.fillna(0.0) >= 0.5).astype(float).to_numpy(dtype=float)

    def _add(colname: str, key: str):
        if key in col:
            df[colname] = pd.to_numeric(raw[col[key]], errors="coerce")

    _add("gap_pct", "GAP")
    _add("atr_usd", "ATR")
    _add("rvol", "RVOL")
    _add("pm_vol_m", "PMVOL")
    _add("pm_dol_m", "PM$")
    _add("float_m", "FLOAT")
    _add("mcap_m", "MCAP")
    _add("si_pct", "SI")
    if "CAT" in col:
        df["catalyst"] = _parse_catalyst_col(raw[col["CAT"]])
    else:
        df["catalyst"] = 0.0

    # Build candidate features (your list)
    def _ln_gapf(r):   return _safe_log(_nz(r.get("gap_pct"),0.0)/100.0)
    def _ln_atr(r):    return _safe_log(r.get("atr_usd"))
    def _ln_mcap(r):   return _safe_log(r.get("mcap_m"))
    def _ln1p_rvol(r): return _safe_log(1.0 + _nz(r.get("rvol"),0.0))
    def _ln1p_pmvol(r):return _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)
    def _ln1p_pmmc(r):
        pm_dol = _nz(r.get("pm_dol_m"),0.0); mc = _nz(r.get("mcap_m"),0.0)
        return _safe_log(pm_dol / max(1e-6, mc) + 1.0)
    def _ln1p_si(r):   return _safe_log(1.0 + _nz(r.get("si_pct"),0.0))
    def _catalyst(r):  return float(_nz(r.get("catalyst"),0.0))
    def _ln_float(r):  return _safe_log(r.get("float_m"))

    feat_defs = [
        ("ln_gapf",    _ln_gapf,    ["gap_pct"]),
        ("ln_atr",     _ln_atr,     ["atr_usd"]),
        ("ln_mcap",    _ln_mcap,    ["mcap_m"]),
        ("ln1p_rvol",  _ln1p_rvol,  ["rvol"]),
        ("ln1p_pmvol", _ln1p_pmvol, ["pm_vol_m"]),
        ("ln1p_pmmc",  _ln1p_pmmc,  ["pm_dol_m","mcap_m"]),
        ("ln1p_si",    _ln1p_si,    ["si_pct"]),
        ("catalyst",   _catalyst,   ["catalyst"]),
        ("ln_float",   _ln_float,   ["float_m"]),
    ]
    use_defs = [(n,f,req) for (n,f,req) in feat_defs if all(k in df.columns for k in req)]
    if not use_defs:
        st.error("No usable features found.")
        return

    X_list = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        X_list.append([f(r) for (n,f,_) in use_defs])
    X = np.array(X_list, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    Z = np.clip(Z, -6.0, 6.0)

    # Class weights
    pos = float(y.sum()); neg = float(len(y) - pos)
    w_pos = 0.5 / max(1e-9, pos) if pos > 0 else 0.0
    w_neg = 0.5 / max(1e-9, neg) if neg > 0 else 0.0
    sample_w = np.where(y >= 0.5, w_pos, w_neg).astype(float)

    # ===== LASSO selection by 5-fold CV =====
    if Z.shape[0] < 12 or np.unique(y).size < 2:
        st.error("Need ≥12 rows and both classes for training.")
        return

    lam_best, beta_l1, b_l1 = lasso_cv(Z, y, sample_w, nfolds=5)
    sel_idx = np.where(np.abs(beta_l1) > 1e-8)[0]
    sel_names = [use_defs[i][0] for i in sel_idx]

    if len(sel_idx) == 0:
        st.warning("LASSO selected no features; falling back to weak ridge on all.")
        sel_idx = np.arange(Z.shape[1])
        sel_names = [n for (n,_,_) in use_defs]

    Zs = Z[:, sel_idx]

    # Final stable weighted L2 logistic on selected features
    coef, bias = logit_fit_weighted_L2(Zs, y, sample_w, l2=1.0, max_iter=200, tol=1e-6)
    p_cal = _sigmoid(bias + Zs @ coef)
    st.session_state.TRAIN_PREDS = p_cal.astype(float)

    st.session_state.FT = {
        "feat_names_all": [n for (n,_,_) in use_defs],
        "mu_all": mu, "sd_all": sd,
        "sel_idx": sel_idx.astype(int),
        "sel_names": sel_names,
        "coef": coef.astype(float),
        "bias": float(bias),
    }

    st.success(f"LASSO CV chose λ≈{lam_best:.4g}; selected features: {', '.join(sel_names) if sel_names else '(none)'}")
    st.session_state.flash = "Learning complete."
    _rerun()

if learn_btn:
    if not uploaded:
        st.error("Upload a workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if sheet_name not in xls.sheet_names:
                st.error(f"Sheet '{sheet_name}' not found. Available: {xls.sheet_names}")
            else:
                _load_and_learn(xls, sheet_name)
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============ Inference ============
def _predict_ft_prob_live(inputs: Dict[str, float]) -> float:
    ART = st.session_state.get("FT", {})
    sel_idx = ART.get("sel_idx", None)
    coef = ART.get("coef", None); bias = ART.get("bias", 0.0)
    mu_all = ART.get("mu_all", None); sd_all = ART.get("sd_all", None)
    feat_names_all = ART.get("feat_names_all", [])
    if sel_idx is None or coef is None or mu_all is None or sd_all is None or not feat_names_all:
        return 0.50

    def _ln_gapf():   return _safe_log(_nz(inputs.get("gap_pct"),0.0)/100.0)
    def _ln_atr():    return _safe_log(inputs.get("atr_usd"))
    def _ln_mcap():   return _safe_log(inputs.get("mcap_m"))
    def _ln1p_rvol(): return _safe_log(1.0 + _nz(inputs.get("rvol"),0.0))
    def _ln1p_pmvol():return _safe_log(_nz(inputs.get("pm_vol_m"),0.0) + 1.0)
    def _ln1p_pmmc():
        pm_dol = _nz(inputs.get("pm_dol_m"),0.0); mc = _nz(inputs.get("mcap_m"),0.0)
        return _safe_log(pm_dol / max(1e-6, mc) + 1.0)
    def _ln1p_si():   return _safe_log(1.0 + _nz(inputs.get("si_pct"),0.0))
    def _catalyst():  return float(_nz(inputs.get("catalyst"),0.0))
    def _ln_float():  return _safe_log(inputs.get("float_m"))

    maker = {
        "ln_gapf": _ln_gapf, "ln_atr": _ln_atr, "ln_mcap": _ln_mcap,
        "ln1p_rvol": _ln1p_rvol, "ln1p_pmvol": _ln1p_pmvol, "ln1p_pmmc": _ln1p_pmmc,
        "ln1p_si": _ln1p_si, "catalyst": _catalyst, "ln_float": _ln_float,
    }
    x_all = np.array([maker[n]() for n in feat_names_all], dtype=float)
    z_all = (x_all - mu_all) / sd_all
    z_all = np.clip(z_all, -6.0, 6.0)
    zs = z_all[sel_idx]
    p = float(_sigmoid(bias + np.dot(zs, coef)))
    return float(np.clip(p, 1e-4, 1-1e-4))

GRADE_CUTS = make_grade_cuts()

# ============ Tabs ============
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

# ============ Add Stock ============
with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2,1.2,1.0])
        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mcap_m   = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %",                    0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"], index=0)  # under $Vol
        with c3:
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        inputs = {
            "mcap_m": mcap_m, "float_m": float_m, "si_pct": si_pct,
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol,
            "pm_vol_m": pm_vol_m, "pm_dol_m": pm_dol_m,
            "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }
        p = _predict_ft_prob_live(inputs)
        grade = _prob_to_grade(p, GRADE_CUTS)

        row = {
            "Ticker": ticker,
            "Level": grade,
            "FT Probability %": round(100.0 * p, 2),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} — Grade {row['Level']} (P={row['FT Probability %']:.2f}%)"
        _rerun()

    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Grade", l.get("Level","—"))
        c.metric("FT Probability", f"{l.get('FT Probability %',0):.2f}%")

# ============ Ranking ============
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        if "FT Probability %" in df.columns:
            df = df.sort_values("FT Probability %", ascending=False).reset_index(drop=True)

        cols_to_show = ["Ticker","Level","FT Probability %"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Level") else 0.0

        st.dataframe(
            df[cols_to_show],
            use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Level": st.column_config.TextColumn("Grade"),
                "FT Probability %": st.column_config.NumberColumn("FT Probability %", format="%.2f"),
            }
        )

        st.markdown("#### Delete rows")
        del_cols = st.columns(4)
        head12 = df.head(12).reset_index(drop=True)
        for i, r in head12.iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    _rerun()

        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        st.markdown("### 📋 Ranking (Markdown view)")
        md_text = df_to_markdown_table(df, cols_to_show)
        st.code(md_text, language="markdown")
        st.download_button(
            "Download Markdown",
            md_text.encode("utf-8"),
            "ranking.md", "text/markdown", use_container_width=True
        )

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                _rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
