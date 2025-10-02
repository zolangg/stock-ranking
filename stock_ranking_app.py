# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Helpers ==============================
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}
    # exact (case-insensitive)
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc:
                return c
    # normalized exact
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c
    # normalized contains
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]:
                return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        ss = str(s).strip().replace(" ", "")
        if "," in ss and "." not in ss:
            ss = ss.replace(",", ".")
        else:
            ss = ss.replace(",", "")
        return float(ss)
    except Exception:
        return np.nan

def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))

# ---------- Winsorization ----------
def _compute_bounds(arr: np.ndarray, lo_q=0.01, hi_q=0.99):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

def _apply_bounds(arr: np.ndarray, lo: float, hi: float):
    out = arr.copy()
    if np.isfinite(lo): out = np.maximum(out, lo)
    if np.isfinite(hi): out = np.minimum(out, hi)
    return out

# ---------- Isotonic Regression (PAV) ----------
def _pav_isotonic(x: np.ndarray, y: np.ndarray):
    """
    Pool-Adjacent-Violators algorithm.
    x must be finite; y must be finite; no weights (equal weights).
    Returns monotone (non-decreasing) piecewise-constant fit evaluated at x (in x-order),
    and breakpoints (unique x_sorted) with fitted y for later interpolation at inference.
    """
    # sort by x
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    # initialize levels
    level_y = ys.astype(float).copy()
    level_n = np.ones_like(level_y)

    # merge adjacent blocks when monotonicity violated
    i = 0
    while i < len(level_y) - 1:
        if level_y[i] > level_y[i+1]:
            # pool blocks i and i+1
            new_y = (level_y[i]*level_n[i] + level_y[i+1]*level_n[i+1]) / (level_n[i] + level_n[i+1])
            new_n = level_n[i] + level_n[i+1]
            level_y[i] = new_y
            level_n[i] = new_n
            # remove i+1 by shifting left (lazy compaction)
            level_y = np.delete(level_y, i+1)
            level_n = np.delete(level_n, i+1)
            xs = np.delete(xs, i+1)
            # step back to check previous block if any
            if i > 0:
                i -= 1
        else:
            i += 1

    # Now xs are the breakpoints, level_y are fitted step values
    return xs, level_y

def _iso_predict(break_x: np.ndarray, break_y: np.ndarray, x_new: np.ndarray):
    """
    Piecewise-constant, left-continuous isotonic prediction with linear interpolation between steps
    (small smoothing): we will do piecewise-linear interpolation across breakpoints to avoid jumps.
    """
    if break_x.size == 0:
        return np.full_like(x_new, np.nan, dtype=float)
    # ensure increasing order
    idx = np.argsort(break_x)
    bx = break_x[idx]
    by = break_y[idx]
    # If only one breakpoint, constant
    if bx.size == 1:
        return np.full_like(x_new, by[0], dtype=float)

    # Linear interpolation across breakpoints; clip outside to ends
    return np.interp(x_new, bx, by, left=by[0], right=by[-1])

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "models" not in st.session_state: st.session_state.models = {}
if "lassoA" not in st.session_state: st.session_state.lassoA = {}   # ratio model + winsor + calibrator

# ============================== Core & Moderate sets ==============================
# CORE (legacy RVOL removed). New PM features added.
VAR_CORE = [
    "Gap_%",
    "FR_x",
    "PM$Vol/MC_%",
    "Catalyst",
    "PM_Vol_%",
    "Max_Pull_PM_%",     # Premarket Max Pullback (%)
    "RVOL_Max_PM_cum"    # Premarket Max RVOL (cum)
]
# MODERATE — canonical PM-max bases + legacy fallbacks (for old files)
VAR_MODERATE = [
    "MC_PM_Max_M",       # Premarket Market Cap (M$)
    "Float_PM_Max_M",    # Premarket Float (M)
    "PM_Vol_M",
    "PM_$Vol_M$",
    "ATR_$",
    "Daily_Vol_M",
    "MarketCap_M$",      # fallback
    "Float_M",           # fallback
]
VAR_ALL = VAR_CORE + VAR_MODERATE

# ============================== Simple LASSO (coordinate descent) ==============================
def _kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)

def _lasso_cd_std(Xs, y, lam, max_iter=1200, tol=1e-6):
    # Xs standardized; coordinate descent
    n, p = Xs.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        w_old = w.copy()
        y_hat = Xs @ w
        for j in range(p):
            r_j = y - y_hat + Xs[:, j] * w[j]
            rho = (Xs[:, j] @ r_j) / n
            if   rho < -lam/2: w[j] = rho + lam/2
            elif rho >  lam/2: w[j] = rho - lam/2
            else:              w[j] = 0.0
            y_hat = Xs @ w
        if np.linalg.norm(w - w_old) < tol:
            break
    return w

# ============================== Train ratio model with winsorization + isotonic calibration ==============================
def train_ratio_winsor_iso(df: pd.DataFrame, lo_q=0.01, hi_q=0.99) -> dict:
    """
    Target: ln(multiplier) where multiplier = max(Daily_Vol_M / PM_Vol_M, 1).
    Features (log unless noted): ln_mcap_pmmax, ln_gapf, ln_atr, ln_pm, ln_pm_dol, ln_fr,
      catalyst (0/1), ln_float_pmmax, maxpullpm (linear %), ln_rvolmaxpm, PM$Vol/MC_% (linear %)
    Steps:
      1) Build features + target
      2) Split by row order: first 80% train, last 20% validation
      3) Winsorize selected inputs + multiplier on TRAIN only; apply same bounds to VAL
      4) LASSO on standardized features (TRAIN) → select → OLS refit (TRAIN) on original (winsorized) features
      5) Predict on VAL → isotonic calibration (PAV) mapping raw multiplier → true multiplier
      6) Return model dict with terms, betas, eps, winsor bounds per feature, and isotonic breakpoints
    """
    eps = 1e-6

    # Prefer PM-Max columns; keep legacy fallbacks for older DBs
    mcap_series  = df["MC_PM_Max_M"]    if "MC_PM_Max_M"    in df.columns else df.get("MarketCap_M$")
    float_series = df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df.get("Float_M")
    need_min = {"ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"}
    if mcap_series is None or float_series is None or not need_min.issubset(df.columns):
        return {}

    PM  = pd.to_numeric(df["PM_Vol_M"],    errors="coerce").values
    DV  = pd.to_numeric(df["Daily_Vol_M"], errors="coerce").values
    valid_pm = np.isfinite(PM) & np.isfinite(DV) & (PM > 0) & (DV > 0)
    if valid_pm.sum() < 50:
        return {}

    # Base features
    ln_mcap   = np.log(np.clip(pd.to_numeric(mcap_series, errors="coerce").values,  eps, None))
    ln_gapf   = np.log(np.clip(pd.to_numeric(df["Gap_%"], errors="coerce").values,  0,   None) / 100.0 + eps)
    ln_atr    = np.log(np.clip(pd.to_numeric(df["ATR_$"], errors="coerce").values,  eps, None))
    ln_pm     = np.log(np.clip(pd.to_numeric(df["PM_Vol_M"], errors="coerce").values, eps, None))
    ln_pm_dol = np.log(np.clip(pd.to_numeric(df["PM_$Vol_M$"], errors="coerce").values, eps, None))
    ln_fr     = np.log(np.clip(pd.to_numeric(df["FR_x"], errors="coerce").values,   eps, None))
    ln_float_pmmax = np.log(np.clip(pd.to_numeric(float_series, errors="coerce").values, eps, None))
    maxpullpm      = pd.to_numeric(df.get("Max_Pull_PM_%", np.nan), errors="coerce").values
    ln_rvolmaxpm   = np.log(np.clip(pd.to_numeric(df.get("RVOL_Max_PM_cum", np.nan), errors="coerce").values, eps, None))
    pm_dol_over_mc = pd.to_numeric(df.get("PM$Vol/MC_%", np.nan), errors="coerce").values

    catalyst_raw   = df.get("Catalyst", np.nan)
    catalyst       = pd.to_numeric(catalyst_raw, errors="coerce").fillna(0.0).clip(0,1).values

    # Target: ln(multiplier) with multiplier ≥ 1
    multiplier = np.maximum(DV / PM, 1.0)
    y_ln = np.log(multiplier)

    # Stack all feature vectors (order matters for terms names)
    feats = [
        ("ln_mcap_pmmax",  ln_mcap),
        ("ln_gapf",        ln_gapf),
        ("ln_atr",         ln_atr),
        ("ln_pm",          ln_pm),
        ("ln_pm_dol",      ln_pm_dol),
        ("ln_fr",          ln_fr),
        ("catalyst",       catalyst),
        ("ln_float_pmmax", ln_float_pmmax),
        ("maxpullpm",      maxpullpm),        # linear %
        ("ln_rvolmaxpm",   ln_rvolmaxpm),
        ("pm_dol_over_mc", pm_dol_over_mc),  # linear %
    ]

    # Assemble X and validity mask
    cols = []
    for _, arr in feats:
        cols.append(arr.reshape(-1,1))
    X_all = np.hstack(cols)

    mask = valid_pm & np.isfinite(y_ln) & np.isfinite(X_all).all(axis=1)
    X_all = X_all[mask]; y_ln = y_ln[mask]; PMm = PM[mask]; mult_true = multiplier[mask]

    if X_all.shape[0] < 50:
        return {}

    n = X_all.shape[0]
    split = max(10, int(n * 0.8))  # first 80% train, last 20% val (row order as proxy for time)
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr, y_va = y_ln[:split],   y_ln[split:]
    mult_tr, mult_va = mult_true[:split], mult_true[split:]

    # Winsorize selected features (train only) and apply to val with same bounds
    # Features to winsorize (linear scale): maxpullpm, pm_dol_over_mc
    # Also winsorize multiplier (target in linear space)
    winsor_bounds = {}  # store for inference clipping
    # helper to process column by feature index
    def _winsor_feature(col_idx, is_log=False):
        arr_tr = X_tr[:, col_idx]
        if is_log:
            # For log-vars we usually skip winsor; they are already stabilized.
            return
        lo, hi = _compute_bounds(arr_tr[np.isfinite(arr_tr)])
        winsor_bounds[feats[col_idx][0]] = (lo, hi)
        X_tr[:, col_idx] = _apply_bounds(arr_tr, lo, hi)
        X_va[:, col_idx] = _apply_bounds(X_va[:, col_idx], lo, hi)

    # Apply winsorization
    name_to_idx = {name:i for i,(name,_) in enumerate(feats)}
    # linear (%) features
    for nm in ["maxpullpm", "pm_dol_over_mc"]:
        if nm in name_to_idx:
            _winsor_feature(name_to_idx[nm], is_log=False)

    # Target multiplier winsorization (linear space), then back to y_ln
    m_lo, m_hi = _compute_bounds(mult_tr)
    mult_tr_w = _apply_bounds(mult_tr, m_lo, m_hi)
    y_tr = np.log(mult_tr_w)
    # apply same bounds to validation (for calibration target)
    mult_va_w = _apply_bounds(mult_va, m_lo, m_hi)

    # Standardize for LASSO using TRAIN stats only
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs_tr = (X_tr - mu) / sd
    Xs_va = (X_va - mu) / sd

    # LASSO → select terms
    folds = _kfold_indices(len(y_tr), k=min(5, max(2, len(y_tr)//10)), seed=42)
    lam_grid = np.geomspace(0.001, 1.0, 30)
    cv_mse = []
    for lam in lam_grid:
        errs = []
        for vi in range(len(folds)):
            te_idx = folds[vi]; tr_idx = np.hstack([folds[j] for j in range(len(folds)) if j != vi])
            Xtr, ytr = Xs_tr[tr_idx], y_tr[tr_idx]
            Xte, yte = Xs_tr[te_idx], y_tr[te_idx]
            w = _lasso_cd_std(Xtr, ytr, lam=lam, max_iter=1600)
            yhat = Xte @ w
            errs.append(np.mean((yhat - yte)**2))
        cv_mse.append(np.mean(errs))
    lam_best = float(lam_grid[int(np.argmin(cv_mse))])
    w_l1 = _lasso_cd_std(Xs_tr, y_tr, lam=lam_best, max_iter=2500)
    sel = np.flatnonzero(np.abs(w_l1) > 1e-8)

    if sel.size == 0:
        return {}

    # Refit OLS on TRAIN (original winsorized features) with selected terms
    Xtr_sel = X_tr[:, sel]
    X_design = np.column_stack([np.ones(Xtr_sel.shape[0]), Xtr_sel])
    coef_ols, *_ = np.linalg.lstsq(X_design, y_tr, rcond=None)
    b0 = float(coef_ols[0]); bet = coef_ols[1:].astype(float)

    # Predict on VALIDATION (raw multiplier prediction)
    Xva_sel = X_va[:, sel]
    yhat_va_ln = (np.column_stack([np.ones(Xva_sel.shape[0]), Xva_sel]) @ coef_ols).astype(float)
    mult_pred_va = np.exp(yhat_va_ln)

    # Isotonic calibration on (mult_pred_va → mult_va_w)
    # Keep only finite pairs
    finite_mask = np.isfinite(mult_pred_va) & np.isfinite(mult_va_w)
    if finite_mask.sum() >= 5:
        bx, by = _pav_isotonic(mult_pred_va[finite_mask], mult_va_w[finite_mask])
    else:
        # Fallback: identity mapping using a single breakpoint
        bx, by = np.array([1.0]), np.array([1.0])

    selected_terms = [feats[i][0] for i in sel]
    # Store per-feature winsor bounds for inference clipping
    # For features not winsorized, store (nan, nan)
    for i, (nm, _) in enumerate(feats):
        if nm not in winsor_bounds:
            winsor_bounds[nm] = (np.nan, np.nan)

    # Pack model
    model = {
        "eps": eps,
        "terms": selected_terms,
        "b0": b0,
        "betas": bet,
        "sel_idx": sel.tolist(),   # for debugging
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "winsor_bounds": {k: (float(v[0]) if np.isfinite(v[0]) else np.nan,
                              float(v[1]) if np.isfinite(v[1]) else np.nan)
                          for k, v in winsor_bounds.items()},
        "iso_bx": bx.tolist(),
        "iso_by": by.tolist(),
        "feat_order": [nm for nm,_ in feats],  # to reconstruct feature vector in same order
        "mult_bounds": (float(m_lo) if np.isfinite(m_lo) else np.nan,
                        float(m_hi) if np.isfinite(m_hi) else np.nan)  # not used at inference, informational
    }
    return model

# ============================== Predict Daily volume via calibrated ratio model ==============================
def predict_daily_calibrated(row: dict, model: dict) -> float:
    """
    1) Build feature vector in training order
    2) Apply TRAIN winsor bounds to linear features (same as training)
    3) Select trained terms, refit form b0+betas
    4) raw multiplier = exp(yhat_ln)
    5) calibrated multiplier = iso(raw multiplier)
    6) PredDaily = PM * max(calibrated multiplier, 1)
    """
    if not model or "betas" not in model:
        return np.nan

    eps = float(model.get("eps", 1e-6))
    feat_order = model["feat_order"]
    winsor_bounds = model.get("winsor_bounds", {})
    iso_bx = np.array(model.get("iso_bx", [1.0]), dtype=float)
    iso_by = np.array(model.get("iso_by", [1.0]), dtype=float)
    terms = model["terms"]
    sel = model.get("sel_idx", [])
    mu = np.array(model.get("mu", []), dtype=float)
    sd = np.array(model.get("sd", []), dtype=float)
    b0 = float(model["b0"]); bet = np.array(model["betas"], dtype=float)

    def safe_log(v):
        v = float(v) if v is not None else np.nan
        return np.log(np.clip(v, eps, None)) if np.isfinite(v) else np.nan

    # Build features in the same order as training
    ln_mcap_pmmax  = safe_log(row.get("MC_PM_Max_M") or row.get("MarketCap_M$"))
    ln_gapf        = np.log(np.clip((row.get("Gap_%") or 0.0)/100.0 + eps, eps, None)) if row.get("Gap_%") is not None else np.nan
    ln_atr         = safe_log(row.get("ATR_$"))
    ln_pm          = safe_log(row.get("PM_Vol_M"))
    ln_pm_dol      = safe_log(row.get("PM_$Vol_M$"))
    ln_fr          = safe_log(row.get("FR_x"))
    catalyst       = 1.0 if (str(row.get("CatalystYN","No")).lower()=="yes" or float(row.get("Catalyst",0))>=0.5) else 0.0
    ln_float_pmmax = safe_log(row.get("Float_PM_Max_M") or row.get("Float_M"))
    maxpullpm      = float(row.get("Max_Pull_PM_%")) if row.get("Max_Pull_PM_%") is not None else np.nan
    ln_rvolmaxpm   = safe_log(row.get("RVOL_Max_PM_cum"))
    pm_dol_over_mc = float(row.get("PM$Vol/MC_%")) if row.get("PM$Vol/MC_%") is not None else np.nan

    feat_map = {
        "ln_mcap_pmmax":  ln_mcap_pmmax,
        "ln_gapf":        ln_gapf,
        "ln_atr":         ln_atr,
        "ln_pm":          ln_pm,
        "ln_pm_dol":      ln_pm_dol,
        "ln_fr":          ln_fr,
        "catalyst":       catalyst,
        "ln_float_pmmax": ln_float_pmmax,
        "maxpullpm":      maxpullpm,
        "ln_rvolmaxpm":   ln_rvolmaxpm,
        "pm_dol_over_mc": pm_dol_over_mc,
    }

    X_vec = []
    for nm in feat_order:
        v = feat_map.get(nm, np.nan)
        if not np.isfinite(v): return np.nan
        # Apply winsor clipping at inference for linear features (same bounds as training)
        lo, hi = winsor_bounds.get(nm, (np.nan, np.nan))
        if np.isfinite(lo) or np.isfinite(hi):
            v = float(np.clip(v, lo if np.isfinite(lo) else v, hi if np.isfinite(hi) else v))
        X_vec.append(v)
    X_vec = np.array(X_vec, dtype=float)

    # Select trained columns
    if not sel:
        return np.nan
    X_sel = X_vec[sel]

    # Standardize using TRAIN stats
    mu = mu if mu.size else np.zeros_like(X_vec)
    sd = sd if sd.size else np.ones_like(X_vec)
    Xs_all = (X_vec - mu) / (sd + 1e-12)
    Xs_sel = Xs_all[sel]

    # OLS prediction in original (not standardized) space:
    # (We stored betas from OLS on winsorized ORIGINAL features, not std.)
    # So we should use ORIGINAL features used in OLS:
    # Reconstruct those from X_vec (not standardized)
    X_orig_sel = X_vec[sel]
    yhat_ln = b0 + float(np.dot(bet, X_orig_sel))

    raw_mult = np.exp(yhat_ln) if np.isfinite(yhat_ln) else np.nan
    if not np.isfinite(raw_mult): return np.nan

    # Isotonic calibration
    cal_mult = float(_iso_predict(iso_bx, iso_by, np.array([raw_mult]))[0])
    # Ensure structural floor (≥1)
    cal_mult = max(cal_mult, 1.0)

    PM = float(row.get("PM_Vol_M") or np.nan)
    if not np.isfinite(PM) or PM <= 0:
        return np.nan
    return float(PM * cal_mult)

# ============================== Upload DB → Build Medians & Train Model ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
            sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
            raw = pd.read_excel(xls, sheet)

            # --- auto-detect group column ---
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
                for c in raw.columns:
                    vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group = c
                        break

            if col_group is None:
                st.error("Could not detect FT column (0/1). Please ensure your sheet has an FT or binary column.")
            else:
                df = pd.DataFrame()
                df["GroupRaw"] = raw[col_group]

                def add_num(dfout, name, src_candidates):
                    src = _pick(raw, src_candidates)
                    if src:
                        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                # ===== Map columns (new names + legacy fallbacks) =====
                add_num(df, "MC_PM_Max_M",      ["mc pm max (m)","mc pm max m","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m"])
                add_num(df, "Float_PM_Max_M",   ["float pm max (m)","float pm max m","float_pm_max_m","float pm max (m shares)"])
                add_num(df, "MarketCap_M$",     ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])  # fallback
                add_num(df, "Float_M",          ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])                                   # fallback

                add_num(df, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
                add_num(df, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
                add_num(df, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
                add_num(df, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
                add_num(df, "PM_Vol_%",         ["pm vol (%)","pm_vol_%","pm vol percent","pm volume (%)","pm_vol_percent"])
                add_num(df, "Daily_Vol_M",      ["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])
                add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
                add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

                # Catalyst (binary/YN/0-1)
                cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
                def _to_binary_local(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1.0
                    if sv in {"0","false","no","n","f"}: return 0.0
                    try:
                        fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                    except: return np.nan
                if cand_catalyst:
                    df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

                # ===== Derived (prefer PM-Max fields) =====
                float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
                if {"PM_Vol_M", float_basis}.issubset(df.columns):
                    df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)

                mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
                if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                    df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

                # Percent-like fields as real %
                if "Gap_%" in df.columns:            df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
                if "PM_Vol_%" in df.columns:         df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
                if "Max_Pull_PM_%" in df.columns:    df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

                # FT groups
                def _to_binary(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1
                    if sv in {"0","false","no","n","f"}: return 0
                    try:
                        fv = float(sv); return 1 if fv >= 0.5 else 0
                    except: return np.nan

                df["FT01"] = df["GroupRaw"].map(_to_binary)
                df = df[df["FT01"].isin([0,1])]
                if df.empty or df["FT01"].nunique() < 2:
                    st.error("Could not find both FT=1 and FT=0 rows in the DB. Please check the group column.")
                else:
                    df["Group"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

                    var_core = [v for v in VAR_CORE if v in df.columns]
                    var_mod  = [v for v in VAR_MODERATE if v in df.columns]
                    var_all  = var_core + var_mod

                    # Medians/MADs unchanged (still on original variables)
                    gmed  = df.groupby("Group")[var_all].median(numeric_only=True).T
                    gmads = df.groupby("Group")[var_all].apply(lambda g: g.apply(_mad)).T

                    st.session_state.models = {
                        "models_tbl": gmed,
                        "mad_tbl": gmads,
                        "var_core": var_core,
                        "var_moderate": var_mod
                    }

                    # Train calibrated ratio model
                    lasso_model = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99)
                    st.session_state.lassoA = lasso_model or {}

                    if lasso_model:
                        st.success(f"Built medians and trained calibrated ratio model. Selected terms: {st.session_state.lassoA.get('terms')}")
                    else:
                        st.warning("Training skipped (insufficient or incompatible data). Medians built; predictions disabled.")
                    do_rerun()

        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============================== Medians tables (grouped) ==============================
models_data = st.session_state.models
if models_data and isinstance(models_data, dict) and not models_data.get("models_tbl", pd.DataFrame()).empty:
    with st.expander("Model Medians (FT=1 vs FT=0) — grouped", expanded=False):
        med_tbl: pd.DataFrame = models_data["models_tbl"]
        mad_tbl: pd.DataFrame = models_data.get("mad_tbl", pd.DataFrame())
        var_core = models_data.get("var_core", [])
        var_mod  = models_data.get("var_moderate", [])

        sig_thresh = st.slider("Significance threshold (σ)", 0.0, 5.0, 3.0, 0.1,
                               help="Highlight rows where |FT=1 − FT=0| / (MAD₁ + MAD₀) ≥ σ")
        st.session_state["sig_thresh"] = float(sig_thresh)

        def show_grouped_table(title, vars_list):
            if not vars_list:
                st.info(f"No variables available for {title}.")
                return
            sub_med = med_tbl.loc[[v for v in vars_list if v in med_tbl.index]].copy()
            if not mad_tbl.empty and {"FT=1","FT=0"}.issubset(mad_tbl.columns):
                eps = 1e-9
                diff = (sub_med["FT=1"] - sub_med["FT=0"]).abs()
                spread = (mad_tbl.loc[diff.index, "FT=1"].fillna(0.0) + mad_tbl.loc[diff.index, "FT=0"].fillna(0.0))
                sig = diff / (spread.replace(0.0, np.nan) + eps)
                sig_flag = sig >= st.session_state["sig_thresh"]
                def _style_sig(col: pd.Series):
                    return ["background-color: #fde68a; font-weight: 600;" if sig_flag.get(idx, False) else "" for idx in col.index]
                st.markdown(f"**{title}**")
                styled = (sub_med.style.apply(_style_sig, subset=["FT=1"]).apply(_style_sig, subset=["FT=0"]).format("{:.2f}"))
                st.dataframe(styled, use_container_width=True)
            else:
                st.markdown(f"**{title}**")
                cfg = {"FT=1": st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
                       "FT=0": st.column_config.NumberColumn("FT=0 (median)", format="%.2f")}
                st.dataframe(sub_med, use_container_width=True, column_config=cfg, hide_index=False)

        show_grouped_table("Core variables", var_core)
        show_grouped_table("Moderate variables", var_mod)

# ============================== ➕ Manual Input ==============================
st.markdown("---")
st.subheader("Add Stock")

with st.form("add_form", clear_on_submit=True):
    # Two columns of inputs, third just Catalyst
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])

    with c1:
        ticker      = st.text_input("Ticker", "").strip().upper()
        mc_pmmax    = st.number_input("Premarket Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_pm    = st.number_input("Premarket Float (M)", 0.0, step=0.01, format="%.2f")
        gap_pct     = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")
        max_pull_pm = st.number_input("Premarket Max Pullback (%)", 0.0, step=0.1, format="%.1f")

    with c2:
        atr_usd     = st.number_input("Prior Day ATR ($)", 0.0, step=0.01, format="%.2f")
        pm_vol      = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol      = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
        rvol_pm_cum = st.number_input("Premarket Max RVOL", 0.0, step=0.01, format="%.2f")

    with c3:
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)

    submitted = st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
    # Derived using PM-Max bases
    fr   = (pm_vol / float_pm) if float_pm > 0 else 0.0
    pmmc = (pm_dol / mc_pmmax * 100.0) if mc_pmmax > 0 else 0.0

    row = {
        "Ticker": ticker,

        # Canonical PM-max fields
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,

        # Core/moderate basics
        "Gap_%": gap_pct,
        "ATR_$": atr_usd,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,

        # Derived
        "FR_x": fr,                 # PM_Vol_M / Float_PM_Max_M
        "PM$Vol/MC_%": pmmc,        # PM_$Vol_M$ / MC_PM_Max_M * 100

        # New CORE
        "Max_Pull_PM_%": max_pull_pm,
        "RVOL_Max_PM_cum": rvol_pm_cum,

        # Catalyst (binary + label)
        "CatalystYN": catalyst_yn,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
    }

    # Predict Daily Volume (calibrated ratio model)
    pred = predict_daily_calibrated(row, st.session_state.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan

    # PM_Vol_% uses PredVol_M (structurally ≤ 100%)
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    st.session_state.rows.append(row)
    st.session_state.last = row
    st.success(f"Saved {ticker}.")
    do_rerun()

# ============================== Toolbar: Delete + Multiselect (same row) ==============================
tickers = [r.get("Ticker","").strip().upper() for r in st.session_state.rows if r.get("Ticker","")]
tcol_btn, tcol_sel = st.columns([1, 3])
with tcol_btn:
    if st.button("Delete selected", use_container_width=True, type="primary"):
        sel = st.session_state.get("to_delete", [])
        if not sel:
            st.info("No rows selected.")
        else:
            before = len(st.session_state.rows)
            sel_set = set([s.strip().upper() for s in sel])
            st.session_state.rows = [
                r for r in st.session_state.rows
                if str(r.get("Ticker","")).strip().upper() not in sel_set
            ]
            removed = before - len(st.session_state.rows)
            if removed > 0:
                st.success(f"Removed {removed} row(s): {', '.join(sorted(sel_set))}")
            else:
                st.info("No rows removed.")
            do_rerun()
with tcol_sel:
    st.multiselect(
        label="",
        options=tickers,
        default=[],
        key="to_delete",
        placeholder="Select tickers to delete…",
        label_visibility="collapsed"
    )

# ============================== Alignment (DataTables child-rows) ==============================
st.markdown("### Alignment")

def _compute_alignment_counts_weighted(
    stock_row: dict,
    models_tbl: pd.DataFrame,
    var_core: list[str],
    var_mod: list[str],
    w_core: float = 1.0,
    w_mod: float = 0.5,
) -> dict:
    """
    Weighted nearest-median voting:
      - Core vars vote with weight w_core (default 1.0)
      - Moderate vars vote with weight w_mod (default 0.5)
      - Ties (equidistant to FT=1 and FT=0 medians) cast no vote
    Returns absolute weights and percentage shares for FT=1 and FT=0.
    """
    if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
        return {}
    groups = ["FT=1","FT=0"]
    TOL = 1e-9

    counts = {"FT=1": 0.0, "FT=0": 0.0}
    used_core = used_mod = 0

    def vote_for(var, weight):
        nonlocal used_core, used_mod
        xv = pd.to_numeric(stock_row.get(var), errors="coerce")
        if not np.isfinite(xv) or var not in models_tbl.index:
            return
        med = models_tbl.loc[var, groups].astype(float)
        if med.isna().any():
            return
        d1 = abs(xv - med["FT=1"]); d0 = abs(xv - med["FT=0"])
        if abs(d1 - d0) <= TOL:
            return  # ignore ties
        if d1 < d0:
            counts["FT=1"] += weight
        else:
            counts["FT=0"] += weight
        if weight == w_core:
            used_core += 1
        else:
            used_mod += 1

    for v in var_core:
        vote_for(v, w_core)
    for v in var_mod:
        vote_for(v, w_mod)

    total_weight = counts["FT=1"] + counts["FT=0"]
    pct1 = round(100.0 * counts["FT=1"] / total_weight, 0) if total_weight > 0 else 0.0
    pct0 = round(100.0 * counts["FT=0"] / total_weight, 0) if total_weight > 0 else 0.0

    return {
        "FT=1": counts["FT=1"],
        "FT=0": counts["FT=0"],
        "FT1_pct": pct1,
        "FT0_pct": pct0,
        "N_core_used": used_core,
        "N_mod_used": used_mod,
    }

models_tbl = (st.session_state.get("models") or {}).get("models_tbl", pd.DataFrame())
mad_tbl = (st.session_state.get("models") or {}).get("mad_tbl", pd.DataFrame())
var_core = (st.session_state.get("models") or {}).get("var_core", [])
var_mod  = (st.session_state.get("models") or {}).get("var_moderate", [])
SIG_THR = float(st.session_state.get("sig_thresh", 2.0))

if st.session_state.rows and not models_tbl.empty and {"FT=1","FT=0"}.issubset(models_tbl.columns):
    summary_rows, detail_map = [], {}

    # Show PredVol_M under Moderate section (compared vs. Daily_Vol_M medians)
    extra_pred_rows = ["PredVol_M"]
    detail_order = [("Core variables", var_core),
                    ("Moderate variables", var_mod + [r for r in extra_pred_rows if r not in var_mod])]

    for row in st.session_state.rows:
        stock = dict(row)
        tkr = stock.get("Ticker") or "—"
        counts = _compute_alignment_counts_weighted(stock, models_tbl, var_core, var_mod, w_core=1.0, w_mod=0.5)
        if not counts:
            continue

        summary_rows.append({"Ticker": tkr, "FT1_val": counts.get("FT1_pct", 0.0), "FT0_val": counts.get("FT0_pct", 0.0)})

        # ---- child details ----
        drows_grouped = []
        for grp_label, grp_vars in detail_order:
            drows_grouped.append({"__group__": grp_label})
            for v in grp_vars:
                if v == "Daily_Vol_M":
                    continue

                va = pd.to_numeric(stock.get(v), errors="coerce")
                med_var = "Daily_Vol_M" if v in ("PredVol_M",) else v

                v1 = models_tbl.loc[med_var, "FT=1"] if (med_var in models_tbl.index) else np.nan
                v0 = models_tbl.loc[med_var, "FT=0"] if (med_var in models_tbl.index) else np.nan
                m1 = mad_tbl.loc[med_var, "FT=1"] if (not mad_tbl.empty and med_var in mad_tbl.index and "FT=1" in mad_tbl.columns) else np.nan
                m0 = mad_tbl.loc[med_var, "FT=0"] if (not mad_tbl.empty and med_var in mad_tbl.index and "FT=0" in mad_tbl.columns) else np.nan

                if v not in ("PredVol_M",) and pd.isna(va) and pd.isna(v1) and pd.isna(v0):
                    continue

                def _sig(delta, mad):
                    if pd.isna(delta): return np.nan
                    if pd.isna(mad):   return np.nan
                    if mad == 0:
                        return np.inf if abs(delta) > 0 else 0.0
                    return abs(delta) / abs(mad)

                d1 = None if (pd.isna(va) or pd.isna(v1)) else float(va - v1)
                d0 = None if (pd.isna(va) or pd.isna(v0)) else float(va - v0)
                s1 = _sig(d1, float(m1) if pd.notna(m1) else np.nan)
                s0 = _sig(d0, float(m0) if pd.notna(m0) else np.nan)
                
                is_core = v in var_core
                # Flag significance for BOTH Core and Moderate
                sig1 = (not pd.isna(s1)) and (s1 >= SIG_THR)
                sig0 = (not pd.isna(s0)) and (s0 >= SIG_THR)
                significant = sig1 or sig0
                
                # Choose a dominant side (the larger |Δ| among significant sides)
                dominant = None
                if significant:
                    abs1 = -np.inf if (d1 is None or pd.isna(d1) or not sig1) else abs(float(d1))
                    abs0 = -np.inf if (d0 is None or pd.isna(d0) or not sig0) else abs(float(d0))
                    if abs1 >= abs0:
                        dominant = float(d1) if d1 is not None and pd.notna(d1) else None
                    else:
                        dominant = float(d0) if d0 is not None and pd.notna(d0) else None
                
                # Direction string for JS: 'up' | 'down' | ''
                sig_dir = ''
                if dominant is not None and np.isfinite(dominant):
                    sig_dir = 'up' if dominant >= 0 else 'down'
                
                drows_grouped.append({
                    "Variable": v,
                    "Value": None if pd.isna(va) else float(va),
                    "FT1":   None if pd.isna(v1) else float(v1),
                    "FT0":   None if pd.isna(v0) else float(v0),
                    "d_vs_FT1": None if d1 is None else d1,
                    "d_vs_FT0": None if d0 is None else d0,
                    "sig1": bool(sig1),
                    "sig0": bool(sig0),
                    "is_core": is_core,
                    # NEW: single, authoritative significance-direction flag for JS
                    "sig_dir": sig_dir,
                    "significant": bool(significant),
                })

        detail_map[tkr] = drows_grouped

    if summary_rows:
        import streamlit.components.v1 as components
        payload = {"rows": summary_rows, "details": detail_map}
        html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  table.dataTable tbody tr { cursor: pointer; }
  .bar-wrap { display:flex; justify-content:center; align-items:center; gap:6px; }
  .bar { height: 12px; width: 120px; border-radius: 8px; background: #eee; position: relative; overflow: hidden; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; width: 0%; }
  .bar-label { font-size: 11px; white-space: nowrap; color:#374151; min-width: 24px; text-align:center; }
  .blue > span { background:#3b82f6; }  .red  > span { background:#ef4444; }
  #align td:nth-child(2), #align th:nth-child(2),
  #align td:nth-child(3), #align th:nth-child(3) { text-align: center; }
  .child-table { width: 100%; border-collapse: collapse; margin: 2px 0 2px 24px; table-layout: fixed; }
  .child-table th, .child-table td {
    font-size: 11px; padding: 3px 6px; border-bottom: 1px solid #e5e7eb;
    text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .child-table th:first-child, .child-table td:first-child { text-align:left; }
  tr.group-row td { background: #f3f4f6 !important; color:#374151; font-weight:600; text-transform:uppercase; letter-spacing:.02em;
    border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; }
    /* Non-significant Moderates */
    tr.moderate td { background: #f9fafb !important; }
    
    /* Significant rows, Core or Moderate */
    tr.sig_up td   { background: rgba(253, 230, 138, 0.9) !important; }  /* yellow */
    tr.sig_down td { background: rgba(254, 202, 202, 0.9) !important; }  /* red    */
  .col-var { width: 18%; } .col-val { width: 12%; } .col-ft1 { width: 18%; } .col-ft0 { width: 18%; } .col-d1  { width: 17%; } .col-d0  { width: 17%; }
  .pos { color:#059669; } .neg { color:#dc2626; }
</style>
</head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>FT=1</th><th>FT=0</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;
    function barCellBlue(val){ const v=(val==null||isNaN(val))?0:Math.max(0,Math.min(100,val)); return `
      <div class="bar-wrap"><div class="bar blue"><span style="width:${v}%"></span></div><div class="bar-label">${v.toFixed(0)}</div></div>`;}
    function barCellRed(val){ const v=(val==null||isNaN(val))?0:Math.max(0,Math.min(100,val)); return `
      <div class="bar-wrap"><div class="bar red"><span style="width:${v}%"></span></div><div class="bar-label">${v.toFixed(0)}</div></div>`;}
    function formatVal(x){ return (x==null || isNaN(x)) ? '' : Number(x).toFixed(2); }
    function childTableHTML(ticker){
      const rows = data.details[ticker] || [];
      if (!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No variable overlaps for this stock.</div>';
      const cells = rows.map(r=>{
        if (r.__group__) return '<tr class="group-row"><td colspan="6">'+r.__group__+'</td></tr>';
        const v=formatVal(r.Value), f1=formatVal(r.FT1), f0=formatVal(r.FT0);
        const d1=formatVal(r.d_vs_FT1), d0=formatVal(r.d_vs_FT0);
        const c1=(!d1)?'':(parseFloat(d1)>=0?'pos':'neg');
        const c0=(!d0)?'':(parseFloat(d0)>=0?'pos':'neg');
        const isCore=!!r.is_core, s1=isCore&&!!r.sig1, s0=isCore&&!!r.sig0;
        const d1num=(r.d_vs_FT1==null||isNaN(r.d_vs_FT1))?NaN:Number(r.d_vs_FT1);
        const d0num=(r.d_vs_FT0==null||isNaN(r.d_vs_FT0))?NaN:Number(r.d_vs_FT0);
        let rowClass = '';
        if (r.significant) {
          rowClass = (r.sig_dir === 'up') ? 'sig_up'
                   : (r.sig_dir === 'down') ? 'sig_down'
                   : ''; // fallback, should rarely happen
        } else {
          // non-significant Moderate rows stay gray; Core stays default
          if (!isCore) rowClass = 'moderate';
        }
        return `<tr class="${rowClass}">
          <td class="col-var">${r.Variable}</td>
          <td class="col-val">${v}</td>
          <td class="col-ft1">${f1}</td>
          <td class="col-ft0">${f0}</td>
          <td class="col-d1 ${c1}">${d1}</td>
          <td class="col-d0 ${c0}">${d0}</td>
        </tr>`;
      }).join('');
      return `<table class="child-table">
        <colgroup><col class="col-var"/><col class="col-val"/><col class="col-ft1"/><col class="col-ft0"/><col class="col-d1"/><col class="col-d0"/></colgroup>
        <thead><tr><th class="col-var">Variable</th><th class="col-val">Value</th><th class="col-ft1">FT=1 median</th><th class="col-ft0">FT=0 median</th><th class="col-d1">Δ vs FT=1</th><th class="col-d0">Δ vs FT=0</th></tr></thead>
        <tbody>${cells}</tbody></table>`;
    }
    $(function(){
      const table = $('#align').DataTable({
        data: data.rows, responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[ {data:'Ticker'}, {data:'FT1_val', render:(d)=>barCellBlue(d)}, {data:'FT0_val', render:(d)=>barCellRed(d)} ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { const ticker=row.data().Ticker; row.child(childTableHTML(ticker)).show(); $(this).addClass('shown'); }
      });
    });
  </script>
</body></html>
        """
        html = html.replace("%%PAYLOAD%%", json.dumps(payload))
        import streamlit.components.v1 as components
        components.html(html, height=620, scrolling=True)
    else:
        st.info("No eligible rows yet. Add manual stocks and/or ensure FT=1/FT=0 medians are built.")
elif st.session_state.rows and (models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns)):
    st.info("Upload DB and click **Build model stocks** to compute FT=1/FT=0 medians first.")
else:
    st.info("Add at least one stock above to compute alignment.")
