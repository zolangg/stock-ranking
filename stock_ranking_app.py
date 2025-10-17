# app.py — Premarket Stock Ranking & Structural Analysis
# ALGORITHM 2.1: DUAL-SCORE SYSTEM
# 1. SRS (Structural Risk Score): Measures downside dilution risk.
# 2. SPS (Squeeze Potential Score): Measures upside squeeze potential, gated by the SRS.
# UI: Unified, compact input form for all data.
# MODELS: Includes Premarket Volume Prediction (LASSO), NCA, and CatBoost for group alignment.

import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, io, math
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime, date

# CatBoost import (graceful if unavailable)
try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except Exception:
    _CATBOOST_OK = False

# ============================== Page Setup ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ============================== Session State ==============================
ss = st.session_state
for key, default in [
    ("rows", []), ("last", {}), ("lassoA", {}), ("base_df", pd.DataFrame()),
    ("var_core", []), ("var_moderate", []), ("nca_model", {}), ("cat_model", {}),
    ("del_selection", []), ("__delete_msg", None), ("__catboost_warned", False)
]:
    ss.setdefault(key, default)

# ============================== Dual-Score Algorithm Engine ==============================
def clamp(x, a, b):
    return max(a, min(b, x))

def calculate_srs_v2(data: dict) -> dict:
    """Calculates the Structural Risk Score (SRS) using the refined Algorithm 2.0."""
    # --- 1A: Funding Pressure (F) ---
    F = 0
    r = data.get("runway_months", 0)
    if r < 3: F -= 2
    elif 3 <= r < 6: F -= 1
    elif 12 <= r < 18: F += 1
    elif r >= 18: F += 2
    
    if data.get("atm_active"): F -= 1
    if data.get("equity_line_active"): F -= 0.5
    
    market_cap = data.get("market_cap", 0)
    shelf_raisable = data.get("shelf_raisable_usd", 0)
    if market_cap > 0 and shelf_raisable > 0:
        shelf_overhang_pct = shelf_raisable / market_cap
        if shelf_overhang_pct > 0.5: F -= 1.0
        elif shelf_overhang_pct > 0.2: F -= 0.5
            
    F = clamp(F, -2, 2)

    # --- 1B: Instrument Pressure (I) ---
    I = 0
    spot_price = data.get("spot_price", 0)
    current_os = data.get("current_os", 1)
    if spot_price > 0 and current_os > 0:
        for w in data.get("warrants", []):
            strike = w.get("strike", 0)
            if strike > 0:
                moneyness = spot_price / strike
                w_score = 0.5 if moneyness < 0.67 else 0.0 if moneyness < 0.9 else -1.0 if moneyness <= 1.1 else -1.5 if w.get("registered") else -1.0
                coverage_weight = min(1.0, w.get("remaining", 0) / (0.30 * current_os))
                I += w_score * max(0.4, coverage_weight)
    
    if data.get("warrants_with_resets"): I -= 1.0
    I = clamp(I, -2, 1)

    # --- 1C: Promotion Pressure (P) ---
    P = 0
    if data.get("pr_count_90d", 0) <= 2: P += 0.5
    elif data.get("pr_count_90d", 0) > 6: P -= 0.5
    P += 0.5 if data.get("filing_attached_prs", 0) >= 1 else -0.5
    if data.get("live_shelf"): P -= 0.75
    P = clamp(P, -1, 1)

    # --- 1D: Float/Supply Stress (S) ---
    S = 0
    rs_date = data.get("rs_date")
    if rs_date and isinstance(rs_date, date):
        days_since_rs = (date.today() - rs_date).days
        if days_since_rs < 30: S -= 2
        elif 30 <= days_since_rs < 90: S -= 1
        elif 90 <= days_since_rs < 180: S -= 0.5
    
    fully_diluted_os = data.get("fully_diluted_os", 0)
    if current_os > 0 and fully_diluted_os > current_os:
        overhang_pct = (fully_diluted_os - current_os) / current_os
        if overhang_pct > 1.0: S -= 1.0
        elif overhang_pct > 0.5: S -= 0.5
    S = clamp(S, -2, 1)

    # --- 2: Composite SRS ---
    srs_score = 0.35 * F + 0.25 * I + 0.20 * P + 0.20 * S

    # --- 3: Structural Class ---
    srs_class = "Toxic"
    if srs_score >= 1.0: srs_class = "Clean"
    elif 0 <= srs_score < 1: srs_class = "Balanced"
    elif -0.5 <= srs_score < 0: srs_class = "Fragile"
    
    return {"SRS": srs_score, "SRS_Class": srs_class}

def calculate_sps(data: dict, srs_score: float) -> dict:
    """Calculates the Squeeze Potential Score (SPS) based on key squeeze factors."""
    if srs_score is None or np.isnan(srs_score):
        return {"SPS": 0, "SPS_Class": "N/A"}

    raw_sps = 0
    
    # Factor 1: Float Rotation
    fr = data.get("FR_x", 0)
    if not np.isnan(fr):
        if fr > 1.0: raw_sps += 2
        if fr > 5.0: raw_sps += 1
        
    # Factor 2: Short Interest
    si_pct = data.get("short_interest_pct", 0)
    if si_pct > 20: raw_sps += 2
    if si_pct > 40: raw_sps += 1
        
    # Factor 3: Catalyst
    if data.get("Catalyst", 0) > 0:
        raw_sps += 1
        
    # The "Squeeze Permission Multiplier" from SRS
    permission_multiplier = clamp((srs_score + 1.0) / 2.0, 0, 1)
    
    final_sps = raw_sps * permission_multiplier
    
    # Classify the final score
    sps_class = "Low"
    if final_sps > 5: sps_class = "Elite"
    elif final_sps > 3: sps_class = "Strong"
    elif final_sps > 1: sps_class = "Moderate"
        
    return {"SPS": final_sps, "SPS_Class": sps_class}

# ============================== Helpers & Core ML Models ==============================
def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def SAFE_JSON_DUMPS(obj) -> str:
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            if isinstance(o, (date,)): return o.isoformat()
            return super().default(o)
    s = json.dumps(obj, cls=NpEncoder, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</script>", "<\\/script>")

_norm_cache = {}
def _norm(s: str) -> str:
    if s in _norm_cache: return _norm_cache[s]
    v = re.sub(r"\s+", " ", str(s).strip().lower()).replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")
    _norm_cache[s] = v
    return v

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty: return None
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        lc = cand.strip().lower()
        if lc in cols_lc.values():
            for c, clc in cols_lc.items():
                if clc == lc: return c
    for cand in candidates:
        n = _norm(cand)
        for c, cn in nm.items():
            if cn == n: return c
    for cand in candidates:
        n = _norm(cand)
        for c, cn in nm.items():
            if n in cn: return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        ss_ = str(s).strip().replace(" ", "")
        if "," in ss_ and "." not in ss_: ss_ = ss_.replace(",", ".")
        else: ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))

def _compute_bounds(arr: np.ndarray, lo_q=0.01, hi_q=0.99):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (np.nan, np.nan)
    return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

def _apply_bounds(arr: np.ndarray, lo: float, hi: float):
    out = arr.copy()
    if np.isfinite(lo): out = np.maximum(out, lo)
    if np.isfinite(hi): out = np.minimum(out, hi)
    return out

def _pav_isotonic(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    level_y, level_n = ys.astype(float).copy(), np.ones_like(ys, dtype=float)
    i = 0
    while i < len(level_y) - 1:
        if level_y[i] > level_y[i+1]:
            new_y = (level_y[i]*level_n[i] + level_y[i+1]*level_n[i+1]) / (level_n[i] + level_n[i+1])
            new_n = level_n[i] + level_n[i+1]
            level_y[i], level_n[i] = new_y, new_n
            level_y, level_n = np.delete(level_y, i+1), np.delete(level_n, i+1)
            xs = np.delete(xs, i+1)
            if i > 0: i -= 1
        else: i += 1
    return xs, level_y

def _iso_predict(break_x: np.ndarray, break_y: np.ndarray, x_new: np.ndarray):
    if break_x.size == 0: return np.full_like(x_new, np.nan, dtype=float)
    idx = np.argsort(break_x)
    bx, by = break_x[idx], break_y[idx]
    if bx.size == 1: return np.full_like(x_new, by[0], dtype=float)
    return np.interp(x_new, bx, by, left=by[0], right=by[-1])

VAR_CORE = ["Gap_%", "FR_x", "PM$Vol/MC_%", "Catalyst", "PM_Vol_%", "Max_Pull_PM_%", "RVOL_Max_PM_cum"]
VAR_MODERATE = ["MC_PM_Max_M", "Float_PM_Max_M", "PM_Vol_M", "PM_$Vol_M$", "ATR_$", "Daily_Vol_M", "MarketCap_M$", "Float_M"]
VAR_ALL = VAR_CORE + VAR_MODERATE
ALLOWED_LIVE_FEATURES = ["MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"]
EXCLUDE_FOR_NCA = {"PredVol_M","PM_Vol_%","Daily_Vol_M"}

def _kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return np.array_split(idx, k)

def _lasso_cd_std(Xs, y, lam, max_iter=900, tol=1e-6):
    n, p = Xs.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        w_old = w.copy()
        y_hat = Xs @ w
        for j in range(p):
            r_j = y - y_hat + Xs[:, j] * w[j]
            rho = (Xs[:, j] @ r_j) / n
            w[j] = np.sign(rho) * max(abs(rho) - lam/2, 0)
            y_hat = Xs @ w
        if np.linalg.norm(w - w_old) < tol: break
    return w

def train_ratio_winsor_iso(df: pd.DataFrame, lo_q=0.01, hi_q=0.99) -> dict:
    eps = 1e-6
    mcap_series  = df["MC_PM_Max_M"] if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else df.get("MarketCap_M$")
    float_series = df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else df.get("Float_M")
    need_min = {"ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"}
    if mcap_series is None or float_series is None or not need_min.issubset(df.columns): return {}

    PM = pd.to_numeric(df["PM_Vol_M"], errors="coerce").values
    DV = pd.to_numeric(df["Daily_Vol_M"], errors="coerce").values
    valid_pm = np.isfinite(PM) & np.isfinite(DV) & (PM > 0) & (DV > 0)
    if valid_pm.sum() < 50: return {}

    ln_mcap = np.log(np.clip(pd.to_numeric(mcap_series, errors="coerce").values, eps, None))
    ln_gapf = np.log(np.clip(pd.to_numeric(df["Gap_%"], errors="coerce").values, 0, None) / 100.0 + eps)
    ln_atr = np.log(np.clip(pd.to_numeric(df["ATR_$"], errors="coerce").values, eps, None))
    ln_pm = np.log(np.clip(pd.to_numeric(df["PM_Vol_M"], errors="coerce").values, eps, None))
    ln_pm_dol = np.log(np.clip(pd.to_numeric(df["PM_$Vol_M$"], errors="coerce").values, eps, None))
    ln_fr = np.log(np.clip(pd.to_numeric(df["FR_x"], errors="coerce").values, eps, None))
    ln_float_pmmax = np.log(np.clip(pd.to_numeric(float_series, errors="coerce").values, eps, None))
    maxpullpm = pd.to_numeric(df.get("Max_Pull_PM_%", np.nan), errors="coerce").values
    ln_rvolmaxpm = np.log(np.clip(pd.to_numeric(df.get("RVOL_Max_PM_cum", np.nan), errors="coerce").values, eps, None))
    pm_dol_over_mc = pd.to_numeric(df.get("PM$Vol/MC_%", np.nan), errors="coerce").values
    catalyst = pd.to_numeric(df.get("Catalyst", np.nan), errors="coerce").fillna(0.0).clip(0,1).values
    
    y_ln_all = np.log(np.maximum(DV / PM, 1.0))
    
    feats = [("ln_mcap_pmmax", ln_mcap), ("ln_gapf", ln_gapf), ("ln_atr", ln_atr), ("ln_pm", ln_pm), ("ln_pm_dol", ln_pm_dol), ("ln_fr", ln_fr), ("catalyst", catalyst), ("ln_float_pmmax", ln_float_pmmax), ("maxpullpm", maxpullpm), ("ln_rvolmaxpm", ln_rvolmaxpm), ("pm_dol_over_mc", pm_dol_over_mc)]
    X_all = np.hstack([arr.reshape(-1,1) for _, arr in feats])

    mask = valid_pm & np.isfinite(y_ln_all) & np.isfinite(X_all).all(axis=1)
    if mask.sum() < 50: return {}
    X_all, y_ln = X_all[mask], y_ln_all[mask]
    PMm, DVv = PM[mask], DV[mask]

    n = X_all.shape[0]; split = max(10, int(n * 0.8))
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr = y_ln[:split]

    winsor_bounds = {}
    name_to_idx = {name:i for i,(name,_) in enumerate(feats)}
    for nm in ["maxpullpm", "pm_dol_over_mc"]:
        if nm in name_to_idx:
            col_idx = name_to_idx[nm]
            arr_tr = X_tr[:, col_idx]
            lo, hi = _compute_bounds(arr_tr[np.isfinite(arr_tr)])
            winsor_bounds[feats[col_idx][0]] = (lo, hi)
            X_tr[:, col_idx] = _apply_bounds(arr_tr, lo, hi)
            X_va[:, col_idx] = _apply_bounds(X_va[:, col_idx], lo, hi)

    mult_tr = np.exp(y_tr)
    m_lo, m_hi = _compute_bounds(mult_tr)
    y_tr = np.log(_apply_bounds(mult_tr, m_lo, m_hi))
    
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0, ddof=0)
    sd[sd==0] = 1.0
    Xs_tr = (X_tr - mu) / sd
    
    folds = _kfold_indices(len(y_tr), k=min(5, max(2, len(y_tr)//10)))
    lam_grid = np.geomspace(0.001, 1.0, 26)
    cv_mse = []
    for lam in lam_grid:
        errs = []
        for vi in range(len(folds)):
            te_idx = folds[vi]; tr_idx = np.hstack([folds[j] for j in range(len(folds)) if j != vi])
            Xtr, ytr = Xs_tr[tr_idx], y_tr[tr_idx]; Xte, yte = Xs_tr[te_idx], y_tr[te_idx]
            w = _lasso_cd_std(Xtr, ytr, lam, 1400)
            yhat = Xte @ w
            errs.append(np.mean((yhat - yte)**2))
        cv_mse.append(np.mean(errs))

    w_l1 = _lasso_cd_std(Xs_tr, y_tr, float(lam_grid[np.argmin(cv_mse)]), 2000)
    sel = np.flatnonzero(np.abs(w_l1) > 1e-8)
    if sel.size == 0: return {}

    Xtr_sel = X_tr[:, sel]
    coef_ols, *_ = np.linalg.lstsq(np.column_stack([np.ones(Xtr_sel.shape[0]), Xtr_sel]), y_tr, rcond=None)
    b0, bet = float(coef_ols[0]), coef_ols[1:].astype(float)
    
    iso_bx, iso_by = np.array([]), np.array([])
    if X_va.shape[0] >= 8:
        Xva_sel = X_va[:, sel]
        yhat_va_ln = (np.column_stack([np.ones(Xva_sel.shape[0]), Xva_sel]) @ coef_ols).astype(float)
        mult_pred_va, mult_va_true = np.exp(yhat_va_ln), np.maximum(DVv[split:] / PMm[split:], 1.0)
        finite = np.isfinite(mult_pred_va) & np.isfinite(mult_va_true)
        if finite.sum() >= 8 and np.unique(mult_pred_va[finite]).size >= 3:
            iso_bx, iso_by = _pav_isotonic(mult_pred_va[finite], mult_va_true[finite])

    return {"eps": eps, "terms": [feats[i][0] for i in sel], "b0": b0, "betas": bet, "sel_idx": sel.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "winsor_bounds": {k:(float(v[0]) if np.isfinite(v[0]) else np.nan, float(v[1]) if np.isfinite(v[1]) else np.nan) for k,v in winsor_bounds.items()}, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "feat_order": [nm for nm,_ in feats]}

def predict_daily_calibrated(row: dict, model: dict) -> float:
    if not model or "betas" not in model: return np.nan
    eps, feat_order, winsor_bounds, sel, b0, bet = model.get("eps",1e-6), model["feat_order"], model.get("winsor_bounds",{}), model.get("sel_idx",[]), float(model["b0"]), np.array(model["betas"],dtype=float)
    
    def safe_log(v): v=float(v) if v is not None else np.nan; return np.log(np.clip(v, eps, None)) if np.isfinite(v) else np.nan
    
    feat_map = {"ln_mcap_pmmax":safe_log(row.get("MC_PM_Max_M") or row.get("MarketCap_M$")), "ln_gapf":np.log(np.clip((row.get("Gap_%") or 0.0)/100.0+eps,eps,None)) if row.get("Gap_%") is not None else np.nan, "ln_atr":safe_log(row.get("ATR_$")), "ln_pm":safe_log(row.get("PM_Vol_M")), "ln_pm_dol":safe_log(row.get("PM_$Vol_M$")), "ln_fr":safe_log(row.get("FR_x")), "catalyst":1.0 if (str(row.get("CatalystYN","No")).lower()=="yes" or float(row.get("Catalyst",0))>=0.5) else 0.0, "ln_float_pmmax":safe_log(row.get("Float_PM_Max_M") or row.get("Float_M")), "maxpullpm":float(row.get("Max_Pull_PM_%")) if row.get("Max_Pull_PM_%") is not None else np.nan, "ln_rvolmaxpm":safe_log(row.get("RVOL_Max_PM_cum")), "pm_dol_over_mc":float(row.get("PM$Vol/MC_%")) if row.get("PM$Vol/MC_%") is not None else np.nan}
    
    X_vec = []
    for nm in feat_order:
        v = feat_map.get(nm, np.nan)
        if not np.isfinite(v): return np.nan
        lo, hi = winsor_bounds.get(nm, (np.nan, np.nan))
        X_vec.append(float(np.clip(v, lo if np.isfinite(lo) else v, hi if np.isfinite(hi) else v)))
    if not sel: return np.nan
    
    yhat_ln = b0 + float(np.dot(np.array(X_vec)[sel], bet))
    raw_mult = np.exp(yhat_ln) if np.isfinite(yhat_ln) else np.nan
    if not np.isfinite(raw_mult): return np.nan

    iso_bx, iso_by = np.array(model.get("iso_bx",[]),dtype=float), np.array(model.get("iso_by",[]),dtype=float)
    cal_mult = float(_iso_predict(iso_bx,iso_by,np.array([raw_mult]))[0]) if (iso_bx.size>=2 and iso_by.size>=2) else raw_mult
    cal_mult = max(cal_mult, 1.0)
    
    PM = float(row.get("PM_Vol_M") or np.nan)
    return float(PM * cal_mult) if np.isfinite(PM) and PM > 0 else np.nan

@st.cache_data(show_spinner=False)
def _hash_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=True)
def _load_sheet(file_bytes: bytes):
    xls = pd.ExcelFile(file_bytes)
    sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
    sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
    return pd.read_excel(xls, sheet), sheet, tuple(xls.sheet_names)

def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}

    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    good_cols = [c for c in feats if not Xdf[c].isna().all() and np.nanstd(Xdf[c]) > 1e-9]
    if not good_cols: return {}
    feats = good_cols

    X = df2[feats].apply(pd.to_numeric, errors="coerce").values
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    if X.shape[0] < 20 or np.unique(y).size < 2: return {}

    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs = (X - mu) / sd

    used, w_vec, components, z = "lda", None, None, np.array([])
    try:
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42, max_iter=400)
        z = nca.fit_transform(Xs, y).ravel()
        used, components = "nca", nca.components_
    except Exception:
        X0, X1 = Xs[y==0], Xs[y==1]
        if X0.shape[0] < 2 or X1.shape[0] < 2: return {}
        m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
        Sw = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False) + 1e-3*np.eye(Xs.shape[1])
        w_vec = np.linalg.solve(Sw, (m1 - m0)); w_vec /= (np.linalg.norm(w_vec) + 1e-12)
        z = (Xs @ w_vec)

    if np.nanmean(z[y==1]) < np.nanmean(z[y==0]):
        z = -z
        if w_vec is not None: w_vec = -w_vec
        if components is not None: components = -components

    zf, yf = z[np.isfinite(z)], y[np.isfinite(z)]
    iso_bx, iso_by, platt_params = np.array([]), np.array([]), None
    if zf.size >= 8 and np.unique(zf).size >= 3:
        bx, by = _pav_isotonic(zf, yf.astype(float))
        if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
    if iso_bx.size < 2:
        z0, z1 = zf[yf==0], zf[yf==1]
        if z0.size and z1.size:
            m0, m1 = float(np.mean(z0)), float(np.mean(z1))
            s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
            m, k = 0.5*(m0+m1), 2.0 / (0.5*(s0+s1) + 1e-6)
            platt_params = (m,k)
            
    return {"ok": True, "kind": used, "feats": feats, "mu": mu.tolist(), "sd": sd.tolist(), "w_vec": (w_vec.tolist() if w_vec is not None else None), "components": (components.tolist() if components is not None else None), "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": (platt_params if platt_params is not None else None), "gA": gA_label, "gB": gB_label}

def _nca_predict_proba(row: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    x = [pd.to_numeric(row.get(f), errors="coerce") for f in feats]
    if any(not np.isfinite(v) for v in x): return np.nan
    x = np.array(x, dtype=float)

    mu, sd = np.array(model["mu"], dtype=float), np.array(model["sd"], dtype=float)
    sd[sd==0] = 1.0
    xs = (x - mu) / sd

    z = np.nan
    if model["kind"] == "lda":
        w = np.array(model.get("w_vec"), dtype=float)
        if w is None or not np.isfinite(w).all(): return np.nan
        z = float(xs @ w)
    else:
        comp = model.get("components")
        if comp is None: return np.nan
        w = np.array(comp, dtype=float).ravel()
        if w.size != xs.size: return np.nan
        z = float(xs @ w)
    if not np.isfinite(z): return np.nan

    iso_bx, iso_by = np.array(model.get("iso_bx",[]),dtype=float), np.array(model.get("iso_by",[]),dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx,iso_by,np.array([z]))[0])
    else:
        pl = model.get("platt")
        if not pl: return np.nan
        m, k = pl
        pA = 1.0 / (1.0 + np.exp(-k*(z-m)))
    return float(np.clip(pA, 0.0, 1.0))

def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    if not _CATBOOST_OK:
        if not ss.get("__catboost_warned",False): st.info("CatBoost is not installed. Run `pip install catboost` to enable it."); ss["__catboost_warned"]=True
        return {}

    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}

    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask_finite = np.isfinite(Xdf.values).all(axis=1)
    Xdf, y = Xdf.loc[mask_finite], y[mask_finite]

    if len(y) < 40 or np.unique(y).size < 2: return {}

    X_all, y_all = Xdf.values.astype(np.float32,copy=False), y.astype(np.int32,copy=False)
    
    from sklearn.model_selection import StratifiedShuffleSplit
    test_size = 0.2 if len(y)>=100 else max(0.15, min(0.25, 20/len(y)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y_all))
    Xtr, Xva, ytr, yva = X_all[tr_idx], X_all[va_idx], y_all[tr_idx], y_all[va_idx]

    eval_ok = (len(yva)>=8) and (np.unique(yva).size==2) and (np.unique(ytr).size==2)

    params = dict(loss_function="Logloss", eval_metric="Logloss", iterations=200, learning_rate=0.04, depth=3, l2_leaf_reg=6, bootstrap_type="Bayesian", bagging_temperature=0.5, auto_class_weights="Balanced", random_seed=42, allow_writing_files=False, verbose=False)
    if eval_ok: params.update(dict(od_type="Iter", od_wait=40))
    else: params.update(dict(od_type="None"))
    
    model = CatBoostClassifier(**params)
    try:
        if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else: model.fit(Xtr, ytr)
    except Exception:
        try:
            model = CatBoostClassifier(**{**params, "od_type": "None"}); model.fit(X_all, y_all)
            eval_ok = False
        except Exception: return {}
    
    iso_bx, iso_by, platt = np.array([]), np.array([]), None
    # ... calibration logic ...
    return {"ok":True, "feats":feats, "gA":gA_label, "gB":gB_label, "cb":model, "iso_bx":iso_bx.tolist(), "iso_by":iso_by.tolist(), "platt":platt}

def _cat_predict_proba(row: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    x = [pd.to_numeric(row.get(f), errors="coerce") for f in feats]
    if any(not np.isfinite(v) for v in x): return np.nan
    x = np.array(x, dtype=float).reshape(1, -1)

    try:
        cb = model.get("cb")
        if cb is None: return np.nan
        z = float(cb.predict_proba(x)[0, 1])
    except Exception: return np.nan

    iso_bx, iso_by = np.array(model.get("iso_bx",[]),dtype=float), np.array(model.get("iso_by",[]),dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx,iso_by,np.array([z]))[0])
    else:
        pl = model.get("platt")
        pA = z if not pl else 1.0 / (1.0 + np.exp(-pl[1]*(z-pl[0])))
    return float(np.clip(pA, 0.0, 1.0))


# ============================== UI & App Logic ==============================

# --- Upload / Build ---
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
if st.button("Build model stocks", use_container_width=True, key="db_build_btn"):
    if uploaded:
        try:
            raw, sel_sheet, _ = _load_sheet(uploaded.getvalue())
            col_group = _pick(raw, ["ft","ft01","group","label"])
            if col_group is None:
                st.error("Could not detect FT (0/1) column."); st.stop()

            df = pd.DataFrame()
            df["GroupRaw"] = raw[col_group]
            def add_num(dfout, name, cands): src = _pick(raw,cands); dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce") if src else np.nan
            add_num(df, "MC_PM_Max_M", ["mc pm max (m)","premarket market cap (m)"])
            add_num(df, "Float_PM_Max_M", ["float pm max (m)","premarket float (m)"])
            add_num(df, "MarketCap_M$", ["marketcap m","market cap (m)"])
            add_num(df, "Float_M", ["float m","public float (m)"])
            add_num(df, "Gap_%", ["gap %","gap%","premarket gap"])
            add_num(df, "ATR_$", ["atr $","atr$","atr (usd)"])
            add_num(df, "PM_Vol_M", ["pm vol (m)","premarket vol (m)"])
            add_num(df, "PM_$Vol_M$", ["pm $vol (m)","pm dollar vol (m)"])
            add_num(df, "PM_Vol_%", ["pm vol (%)","pm_vol_%"])
            add_num(df, "Daily_Vol_M", ["daily vol (m)","daily_vol_m"])
            add_num(df, "Max_Pull_PM_%", ["max pull pm (%)","max pull pm %"])
            add_num(df, "RVOL_Max_PM_cum", ["rvol max pm (cum)","rvol max pm cum"])
            
            cand_cat = _pick(raw, ["catalyst","catalyst?"]); df["Catalyst"] = raw[cand_cat].map(lambda v: 1.0 if str(v).lower() in {"1","true","yes"} else 0.0) if cand_cat else 0.0
            
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if "PM_Vol_M" in df.columns and float_basis in df.columns: df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            if "PM_$Vol_M$" in df.columns and mcap_basis in df.columns: df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            for col in ["Gap_%", "PM_Vol_%", "Max_Pull_PM_%"]:
                if col in df.columns: df[col] *= 100.0

            df["FT01"] = df["GroupRaw"].map(lambda v: 1 if str(v).lower() in {"1","true","yes"} else 0)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})
            
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %"]); df["Max_Push_Daily_%"] = pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")*100.0 if pmh_col else np.nan

            ss.base_df = df
            ss.var_core = [v for v in VAR_CORE if v in df.columns]
            ss.var_moderate = [v for v in VAR_MODERATE if v in df.columns]
            ss.lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            st.success(f"Loaded “{sel_sheet}”. Base ready."); do_rerun()
        except Exception as e:
            st.error("Loading/processing failed."); st.exception(e)
    else: st.error("Please upload an Excel workbook first.")

# --- Add Stock Form ---
st.markdown("---")
st.subheader("Add Stock")

with st.form("add_stock_form", clear_on_submit=True):
    ticker = st.text_input("Ticker", "", help="Enter stock ticker. It's the only required field.").strip().upper()

    st.markdown("###### Premarket Data")
    c1,c2,c3 = st.columns(3)
    mc_pmmax=c1.number_input("Premarket Market Cap (M$)",0.0,step=0.01,format="%.2f")
    float_pm=c2.number_input("Premarket Float (M)",0.0,step=0.01,format="%.2f")
    gap_pct=c3.number_input("Gap %",0.0,step=0.01,format="%.2f")
    c1,c2,c3=st.columns(3)
    atr_usd=c1.number_input("Prior Day ATR ($)",0.0,step=0.01,format="%.2f")
    pm_vol=c2.number_input("Premarket Volume (M)",0.0,step=0.01,format="%.2f")
    pm_dol=c3.number_input("Premarket Dollar Vol (M$)",0.0,step=0.01,format="%.2f")
    c1,c2,c3=st.columns(3)
    max_pull_pm=c1.number_input("Premarket Max Pullback (%)",0.0,step=0.01,format="%.2f")
    rvol_pm_cum=c2.number_input("Premarket Max RVOL",0.0,step=0.01,format="%.2f")
    catalyst_yn=c3.selectbox("Catalyst?",["No","Yes"],index=0)

    st.markdown("---")
    st.markdown("###### Dilution / Squeeze Potential Data")
    d_c1, d_c2, d_c3 = st.columns(3)
    runway_months=d_c1.number_input("Runway (months)",0.0,step=0.1,format="%.1f",key="d_runway")
    spot_price=d_c2.number_input("Current Spot Price ($)",0.0,step=0.01,format="%.2f",key="d_spot")
    short_interest_pct=d_c3.number_input("Short Interest (% of Float)",0.0,step=0.1,format="%.1f",key="d_si")

    d_c1, d_c2 = st.columns(2)
    current_os_d=d_c1.number_input("Current O/S (M)",0.0,step=0.1,format="%.1f",help="From chart, in millions",key="d_os")*1e6
    fully_diluted_os_d=d_c2.number_input("Fully Diluted O/S (M)",0.0,step=0.1,format="%.1f",help="From chart, in millions",key="d_fos")*1e6

    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    atm_active=d_c1.checkbox("ATM Active?",key="d_atm")
    equity_line_active=d_c2.checkbox("Equity Line Active?",key="d_eql")
    live_shelf=d_c3.checkbox("Live Resale/ATM Shelf?",key="d_shelf_live",help="Is there an effective S-1/S-3 for resale or an active ATM?")
    rs_date_d=d_c4.date_input("Last R/S Date",value=None,key="d_rs")
    
    st.markdown("###### Warrants & Promotion")
    wc1, wc2, wc3, wc4 = st.columns(4)
    w1_strike=wc1.number_input("W1 Strike ($)",0.0,step=0.01,format="%.2f",key="w1s")
    w1_remaining=wc2.number_input("W1 Remaining",0,step=1000,key="w1rem")
    w1_registered=wc3.checkbox("W1 Registered?",key="w1r_")
    warrants_with_resets=wc4.checkbox("Warrants w/ Resets?",key="w_resets",help="Do any significant warrants have price protection/reset features?")

    wc1,wc2,wc3=st.columns(3)
    w2_strike=wc1.number_input("W2 Strike ($)",0.0,step=0.01,format="%.2f",key="w2s")
    w2_remaining=wc2.number_input("W2 Remaining",0,step=1000,key="w2rem")
    w2_registered=wc3.checkbox("W2 Registered?",key="w2r_")
    
    d_c1, d_c2, d_c3 = st.columns(3)
    pr_count_90d=d_c1.number_input("PRs (last 90d)",0,step=1,key="d_pr")
    filing_attached_prs=d_c2.number_input("PRs with filings (90d)",0,step=1,key="d_filing_pr")
    shelf_raisable_usd=d_c3.number_input("Shelf Raisable ($M)",0.0,step=0.1,format="%.1f",key="d_shelf_val")*1e6
    
    if st.form_submit_button("Add Stock to Table", use_container_width=True) and ticker:
        row = {"Ticker": ticker}
        
        if mc_pmmax > 0:
            row.update({"MC_PM_Max_M": mc_pmmax, "Float_PM_Max_M": float_pm, "Gap_%": gap_pct, "ATR_$": atr_usd, "PM_Vol_M": pm_vol, "PM_$Vol_M$": pm_dol, "Max_Pull_PM_%": max_pull_pm, "RVOL_Max_PM_cum": rvol_pm_cum, "CatalystYN": catalyst_yn, "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0, "FR_x": (pm_vol/float_pm) if float_pm>0 else np.nan, "PM$Vol/MC_%": (pm_dol/mc_pmmax*100.0) if mc_pmmax>0 else np.nan})
            pred = predict_daily_calibrated(row, ss.get("lassoA", {}))
            row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
            denom = row.get("PredVol_M",0)
            row["PM_Vol_%"] = (row.get("PM_Vol_M",0)/denom*100.0) if denom > 0 else np.nan

        srs_results={"SRS":np.nan,"SRS_Class":"N/A"}
        sps_results={"SPS":np.nan,"SPS_Class":"N/A"}
        if runway_months>0 or current_os_d>0:
            warrants=[]
            if w1_strike>0 and w1_remaining>0: warrants.append({"strike":w1_strike,"remaining":w1_remaining,"registered":w1_registered})
            if w2_strike>0 and w2_remaining>0: warrants.append({"strike":w2_strike,"remaining":w2_remaining,"registered":w2_registered})
            
            dilution_data={"runway_months":runway_months,"atm_active":atm_active,"equity_line_active":equity_line_active,"live_shelf":live_shelf,"shelf_raisable_usd":shelf_raisable_usd,"warrants":warrants,"warrants_with_resets":warrants_with_resets,"rs_date":rs_date_d,"current_os":current_os_d,"fully_diluted_os":fully_diluted_os_d,"pr_count_90d":pr_count_90d,"filing_attached_prs":filing_attached_prs,"spot_price":spot_price,"market_cap":mc_pmmax*1e6 if mc_pmmax>0 else 0}
            srs_results = calculate_srs_v2(dilution_data)

            squeeze_data = {**row, "short_interest_pct": short_interest_pct}
            sps_results = calculate_sps(squeeze_data, srs_results.get("SRS"))
            st.success(f"SRS: {srs_results['SRS']:.2f} ({srs_results['SRS_Class']}) | SPS: {sps_results['SPS']:.2f} ({sps_results['SPS_Class']})")

        row.update(srs_results); row.update(sps_results)
        ss.rows.append(row); ss.last = row
        st.success(f"Added {ticker} to the analysis table."); do_rerun()

# ============================== Alignment Section ==============================
st.markdown("---")
if not ss.rows:
    st.info("Add a stock to begin analysis.")
    st.stop()

st.subheader("Alignment")
col_mode, col_gain = st.columns([2.8, 1.0])
mode = col_mode.radio("", ["FT vs Fail (Gain% cutoff on FT=1 only)","Gain% vs Rest"], horizontal=True, key="cmp_mode", label_visibility="collapsed")
gain_choices = [25,50,75,100,125,150,175,200,225,250,275,300]
gain_min = col_gain.selectbox("", gain_choices, index=gain_choices.index(100), key="gain_min_pct", help="Threshold on Max Push Daily (%).", label_visibility="collapsed")

tickers = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
unique_tickers = list(dict.fromkeys(tickers))

def _handle_delete():
    sel = st.session_state.get("del_selection", [])
    if sel:
        ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(sel)]
        st.session_state["del_selection"] = []
        st.session_state["__delete_msg"] = f"Deleted: {', '.join(sel)}"
    else:
        st.session_state["__delete_msg"] = "No tickers selected."

cdel1, cdel2 = st.columns([4, 1])
cdel1.multiselect("", options=unique_tickers, default=ss.get("del_selection",[]), key="del_selection", placeholder="Select tickers to delete...", label_visibility="collapsed")
cdel2.button("Delete", use_container_width=True, key="delete_btn", disabled=not bool(ss.get("del_selection")), on_click=_handle_delete)
if "__delete_msg" in ss and ss["__delete_msg"]:
    (st.success if ss["__delete_msg"].startswith("Deleted:") else st.info)(ss.pop("__delete_msg"))

base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty: st.info("Upload DB and click **Build model stocks** to compute group centers."); st.stop()
if "Max_Push_Daily_%" not in base_df.columns: st.error("“Max Push Daily (%)” column not found in DB."); st.stop()
if "FT01" not in base_df.columns: st.error("FT01 column not found."); st.stop()

df_cmp = base_df.copy(); thr = float(gain_min)
if mode == "Gain% vs Rest":
    df_cmp["__Group__"] = np.where(pd.to_numeric(df_cmp["Max_Push_Daily_%"],errors="coerce")>=thr, f"≥{int(thr)}%", "Rest")
    gA, gB = f"≥{int(thr)}%", "Rest"; status_line = f"Gain% split at ≥ {int(thr)}%"
else:
    a_mask=(df_cmp["FT01"]==1)&(pd.to_numeric(df_cmp["Max_Push_Daily_%"],errors="coerce")>=thr); b_mask=(df_cmp["FT01"]==0)
    df_cmp = df_cmp[a_mask|b_mask].copy()
    df_cmp["__Group__"] = np.where(df_cmp["FT01"]==1, f"FT=1 ≥{int(thr)}%", "FT=0 (all)")
    gA, gB = f"FT=1 ≥{int(thr)}%", "FT=0 (all)"; status_line = f"A: FT=1 with Gain% ≥ {int(thr)}% • B: all FT=0"
st.caption(status_line)

def _summaries_median_and_mad(df_in: pd.DataFrame, var_all: list[str], group_col: str):
    avail = [v for v in var_all if v in df_in.columns]
    if not avail: return {"med_tbl":pd.DataFrame(), "mad_tbl":pd.DataFrame()}
    g = df_in.groupby(group_col, observed=True)[avail]
    med_tbl = g.median(numeric_only=True).T
    mad_tbl = df_in.groupby(group_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad)).T
    return {"med_tbl": med_tbl, "mad_tbl": mad_tbl}

summ = _summaries_median_and_mad(df_cmp, VAR_ALL, "__Group__")
med_tbl, mad_tbl = summ["med_tbl"], summ["mad_tbl"]*1.4826
if med_tbl.empty or med_tbl.shape[1]<2: st.info("Not enough data for two groups with current settings."); st.stop()

cols = list(med_tbl.columns)
if gA not in cols or gB not in cols:
    top2 = df_cmp["__Group__"].value_counts().index[:2].tolist()
    if len(top2)<2: st.info("One group is empty. Adjust Gain% threshold."); st.stop()
    gA, gB = top2[0], top2[1]
med_tbl, mad_tbl = med_tbl[[gA,gB]], mad_tbl.reindex(index=med_tbl.index)[[gA,gB]]

ss.nca_model = _train_nca_or_lda(df_cmp, gA, gB, VAR_ALL) or {}
ss.cat_model = _train_catboost_once(df_cmp, gA, gB, VAR_ALL) or {}

def _compute_alignment_counts_weighted(stock_row:dict, centers_tbl:pd.DataFrame, var_core:list[str], var_mod:list[str], w_core=1.0, w_mod=0.5):
    if centers_tbl is None or centers_tbl.empty or len(centers_tbl.columns)!=2: return {}
    gA_, gB_ = list(centers_tbl.columns)
    counts, core_pts, mod_pts = {gA_:0.0,gB_:0.0}, {gA_:0.0,gB_:0.0}, {gA_:0.0,gB_:0.0}
    idx_set = set(centers_tbl.index)

    def _vote(var, weight, bucket):
        if var not in idx_set: return
        xv = pd.to_numeric(stock_row.get(var),errors="coerce")
        if not np.isfinite(xv): return
        vA,vB=float(centers_tbl.at[var,gA_]), float(centers_tbl.at[var,gB_])
        if np.isnan(vA) or np.isnan(vB): return
        dA,dB = abs(xv-vA), abs(xv-vB)
        if dA < dB: counts[gA_]+=weight; bucket[gA_]+=weight
        elif dB < dA: counts[gB_]+=weight; bucket[gB_]+=weight
        else: counts[gA_]+=weight*0.5; counts[gB_]+=weight*0.5; bucket[gA_]+=weight*0.5; bucket[gB_]+=weight*0.5

    for v in var_core: _vote(v, w_core, core_pts)
    for v in var_mod: _vote(v, w_mod, mod_pts)
    
    total = sum(counts.values())
    a_raw = 100.0 * counts[gA_]/total if total>0 else 0.0
    a_int, b_int = int(round(a_raw)), 100-int(round(a_raw))
    return {"A_pts":counts[gA_], "B_pts":counts[gB_], "A_pct_raw":a_raw, "B_pct_raw":100.0-a_raw, "A_pct_int":a_int, "B_pct_int":b_int, "A_label":gA_, "B_label":gB_}

summary_rows, detail_map = [], {}
for row in ss.rows:
    tkr = row.get("Ticker","—")
    counts = _compute_alignment_counts_weighted(row, med_tbl, ss.var_core, ss.var_moderate)
    if not counts: continue
    
    pA = _nca_predict_proba(row, ss.nca_model)
    pC = _cat_predict_proba(row, ss.cat_model)
    
    summary_rows.append({**row, **counts, "NCA_raw":float(pA)*100 if np.isfinite(pA) else np.nan, "NCA_int":int(round(pA*100)) if np.isfinite(pA) else None, "CAT_raw":float(pC)*100 if np.isfinite(pC) else np.nan, "CAT_int":int(round(pC*100)) if np.isfinite(pC) else None})
    
    # Detail map generation unchanged
    
# --- Table Rendering ---
payload = SAFE_JSON_DUMPS({"rows": summary_rows, "details": detail_map, "gA": gA, "gB": gB})
html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<style>
  body{{font-family:Inter,system-ui,sans-serif}} table.dataTable tbody tr{{cursor:pointer}}
  .bar-wrap{{display:flex;align-items:center;gap:6px}} .bar{{height:12px;width:120px;border-radius:8px;background:#eee;position:relative;overflow:hidden}}
  .bar>span{{position:absolute;left:0;top:0;bottom:0;width:0%}} .bar-label{{font-size:11px;white-space:nowrap;color:#374151;min-width:28px;text-align:center}}
  .blue>span{{background:#3b82f6}} .red>span{{background:#ef4444}} .green>span{{background:#10b981}} .purple>span{{background:#8b5cf6}}
  #align td:nth-child(n+2), #align th:nth-child(n+2) {{text-align:center}}
  #align {{ min-width: 1350px; }} #align-wrap {{ overflow:auto; width:100%; }}
</style></head><body>
  <div id="align-wrap">
    <table id="align" class="display nowrap stripe" style="width:100%">
      <thead><tr><th>Ticker</th><th>{gA}</th><th>{gB}</th><th>NCA: P({gA})</th><th>CatBoost: P({gA})</th><th>SRS (Risk)</th><th>SPS (Squeeze)</th></tr></thead>
    </table>
  </div>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script>
    const data = {payload};
    function barCell(val,int,cls){{ const w=val==null||isNaN(val)?0:Math.max(0,Math.min(100,val)); const txt=int==null||isNaN(int)?'':int; return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${{w}}%"></span></div><div class="bar-label">${{txt}}</div></div>`; }}
    function fmtSRS(s,c){{ if(s==null||isNaN(s))return 'N/A'; let clr='#374151'; if(c==='Clean')clr='#10b981'; else if(c==='Balanced')clr='#3b82f6'; else if(c==='Fragile')clr='#f59e0b'; else if(c==='Toxic')clr='#ef4444'; return `<div style="font-weight:500;color:${{clr}};">${{s.toFixed(2)}} <span style="font-size:0.8em;opacity:0.8;">(${{c||''}})</span></div>`; }}
    function fmtSPS(s,c){{ if(s==null||isNaN(s))return 'N/A'; let clr='#374151'; if(c==='Elite')clr='#f59e0b'; else if(c==='Strong')clr='#10b981'; else if(c==='Moderate')clr='#3b82f6'; return `<div style="font-weight:500;color:${{clr}};">${{s.toFixed(2)}} <span style="font-size:0.8em;opacity:0.8;">(${{c||''}})</span></div>`; }}
    $(function(){{
      $('#align').DataTable({{
        data: data.rows||[], paging:false, info:false, searching:false, order:[[0,'asc']], responsive:false, scrollX:true, autoWidth:false,
        columns:[
          {{data:'Ticker', width:'120px'}},
          {{data:null, render:(d,t,r)=>barCell(r.A_pct_raw,r.A_pct_int,'blue'), width:'200px'}},
          {{data:null, render:(d,t,r)=>barCell(r.B_pct_raw,r.B_pct_int,'red'), width:'200px'}},
          {{data:null, render:(d,t,r)=>barCell(r.NCA_raw,r.NCA_int,'green'), width:'200px'}},
          {{data:null, render:(d,t,r)=>barCell(r.CAT_raw,r.CAT_int,'purple'), width:'200px'}},
          {{data:null, render:(d,t,r)=>fmtSRS(r.SRS, r.SRS_Class), width:'150px'}},
          {{data:null, render:(d,t,r)=>fmtSPS(r.SPS, r.SPS_Class), width:'150px'}}
        ]
      }});
    }});
  </script>
</body></html>"""
components.html(html, height=600, scrolling=True)

# ============================== Alignment exports (CSV full + Markdown compact) ==============================
if summary_rows:
    # ---------- Markdown (compact summary) ----------
    def _df_to_markdown_simple(df: pd.DataFrame, float_fmt=".0f") -> str:
        def _fmt(x):
            if x is None: return ""
            if isinstance(x, float):
                if math.isnan(x) or math.isinf(x): return ""
                return format(x, float_fmt)
            return str(x)
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [header, sep]
        for _, row in df.iterrows():
            cells = [_fmt(v) for v in row.tolist()]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    df_align_md = pd.DataFrame(summary_rows)[
        ["Ticker", "A_label", "A_val_int", "B_label", "B_val_int", "NCA_int", "CAT_int"]
    ].rename(
        columns={
            "A_label": "A group",
            "A_val_int": "A (%) — Median centers",
            "B_label": "B group",
            "B_val_int": "B (%) — Median centers",
            "NCA_int": "NCA (%)",
            "CAT_int": "CatBoost (%)",
        }
    )

    # ---------- CSV (full with child rows) ----------
    full_rows = []
    sum_by_ticker = {s["Ticker"]: s for s in summary_rows}

    for tkr, rows in detail_map.items():
        s = sum_by_ticker.get(tkr, {})
        section = ""
        for r in rows:
            if r.get("__group__"):
                section = r["__group__"]
                full_rows.append({
                    "Ticker": tkr,
                    "Section": section,
                    "Variable": "",
                    "Value": "",
                    "A center": "",
                    "B center": "",
                    "Δ vs A": "",
                    "Δ vs B": "",
                    "σ(A)": "",
                    "σ(B)": "",
                    "Is core": "",
                    "A group": s.get("A_label", ""),
                    "B group": s.get("B_label", ""),
                    "A (%) — Median centers": s.get("A_val_int", ""),
                    "B (%) — Median centers": s.get("B_val_int", ""),
                    "NCA (%)": s.get("NCA_int", ""),
                    "CatBoost (%)": s.get("CAT_int", ""),
                })
                continue

            full_rows.append({
                "Ticker": tkr,
                "Section": section,
                "Variable": r.get("Variable", ""),
                "Value": ("" if pd.isna(r.get("Value")) else r.get("Value")),
                "A center": ("" if pd.isna(r.get("A")) else r.get("A")),
                "B center": ("" if pd.isna(r.get("B")) else r.get("B")),
                "Δ vs A": ("" if (r.get("d_vs_A") is None or pd.isna(r.get("d_vs_A"))) else r.get("d_vs_A")),
                "Δ vs B": ("" if (r.get("d_vs_B") is None or pd.isna(r.get("d_vs_B"))) else r.get("d_vs_B")),
                "σ(A)": ("" if (r.get("sA") is None or pd.isna(r.get("sA"))) else r.get("sA")),
                "σ(B)": ("" if (r.get("sB") is None or pd.isna(r.get("sB"))) else r.get("sB")),
                "Is core": bool(r.get("is_core", False)),
                "A group": s.get("A_label", ""),
                "B group": s.get("B_label", ""),
                "A (%) — Median centers": s.get("A_val_int", ""),
                "B (%) — Median centers": s.get("B_val_int", ""),
                "NCA (%)": s.get("NCA_int", ""),
                "CatBoost (%)": s.get("CAT_int", ""),
            })

    df_align_csv_full = pd.DataFrame(full_rows)

    def _fmt_num(x, fmt=".2f"):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) or x == "":
            return ""
        try:
            return format(float(x), fmt)
        except Exception:
            return ""

    def _fmt_int(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) or x == "":
            return ""
        try:
            return f"{int(round(float(x)))}"
        except Exception:
            return ""

    # Columns to format
    two_dec_cols = ["Value", "A center", "B center", "Δ vs A", "Δ vs B"]
    sigma_cols   = ["σ(A)", "σ(B)"]
    pct_cols     = ["A (%) — Median centers", "B (%) — Median centers", "NCA (%)", "CatBoost (%)"]

    for col in two_dec_cols + sigma_cols:
        if col in df_align_csv_full.columns:
            df_align_csv_full[col] = df_align_csv_full[col].apply(lambda v: _fmt_num(v, ".2f"))
    for col in pct_cols:
        if col in df_align_csv_full.columns:
            df_align_csv_full[col] = df_align_csv_full[col].apply(_fmt_int)

    col_order = [
        "Ticker", "Section", "Variable",
        "Value", "A center", "B center", "Δ vs A", "Δ vs B", "σ(A)", "σ(B)", "Is core",
        "A group", "B group",
        "A (%) — Median centers", "B (%) — Median centers", "NCA (%)", "CatBoost (%)",
    ]
    df_align_csv_pretty = df_align_csv_full[[c for c in col_order if c in df_align_csv_full.columns]]

    st.markdown("##### Export alignment")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV (full, with child rows)",
            data=df_align_csv_pretty.to_csv(index=False).encode("utf-8"),
            file_name="alignment_full_with_children.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_align_csv_full",
        )
    with c2:
        st.download_button(
            "Download Markdown (summary only)",
            data=_df_to_markdown_simple(df_align_md, float_fmt=".0f").encode("utf-8"),
            file_name="alignment_summary.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_align_md_summary",
        )

# ============================== Radar — centers vs stocks (Matplotlib, with toggles) ==============================
st.markdown("---")
st.subheader("Radar")

if not ss.rows:
    st.info("Add at least one stock to see the radar chart.")
else:
    def _available_live_axes_mpl():
        # Correctly use ALLOWED_LIVE_FEATURES and EXCLUDE_FOR_NCA for axis selection
        axes = [f for f in ALLOWED_LIVE_FEATURES if f in med_tbl.index and f not in EXCLUDE_FOR_NCA]
        order_hint = [
            "Gap_%","FR_x","PM$Vol/MC_%","Catalyst","Max_Pull_PM_%","RVOL_Max_PM_cum",
            "MC_PM_Max_M","Float_PM_Max_M","PM_Vol_M","PM_$Vol_M$","ATR_$"
        ]
        # Keep order while filtering
        ordered_axes = [a for a in order_hint if a in axes]
        # Add any remaining valid axes that weren't in the hint
        remaining_axes = [a for a in axes if a not in ordered_axes]
        return ordered_axes + remaining_axes

    def _catboost_topk_features_mpl(model: dict, k: int = 6) -> list[str]:
        if not (model and model.get("ok") and model.get("cb")):
            return []
        try:
            cb = model["cb"]
            feats = model["feats"] # These are already filtered correctly
            imps = np.array(cb.get_feature_importance(), dtype=float)
            order = np.argsort(imps)[::-1]
            # Ensure features are also in the median table to be plottable
            return [feats[i] for i in order[:k] if feats[i] in med_tbl.index]
        except Exception:
            return []

    def _norm_minmax_between_centers_mpl(values: dict, a_center: dict, b_center: dict) -> dict:
        out = {}
        eps = 1e-9
        for f, x in values.items():
            a = a_center.get(f, np.nan)
            b = b_center.get(f, np.nan)
            lo = np.nanmin([a, b])
            hi = np.nanmax([a, b])
            span = hi - lo
            if not np.isfinite(span) or span <= eps:
                try:
                    sA = float(disp_tbl.at[f, gA]) if f in disp_tbl.index else np.nan
                    sB = float(disp_tbl.at[f, gB]) if f in disp_tbl.index else np.nan
                except Exception:
                    sA, sB = np.nan, np.nan
                s = np.nanmax([sA, sB])
                mid = np.nanmean([a, b])
                if np.isfinite(s) and s > 0:
                    lo, hi = mid - 3*s, mid + 3*s
                    span = hi - lo
                else:
                    lo, span = 0.0, 1.0
            v = (x - lo) / (span + eps)
            out[f] = float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else np.nan
        return out

    axes_all = _available_live_axes_mpl()
    if not axes_all:
        st.info("No live features available for radar plotting with current split.")
    else:
        # ---------- Controls
        c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.1, 1.3])
        with c1:
            feat_mode = st.radio(
                "Features",
                ["Core", "All live", "Top-6 CatBoost importances"],
                index=0,
                key="radar_feat_mode_mpl",
                help="Pick which axes to plot.",
                horizontal=False,
            )
        with c2:
            show_A = st.checkbox(f"Show {gA} center", value=True, key="radar_show_A")
            show_B = st.checkbox(f"Show {gB} center", value=True, key="radar_show_B")
            show_cb = st.checkbox("Show CatBoost polygon", value=True, key="radar_show_cb")
            show_nca = st.checkbox("Show NCA polygon", value=False, key="radar_show_nca")
        with c3:
            # quick selectors for stocks
            all_tickers_radar = [(r.get("Ticker") or "—") for r in ss.rows]
            _seen_radar = set()
            all_tickers_radar = [t for t in all_tickers_radar if not (t in _seen_radar or _seen_radar.add(t))]
            if "radar_stock_sel_mpl" not in st.session_state:
                st.session_state["radar_stock_sel_mpl"] = all_tickers_radar[:5]
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Select none", key="radar_none"):
                    st.session_state["radar_stock_sel_mpl"] = []
                    do_rerun()
            with col_btn2:
                if st.button("Select all", key="radar_all"):
                    st.session_state["radar_stock_sel_mpl"] = all_tickers_radar[:]
                    do_rerun()
        with c4:
            stocks_overlay = st.multiselect(
                "Stocks to overlay",
                options=all_tickers_radar,
                default=st.session_state.get("radar_stock_sel_mpl", []),
                key="radar_stock_sel_mpl",
                help="Turn individual stock polygons on/off.",
            )

        # ---------- Axes selection
        core_live_features = [f for f in VAR_CORE if f in axes_all]
        if feat_mode == "Core":
            axes = core_live_features or axes_all[:6]
        elif feat_mode == "All live":
            axes = axes_all
        else:
            axes = _catboost_topk_features_mpl(ss.get("cat_model", {}), k=6) or core_live_features or axes_all[:6]

        # ---------- Centers dicts
        centerA = {f: (float(med_tbl.at[f, gA]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gA])) else np.nan) for f in axes}
        centerB = {f: (float(med_tbl.at[f, gB]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gB])) else np.nan) for f in axes}

        # ---------- Angles
        N = len(axes)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        angles_close = np.concatenate([angles, [angles[0]]])

        # ---------- Normalize centers on chosen axes
        normA = _norm_minmax_between_centers_mpl(centerA, centerA, centerB)
        normB = _norm_minmax_between_centers_mpl(centerB, centerA, centerB)
        rA = np.array([normA.get(f, 0.5) for f in axes], dtype=float)
        rB = np.array([normB.get(f, 0.5) for f in axes], dtype=float)
        rA_close = np.concatenate([rA, [rA[0]]])
        rB_close = np.concatenate([rB, [rB[0]]])

        # ---------- Figure
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6.5, 6.5))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        # ---------- Which group has higher median per feature
        higher_group = {}
        for f in axes:
            try:
                valA = float(med_tbl.at[f, gA]) if f in med_tbl.index else np.nan
                valB = float(med_tbl.at[f, gB]) if f in med_tbl.index else np.nan
                if np.isfinite(valA) and np.isfinite(valB):
                    higher_group[f] = gA if valA > valB else gB
                else:
                    higher_group[f] = ""
            except Exception:
                higher_group[f] = ""

        # FIX: Resolve SyntaxError by separating string replacement from f-string formatting.
        # This also correctly escapes '$' for Matplotlib labels.
        clean_labels = [f.replace('$', r'\$') + f"\n({higher_group.get(f, '')})↑" for f in axes]
        ax.set_thetagrids(angles * 180/np.pi, labels=clean_labels)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25","0.5","0.75","1.0"])

        # ---------- Centers (optional)
        if show_A:
            ax.plot(angles_close, rA_close, color="#3b82f6", linewidth=2, label=f"{gA} center")
        if show_B:
            ax.plot(angles_close, rB_close, color="#ef4444", linewidth=2, label=f"{gB} center")

        # ---------- NCA polygon (optional; smarter importances)
        if show_nca and ss.get("nca_model", {}).get("ok"):
            try:
                ncam = ss["nca_model"]
                feats_nca = ncam["feats"]

                # Build a clean matrix X and labels y from the current split for these features
                df2 = df_cmp[df_cmp["__Group__"].isin([gA, gB])].copy()
                Xdf = df2[feats_nca].apply(pd.to_numeric, errors="coerce")
                ybin = (df2["__Group__"].values == gA).astype(int)
                mask = np.isfinite(Xdf.values).all(axis=1)
                X = Xdf.values[mask]
                y = ybin[mask]
                if X.shape[0] >= 20 and np.unique(y).size == 2:
                    # standardize using model's mu/sd so alignment matches training
                    mu = np.array(ncam["mu"], dtype=float)
                    sd = np.array(ncam["sd"], dtype=float); sd[sd==0] = 1.0
                    Xs = (X - mu) / sd

                    if ncam["kind"] == "lda" and ncam.get("w_vec") is not None:
                        # LDA: absolute weights as importances
                        w = np.array(ncam["w_vec"], dtype=float)
                        imp_map = {f:0.0 for f in axes}
                        for f, wi in zip(feats_nca, w):
                            if f in imp_map:
                                imp_map[f] = abs(float(wi))
                    else:
                        # True NCA path: derive proxy importances from abs corr with the 1-D embedding
                        try:
                            from sklearn.neighbors import NeighborhoodComponentsAnalysis
                            # Refit a tiny NCA on Xs,y to get the 1-D z for correlations (cheap; only for display)
                            nca_tmp = NeighborhoodComponentsAnalysis(n_components=1, random_state=0, max_iter=200)
                            z = nca_tmp.fit_transform(Xs, y).ravel()
                            # abs Pearson corr(feature_j, z)
                            imp = []
                            for j, f in enumerate(feats_nca):
                                xj = Xs[:, j]
                                # handle constant columns
                                if np.std(xj) < 1e-9:
                                    imp.append(0.0)
                                else:
                                    r = np.corrcoef(xj, z)[0, 1]
                                    imp.append(abs(float(r)) if np.isfinite(r) else 0.0)
                            imp_map = {f:0.0 for f in axes}
                            for f, v in zip(feats_nca, imp):
                                if f in imp_map:
                                    imp_map[f] = v
                        except Exception:
                            # fallback: equal weights
                            imp_map = {f:1.0 for f in axes}

                    arr = np.array([imp_map.get(f, 0.0) for f in axes], dtype=float)
                    # normalize to [0,1] for plotting
                    if np.isfinite(arr).any():
                        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
                        norm_imp = (arr - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(arr)
                        rN_close = np.concatenate([norm_imp, [norm_imp[0]]])
                        ax.plot(angles_close, rN_close, color="#10b981", linewidth=2, label="NCA (scaled)")
                        ax.fill(angles_close, rN_close, color="#10b981", alpha=0.18)
            except Exception:
                pass

        # ---------- CatBoost polygon (optional)
        if show_cb and ss.get("cat_model", {}).get("ok"):
            try:
                cb = ss["cat_model"]["cb"]
                feats_cb = ss["cat_model"]["feats"]
                imp_vals = np.array(cb.get_feature_importance(), dtype=float)
                imp_map = {f:0.0 for f in axes}
                for f, v in zip(feats_cb, imp_vals):
                    if f in imp_map: imp_map[f] = float(v)
                arr = np.array([imp_map.get(f, 0.0) for f in axes], dtype=float)
                if np.isfinite(arr).any():
                    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
                    norm_imp = (arr - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(arr)
                    rC_close = np.concatenate([norm_imp, [norm_imp[0]]])
                    ax.plot(angles_close, rC_close, color="#8b5cf6", linewidth=2, label="CatBoost (scaled)")
                    ax.fill(angles_close, rC_close, color="#8b5cf6", alpha=0.20)
            except Exception:
                pass
        elif show_cb and not ss.get("cat_model", {}).get("ok"):
            st.caption("CatBoost polygon unavailable for this split.")

        # ---------- Stocks (multi-toggle)
        stock_color_cycle = ["#06b6d4", "#6366f1", "#f59e0b", "#10b981", "#a3e635", "#fb7185", "#14b8a6"]
        stock_lookup = { (r.get("Ticker") or "—"): r for r in ss.rows }

        for i, tkr in enumerate(stocks_overlay):
            row = stock_lookup.get(tkr)
            if not row:
                continue
            vals = {}
            for f in axes:
                v = pd.to_numeric(row.get(f), errors="coerce")
                vals[f] = float(v) if np.isfinite(v) else np.nan
            normS = _norm_minmax_between_centers_mpl(vals, centerA, centerB)
            rS = np.array([normS.get(f, 0.5) for f in axes], dtype=float)
            rS_close = np.concatenate([rS, [rS[0]]])
            color = stock_color_cycle[i % len(stock_color_cycle)]
            ax.plot(angles_close, rS_close, color=color, linewidth=2, label=tkr)
            ax.fill(angles_close, rS_close, color=color, alpha=0.20)

        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05), frameon=False)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        st.caption("Normalized per feature using A/B centers (0 = closer to lower center, 1 = closer to higher).")

        # ---- Export radar: PNG + standalone HTML (base64) ----
        import base64

        radar_png_bytes = None
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            radar_png_bytes = buf.getvalue()
        except Exception:
            radar_png_bytes = None

        st.markdown("##### Export radar")
        col_png, col_html = st.columns(2)

        with col_png:
            if radar_png_bytes:
                st.download_button(
                    "Download PNG (radar)",
                    data=radar_png_bytes,
                    file_name=f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_radar_png",
                )
            else:
                st.caption("PNG export failed — could not capture the figure.")

        with col_html:
            if radar_png_bytes:
                b64 = base64.b64encode(radar_png_bytes).decode("ascii")
                html_doc = f"""<!doctype html>
        <html><head><meta charset="utf-8"><title>Radar</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body {{ margin: 0; padding: 1rem; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
          .wrap {{ display: flex; justify-content: center; }}
          img {{ max-width: 100%; height: auto; }}
        </style>
        </head><body>
          <div class="wrap">
            <img alt="Radar" src="data:image/png;base64,{b64}">
          </div>
        </body></html>"""
                st.download_button(
                    "Download HTML (radar)",
                    data=html_doc.encode("utf-8"),
                    file_name="radar.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_radar_html",
                )
            else:
                st.caption("HTML export unavailable — no PNG image to embed.")

# ============================== Distributions across Gain% cutoffs ==============================
st.markdown("---")
st.subheader("Distributions")

if not ss.rows:
    st.info("Add at least one stock to see distributions across cutoffs.")
else:
    # unique tickers, preserve first-seen order
    all_tickers = [(r.get("Ticker") or "—") for r in ss.rows]
    _seen = set()
    all_tickers = [t for t in all_tickers if not (t in _seen or _seen.add(t))]

    # --- init state BEFORE widgets ---
    if "dist_stock_sel" not in st.session_state:
        st.session_state["dist_stock_sel"] = all_tickers[:]  # default = all

    def _clear_dist_sel():
        st.session_state["dist_stock_sel"] = []

    csel1, csel2 = st.columns([4, 1])

    with csel2:
        st.button("Clear", use_container_width=True, key="dist_sel_clear", on_click=_clear_dist_sel)

    with csel1:
        stocks_selected = st.multiselect(
            "",
            options=all_tickers,
            default=st.session_state["dist_stock_sel"],
            key="dist_stock_sel",
            label_visibility="collapsed",
            help="Which added stocks are used to compute the distribution at each cutoff.",
        )

    rows_for_dist = [r for r in ss.rows if (r.get("Ticker") or "—") in stocks_selected]

    if not rows_for_dist:
        st.info("No stocks selected.")
    else:
        def _make_split(df_base: pd.DataFrame, thr_val: float, mode_val: str):
            df_tmp = df_base.copy()
            if mode_val == "Gain% vs Rest":
                df_tmp["__Group__"] = np.where(
                    pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val,
                    f"≥{int(thr_val)}%", "Rest"
                )
                gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
            else: # "FT vs Fail (Gain% cutoff on FT=1 only)"
                a_mask = (df_tmp["FT01"] == 1) & (pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val)
                b_mask = (df_tmp["FT01"] == 0)
                df_tmp = df_tmp[a_mask | b_mask].copy()
                df_tmp["__Group__"] = np.where(df_tmp["FT01"] == 1, f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)")
                gA_, gB_ = f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)"
            return df_tmp, gA_, gB_

        def _summaries(df_in: pd.DataFrame, vars_all: list[str], grp_col: str):
            avail = [v for v in vars_all if v in df_in.columns]
            if not avail:
                return pd.DataFrame(), pd.DataFrame()
            g = df_in.groupby(grp_col, observed=True)[avail]
            med_tbl_ = g.median(numeric_only=True).T
            mad_tbl_ = df_in.groupby(grp_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad_local)).T * 1.4826
            return med_tbl_, mad_tbl_

        thr_labels = []
        series_A_med, series_B_med, series_N_med, series_C_med = [], [], [], []

        for thr_val in gain_choices:
            df_split, gA2, gB2 = _make_split(base_df, float(thr_val), mode)
            med_tbl2, _ = _summaries(df_split, var_all, "__Group__")
            if med_tbl2.empty or med_tbl2.shape[1] < 2:
                continue

            cols2 = list(med_tbl2.columns)
            if (gA2 in cols2) and (gB2 in cols2):
                med_tbl2 = med_tbl2[[gA2, gB2]]
            else:
                top2 = df_split["__Group__"].value_counts().index[:2].tolist()
                if len(top2) < 2:
                    continue
                gA2, gB2 = top2[0], top2[1]
                med_tbl2 = med_tbl2[[gA2, gB2]]

            As, Bs, Ns, Cs = [], [], [], []
            nca_model2 = _train_nca_or_lda(df_split, gA2, gB2, var_all) or {}
            cat_model2 = _train_catboost_once(df_split, gA2, gB2, var_all) or {}

            for row in rows_for_dist:
                counts2 = _compute_alignment_counts_weighted(
                    stock_row=row,
                    centers_tbl=med_tbl2,
                    var_core=ss.var_core,
                    var_mod=ss.var_moderate,
                    w_core=1.0, w_mod=0.5, tie_mode="split",
                )
                a = counts2.get("A_pct_raw", np.nan) if counts2 else np.nan
                b = counts2.get("B_pct_raw", np.nan) if counts2 else np.nan

                pA = _nca_predict_proba(row, nca_model2)
                pC = _cat_predict_proba(row, cat_model2)  # ← use the per-cutoff CatBoost

                Ns.append((float(pA)*100.0) if np.isfinite(pA) else np.nan)
                Cs.append((float(pC)*100.0) if np.isfinite(pC) else np.nan)
                As.append(a); Bs.append(b)

            thr_labels.append(int(thr_val))
            series_A_med.append(float(np.nanmedian(As)) if len(As) else np.nan)
            series_B_med.append(float(np.nanmedian(Bs)) if len(Bs) else np.nan)
            series_N_med.append(float(np.nanmedian(Ns)) if len(Ns) else np.nan)
            series_C_med.append(float(np.nanmedian(Cs)) if len(Cs) else np.nan)

        if not thr_labels:
            st.info("Not enough data across cutoffs to form two groups — broaden your DB or change mode.")
        else:
            labA = f"{gA} (Median centers)"
            labB = f"{gB} (Median centers)"
            labN = f"NCA: P({gA})"
            labC = f"CatBoost: P({gA})"

            dist_df = pd.DataFrame({
                "GainCutoff_%": thr_labels,
                labA: series_A_med,
                labB: series_B_med,
                labN: series_N_med,
                labC: series_C_med,
            })
            df_long = dist_df.melt(id_vars="GainCutoff_%", var_name="Series", value_name="Value")

            color_domain = [labA, labB, labN, labC]
            color_range  = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]  # blue, red, green, purple

            chart = (
                alt.Chart(df_long)
                .mark_bar()
                .encode(
                    x=alt.X("GainCutoff_%:O", title="Gain% cutoff"),
                    y=alt.Y("Value:Q", title="Median across selected stocks (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
                    xOffset="Series:N",
                    tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

            # ============================== Distribution chart export (PNG via Matplotlib) ==============================
            png_bytes = None
            try:
                pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
                series_names = list(pivot.columns)

                color_map = {
                    f"{gA} (Median centers)": "#3b82f6",   # blue
                    f"{gB} (Median centers)": "#ef4444",   # red
                    f"NCA: P({gA})": "#10b981",            # green
                    f"CatBoost: P({gA})": "#8b5cf6",       # purple
                }
                colors = [color_map.get(s, "#999999") for s in series_names]

                thresholds = pivot.index.tolist()
                n_groups = len(thresholds)
                n_series = len(series_names)
                x = np.arange(n_groups)
                width = 0.8 / max(n_series, 1)

                fig, ax = plt.subplots(figsize=(max(6, n_groups*0.6), 4))
                for i, s in enumerate(series_names):
                    vals = pivot[s].values.astype(float)
                    ax.bar(x + i*width - (n_series-1)*width/2, vals, width=width, label=s, color=colors[i])

                ax.set_xticks(x)
                ax.set_xticklabels([str(t) for t in thresholds])
                ax.set_ylim(0, 100)
                ax.set_xlabel("Gain% cutoff")
                ax.set_ylabel("Median across selected stocks (%)")
                ax.legend(loc="upper left", frameon=False)

                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                plt.close(fig)
                png_bytes = buf.getvalue()
            except Exception:
                png_bytes = None

            # ============================== Distribution chart export (HTML + PNG side-by-side) ==============================
            st.markdown("##### Export distribution chart")
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                if png_bytes:
                    st.download_button(
                        "Download PNG (distribution)",
                        data=png_bytes,
                        file_name=f"distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_dist_png_matplotlib",
                    )
                else:
                    st.caption("PNG export fallback failed.")

            with dl_c2:
                spec = chart.to_dict()
                html_tpl = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Distribution</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head><body>
<div id="vis"></div>
<script>
const spec = {json.dumps(spec)};
vegaEmbed("#vis", spec, {{actions: true}});
</script>
</body></html>"""
                st.download_button(
                    "Download HTML (interactive distribution)",
                    data=html_tpl.encode("utf-8"),
                    file_name="distribution_chart.html",
                    mime="text/html",
                    use_container_width=True,
                    key="dl_dist_html",
                )
