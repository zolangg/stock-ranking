# app.py — Premarket Ranking: Blended Day Volume + FT (learned from uploaded workbook)
# ------------------------------------------------------------------------------------
# What this app does:
#   • Learns two day-volume predictors from your DB:
#       (A) Direct log-linear on ln(Daily Vol M).
#       (B) PM%-of-day model → invert using PMVol.
#     Then blends them in LOG-SPACE using weights from relative R² on your sheet.
#     CI68 is blended in ln-space. PredVol is NEVER below PMVol.
#   • FT classifier uses dataset variables + ln(PredDay).
#   • Catalyst is used end-to-end (dataset + live input).
#   • Robust column mapping per your Legend; numeric inputs accept commas.
#
# Run: streamlit run app.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math, re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking — Blended DayVol", layout="wide")
st.title("Premarket Stock Ranking — Blended Day Volume")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
      .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
      [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
      [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== Session State ==============================
if "ARTIFACTS" not in st.session_state: st.session_state.ARTIFACTS = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}

# ============================== Helpers ==============================
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip().replace(" ", "").replace("’","").replace("'","")
    if s == "": return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

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

# Column finder aligned to Legend (case/spacing/units robust)
_DEF = {
    "FT": ["FT"],
    "MAXPCT": ["Max Push Daily %", "Max Push Daily (%)", "Max Push %"],  # optional
    "GAP": ["Gap %", "Gap"],
    "ATR": ["Daily ATR", "ATR $", "ATR", "ATR (USD)", "ATR$"],
    "RVOL": ["RVOL @ BO", "RVOL", "Relative Volume"],
    "PMVOL": ["PM Vol (M)", "Premarket Vol (M)", "PM Volume (M)"],
    "PM$": ["PM $Vol (M)", "PM Dollar Vol (M)", "PM $ Volume (M)"],      # optional
    "FLOAT": ["Float M Shares", "Public Float (M)", "Float (M)", "Float"],
    "MCAP": ["MarketCap M", "Market Cap (M)", "MCap M"],
    "SI": ["Short Interest %", "Short Float %", "Short Interest (Float) %"],
    "DAILY": ["Daily Vol (M)", "Day Volume (M)", "Volume (M)"],          # for direct model target
    "CAT": ["Catalyst", "News", "PR"],
}

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
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

# math & model utils
def _nz(x, fallback=0.0):
    """Return numeric x or fallback if x is not finite."""
    try:
        xx = float(x)
        return xx if np.isfinite(xx) else float(fallback)
    except Exception:
        return float(fallback)

def _safe_log(x: float, eps: float = 1e-8) -> float:
    return math.log(max(_nz(x, 0.0), eps))

def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    n, k = X.shape
    I = np.eye(k)
    return np.linalg.solve(X.T @ X + l2 * I, X.T @ y)

def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 80, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    w = np.zeros(k+1)
    R = np.eye(k+1); R[0,0] = 0.0; R *= l2
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0/(1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p*(1-p)
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H = Xb.T @ WX + R
        g = Xb.T @ (y - p)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w + delta
        if np.linalg.norm(delta) < tol: w = w_new; break
        w = w_new
    return w[1:].astype(float), float(w[0])

def logit_inv(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0/(1.0 + np.exp(-z))

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn from uploaded sheet", use_container_width=True)

def _load_and_learn(xls: pd.ExcelFile, sheet: str) -> None:
    raw = pd.read_excel(xls, sheet)

    # --- column mapping (Legend) ---
    col = {}
    for key, cands in _DEF.items():
        pick = _pick(raw, cands)
        if pick: col[key] = pick

    if "FT" not in col:
        st.error("No 'FT' column found in sheet.")
        return

    df = pd.DataFrame()
    # Force-binary FT: >=0.5 → 1.0 else 0.0
    ft_series = pd.to_numeric(raw[col["FT"]], errors="coerce")
    df["FT"] = (ft_series.fillna(0.0) >= 0.5).astype(float)

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
    _add("daily_vol_m", "DAILY")
    if "CAT" in col:
        df["catalyst"] = pd.to_numeric(raw[col["CAT"]], errors="coerce").clip(0,1).fillna(0.0)

    # derived
    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"].replace(0, np.nan)
    if {"pm_vol_m","float_m"}.issubset(df.columns):
        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

    # ---------- (A1) PM%-of-Day model (logit) ----------
    pmcoef = None; pmbias = 0.0; pmsigma = 0.60; pm_base = 0.139; pm_feats: List[str] = []
    if "pm_pct_daily" in df.columns and df["pm_pct_daily"].notna().sum() >= 8:
        y_raw = pd.to_numeric(df["pm_pct_daily"], errors="coerce").astype(float)
        y_frac = np.clip(y_raw/100.0, 0.005, 0.95)
        y = np.log(y_frac/(1.0 - y_frac))
        candidates = [
            ("ln_mcap",   lambda r: _safe_log(r.get("mcap_m")),                       "mcap_m"),
            ("ln_gapf",   lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0),       "gap_pct"),
            ("ln_atr",    lambda r: _safe_log(r.get("atr_usd")),                      "atr_usd"),
            ("ln_float",  lambda r: _safe_log(r.get("float_m")),                      "float_m"),
            ("ln1p_rvol", lambda r: _safe_log(1.0 + _nz(r.get("rvol"),0.0)),         "rvol"),
            ("catalyst",  lambda r: float(_nz(r.get("catalyst"),0.0)),               "catalyst"),
        ]
        use = [c for c in candidates if c[2] in df.columns]
        if use:
            X = np.vstack([[f(df.iloc[i].to_dict()) for (name,f,_) in use] for i in range(len(df))]).astype(float)
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            Xf, yf = X[mask], y[mask]
            if Xf.shape[0] >= 8:
                pmcoef = ridge_fit(Xf, yf, l2=1.2)
                resid = yf - (Xf @ pmcoef)
                pmbias = float(np.mean(resid))
                pmsigma = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.60
                pm_base = float(np.nanmedian(y_raw[mask]))/100.0
                pm_feats = [name for (name,_,_) in use]
                st.success(f"PM% model trained (n={len(yf)}; feats: {', '.join(pm_feats)}).")
            else:
                pm_base = float(np.nanmedian(y_raw))/100.0 if np.isfinite(np.nanmedian(y_raw)) else 0.139
                st.info(f"PM% fallback baseline used (median≈{pm_base*100:.1f}%).")
    else:
        st.info(f"PM% fallback baseline used (median≈{pm_base*100:.1f}%).")

    def _pm_pct_predict_row(r) -> float:
        if pmcoef is None or not pm_feats:
            p = float(np.clip(pm_base, 0.02, 0.95))
        else:
            vals = []
            for name in pm_feats:
                if   name == "ln_mcap":   vals.append(_safe_log(r.get("mcap_m")))
                elif name == "ln_gapf":   vals.append(_safe_log(_nz(r.get("gap_pct"),0.0)/100.0))
                elif name == "ln_atr":    vals.append(_safe_log(r.get("atr_usd")))
                elif name == "ln_float":  vals.append(_safe_log(r.get("float_m")))
                elif name == "ln1p_rvol": vals.append(_safe_log(1.0 + _nz(r.get("rvol"),0.0)))
                elif name == "catalyst":  vals.append(float(_nz(r.get("catalyst"),0.0)))
                else: vals.append(0.0)
            z = float(np.dot(vals, pmcoef) + pmbias)
            p = 1.0/(1.0 + math.exp(-z))
        return float(np.clip(p, 0.02, 0.95))

    # ---------- (A2) Direct Day-Volume model on ln(Daily Vol M) ----------
    dv_coef = None; dv_bias = 0.0; dv_sigma = 0.65; dv_feats: List[str] = []; dv_r2 = 0.0
    if "daily_vol_m" in df.columns and df["daily_vol_m"].notna().sum() >= 12:
        y_ln = np.log(np.clip(df["daily_vol_m"].to_numpy(float), 1e-6, None))
        dv_candidates = [
            ("ln_mcap",   lambda r: _safe_log(r.get("mcap_m")),                       "mcap_m"),
            ("ln_gapf",   lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0),       "gap_pct"),
            ("ln_atr",    lambda r: _safe_log(r.get("atr_usd")),                      "atr_usd"),
            ("catalyst",  lambda r: float(_nz(r.get("catalyst"),0.0)),               "catalyst"),
            # stabilizers (if present)
            ("ln_float",  lambda r: _safe_log(r.get("float_m")),                      "float_m"),
            ("ln1p_rvol", lambda r: _safe_log(1.0 + _nz(r.get("rvol"),0.0)),         "rvol"),
        ]
        dv_use = [c for c in dv_candidates if c[2] in df.columns]
        if dv_use:
            Xdv = np.vstack([[f(df.iloc[i].to_dict()) for (name,f,_) in dv_use] for i in range(len(df))]).astype(float)
            mask = np.isfinite(y_ln) & np.all(np.isfinite(Xdv), axis=1)
            Xf, yf = Xdv[mask], y_ln[mask]
            if Xf.shape[0] >= 12:
                dv_coef = ridge_fit(Xf, yf, l2=0.8)
                pred_ln = Xf @ dv_coef
                resid = yf - pred_ln
                dv_bias = float(np.mean(resid))
                dv_sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.65
                ss_res = float(np.sum((resid - np.mean(resid))**2))
                ss_tot = float(np.sum((yf - np.mean(yf))**2))
                dv_r2  = max(0.0, 1.0 - ss_res / max(1e-9, ss_tot))
                dv_feats = [name for (name,_,_) in dv_use]
                st.success(f"Direct DayVol model (n={len(yf)}; R²={dv_r2:.3f}; feats: {', '.join(dv_feats)}).")

    # ---------- (A3) Evaluate PM%-inversion fit on training (ln space) ----------
    inv_r2 = 0.0; inv_sigma = 0.75
    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
        pred_day_inv = []
        for i in range(len(df)):
            r = df.iloc[i].to_dict()
            p = _pm_pct_predict_row(r)
            pm = _nz(r.get("pm_vol_m"), 0.0)
            pred_day_inv.append(max(pm, pm / max(1e-6, p)))
        pred_day_inv = np.array(pred_day_inv, dtype=float)
        mask = np.isfinite(df["daily_vol_m"]) & np.isfinite(pred_day_inv)
        y_ln = np.log(np.clip(df.loc[mask, "daily_vol_m"].to_numpy(float), 1e-6, None))
        yhat_ln = np.log(np.clip(pred_day_inv[mask], 1e-6, None))
        resid = y_ln - yhat_ln
        inv_sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else inv_sigma
        ss_res = float(np.sum((resid - np.mean(resid))**2))
        ss_tot = float(np.sum((y_ln - np.mean(y_ln))**2))
        inv_r2  = max(0.0, 1.0 - ss_res / max(1e-9, ss_tot))
        st.info(f"PM% inversion fit on train: R²={inv_r2:.3f}, σ_ln≈{inv_sigma:.2f}.")

    # ---------- (B) FT classifier ----------
    # Build PredDay per row (using blended scheme below during live; for training we use the inversion estimate
    # to get ln(PredDay) signal; if direct model exists, ln(PredDay) informational content overlaps—still helpful.)
    pred_day_train = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        p = _pm_pct_predict_row(r)
        pm = _nz(r.get("pm_vol_m"), 0.0)
        pred_day_train.append(max(pm, pm / max(1e-6, p)))
    df["pred_day_m"] = np.array(pred_day_train, dtype=float)

    # FT features
    def _ln1p_pmvol(r): return _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)
    def _ln_fr(r):
        pm = _nz(r.get("pm_vol_m"),0.0); fl = _nz(r.get("float_m"),0.0)
        return math.log(max(1e-6, pm / max(1e-6, fl)))
    def _ln1p_pmmc(r):
        pm_dol = _nz(r.get("pm_dol_m"),0.0); mc = _nz(r.get("mcap_m"),0.0)
        return _safe_log(pm_dol / max(1e-6, mc) + 1.0)

    ft_candidates = [
        ("ln_mcap",      lambda r: _safe_log(r.get("mcap_m")),                       ["mcap_m"]),
        ("ln_gapf",      lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0),       ["gap_pct"]),
        ("ln_atr",       lambda r: _safe_log(r.get("atr_usd")),                      ["atr_usd"]),
        ("ln_float",     lambda r: _safe_log(r.get("float_m")),                      ["float_m"]),
        ("ln1p_rvol",    lambda r: _safe_log(1.0 + _nz(r.get("rvol"),0.0)),          ["rvol"]),
        ("ln1p_pmvol",   _ln1p_pmvol,                                               ["pm_vol_m"]),
        ("ln_fr",        _ln_fr,                                                     ["pm_vol_m","float_m"]),
        ("ln1p_pmmc",    _ln1p_pmmc,                                                 ["pm_dol_m","mcap_m"]),
        ("catalyst",     lambda r: float(_nz(r.get("catalyst"),0.0)),                ["catalyst"]),
        ("ln_pred_day",  lambda r: _safe_log(r.get("pred_day_m")),                   ["pred_day_m"]),
    ]
    ft_use = [(n,f,req) for (n,f,req) in ft_candidates if all(k in df.columns for k in req)]

    Xft = np.vstack([[f(df.iloc[i].to_dict()) for (n,f,_) in ft_use] for i in range(len(df))]).astype(float) if ft_use else np.zeros((len(df),0))
    y_ft = df["FT"].to_numpy(dtype=float)
    Xft = np.nan_to_num(Xft, nan=0.0, posinf=0.0, neginf=0.0)
    y_ft = (np.nan_to_num(y_ft, nan=0.0) >= 0.5).astype(float)

    ft_coef = None; ft_bias = 0.0; ft_feats = [name for (name,_,_) in ft_use]
    if Xft.shape[0] >= 12 and np.unique(y_ft).size == 2 and Xft.shape[1] > 0:
        ft_coef, ft_bias = logit_fit(Xft, y_ft, l2=1.0, max_iter=120, tol=1e-6)
        st.success(f"FT classifier trained (n={Xft.shape[0]}; feats: {', '.join(ft_feats)}).")
    else:
        st.error("Unable to train FT classifier (need ≥12 rows, both classes, and valid features).")

    # Calibration thresholds from train preds (with floors)
    if ft_coef is not None:
        p_cal = logit_inv(ft_bias + Xft @ ft_coef)
        def _q(p): return float(np.quantile(p_cal, p)) if p_cal.size else 0.5
        odds_cuts = {
            "very_high": max(0.85, _q(0.98)),
            "high":      max(0.70, _q(0.90)),
            "moderate":  max(0.55, _q(0.65)),
            "low":       max(0.40, _q(0.35)),
        }
        grade_cuts = {
            "App": max(0.92, _q(0.995)),
            "Ap":  max(0.85, _q(0.97)),
            "A":   max(0.75, _q(0.90)),
            "B":   max(0.65, _q(0.65)),
            "C":   max(0.50, _q(0.35)),
        }
    else:
        odds_cuts = {"very_high":0.85,"high":0.70,"moderate":0.55,"low":0.40}
        grade_cuts = {"App":0.92,"Ap":0.85,"A":0.75,"B":0.65,"C":0.50}

    # Save artifacts
    st.session_state.ARTIFACTS = {
        # PM% model
        "pmcoef": pmcoef, "pmbias": pmbias, "pmsigma": pmsigma, "pm_base": pm_base, "pm_feats": pm_feats,
        # Direct DayVol
        "dv_coef": dv_coef, "dv_bias": dv_bias, "dv_sigma": dv_sigma, "dv_feats": dv_feats,
        # Fit scores for blending
        "dv_r2": dv_r2, "inv_r2": inv_r2, "inv_sigma": inv_sigma,
        # FT
        "ft_coef": ft_coef, "ft_bias": ft_bias, "ft_feats": ft_feats,
    }
    st.session_state.ODDS_CUTS = odds_cuts
    st.session_state.GRADE_CUTS = grade_cuts
    st.success("Learning complete.")

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

# ============================== Inference helpers ==============================
def _pm_pct_predict(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, rvol: float, catalyst: float) -> Tuple[float,float,float]:
    """Predict PM% of day (as fraction) with CI68 using learned model or baseline. Returns (p, lo, hi)."""
    A = st.session_state.ARTIFACTS or {}
    pmcoef = A.get("pmcoef"); pmbias = float(A.get("pmbias") or 0.0)
    pmsigma = float(A.get("pmsigma") or 0.60); pm_base = float(A.get("pm_base") or 0.139)
    feats = A.get("pm_feats") or []
    if pmcoef is None or not feats:
        p = float(np.clip(pm_base, 0.02, 0.95))
    else:
        vals = []
        for name in feats:
            if   name == "ln_mcap":   vals.append(_safe_log(mc_m))
            elif name == "ln_gapf":   vals.append(_safe_log(_nz(gap_pct,0.0)/100.0))
            elif name == "ln_atr":    vals.append(_safe_log(atr_usd))
            elif name == "ln_float":  vals.append(_safe_log(float_m))
            elif name == "ln1p_rvol": vals.append(_safe_log(1.0 + _nz(rvol,0.0)))
            elif name == "catalyst":  vals.append(float(_nz(catalyst,0.0)))
            else: vals.append(0.0)
        z = float(np.dot(vals, pmcoef) + pmbias)
        p = 1.0/(1.0 + math.exp(-z))
    p = float(np.clip(p, 0.02, 0.95))
    logit = math.log(p/(1.0-p))
    lo = 1.0/(1.0+math.exp(-(logit - pmsigma)))
    hi = 1.0/(1.0+math.exp(-(logit + pmsigma)))
    lo = float(np.clip(lo, 0.01, 0.98)); hi = float(np.clip(hi, 0.01, 0.98))
    return p, lo, hi

def _predict_dayvol_direct(mc_m: float, gap_pct: float, atr_usd: float,
                           float_m: float, rvol: float, catalyst: float) -> Optional[float]:
    """Direct log-linear DayVol prediction from learned DV model; returns None if model not available."""
    A = st.session_state.ARTIFACTS or {}
    coef, bias, feats = A.get("dv_coef"), A.get("dv_bias"), A.get("dv_feats") or []
    if coef is None or not feats:
        return None
    vals = []
    for name in feats:
        if   name == "ln_mcap":   vals.append(_safe_log(mc_m))
        elif name == "ln_gapf":   vals.append(_safe_log(_nz(gap_pct,0.0)/100.0))
        elif name == "ln_atr":    vals.append(_safe_log(atr_usd))
        elif name == "catalyst":  vals.append(float(_nz(catalyst,0.0)))
        elif name == "ln_float":  vals.append(_safe_log(float_m))
        elif name == "ln1p_rvol": vals.append(_safe_log(1.0 + _nz(rvol,0.0)))
        else: vals.append(0.0)
    ln_pred = float(np.dot(vals, coef) + bias)
    return float(max(1e-6, math.exp(ln_pred)))

def predict_day_volume_blend(pm_vol_m: float, mc_m: float, gap_pct: float, atr_usd: float,
                             float_m: float, rvol: float, catalyst: float) -> Tuple[float,float,float,float,float]:
    """
    Returns:
      pred_M, ci68_lo_M, ci68_hi_M, implied_pm_frac_pct, blend_weight_direct
    """
    pm_vol_m = _nz(pm_vol_m, 0.0)

    # Inversion path
    p_pm, p_lo, p_hi = _pm_pct_predict(mc_m, gap_pct, atr_usd, float_m, rvol, catalyst)
    pred_inv = max(pm_vol_m, pm_vol_m / max(1e-6, p_pm))

    # Direct path
    pred_dir = _predict_dayvol_direct(mc_m, gap_pct, atr_usd, float_m, rvol, catalyst)
    if pred_dir is None:
        pred_dir = pred_inv
    else:
        pred_dir = max(pred_dir, pm_vol_m)

    # Blend weight from R²
    A = st.session_state.ARTIFACTS or {}
    dv_r2  = float(A.get("dv_r2") or 0.0)
    inv_r2 = float(A.get("inv_r2") or 0.0)
    if dv_r2 + inv_r2 <= 1e-9:
        w_dir = 0.65
    else:
        w_dir = dv_r2 / (dv_r2 + inv_r2)

    # Blend in ln-space
    ln_pred = w_dir * math.log(pred_dir) + (1.0 - w_dir) * math.log(pred_inv)
    pred = float(max(pm_vol_m, math.exp(ln_pred)))

    # CI68: blend variances in ln-space
    dv_sig  = float(A.get("dv_sigma") or 0.65)
    inv_sig = float(A.get("inv_sigma") or 0.75)
    sig = math.sqrt(max(1e-9, (w_dir*dv_sig)**2 + ((1.0 - w_dir)*inv_sig)**2))
    lo = float(max(pm_vol_m, math.exp(ln_pred - sig)))
    hi = float(max(pm_vol_m, math.exp(ln_pred + sig)))

    implied_pm_frac_pct = 100.0 * pm_vol_m / max(1e-6, pred)
    return pred, lo, hi, implied_pm_frac_pct, w_dir

def predict_ft_prob(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float,
                    rvol: float, pm_vol_m: float, pm_dol_m: float, catalyst: float) -> float:
    """FT probability using learned logistic classifier (with ln(PredDay) among features)."""
    A = st.session_state.ARTIFACTS or {}
    coef = A.get("ft_coef"); bias = float(A.get("ft_bias") or 0.0)
    ft_feats = A.get("ft_feats") or []

    # Use blended PredDay
    pred_day, _, _, _, _ = predict_day_volume_blend(pm_vol_m, mc_m, gap_pct, atr_usd, float_m, rvol, catalyst)

    vals = []
    for name in ft_feats:
        if   name == "ln_mcap":     vals.append(_safe_log(mc_m))
        elif name == "ln_gapf":     vals.append(_safe_log(_nz(gap_pct,0.0)/100.0))
        elif name == "ln_atr":      vals.append(_safe_log(atr_usd))
        elif name == "ln_float":    vals.append(_safe_log(float_m))
        elif name == "ln1p_rvol":   vals.append(_safe_log(1.0 + _nz(rvol,0.0)))
        elif name == "ln1p_pmvol":  vals.append(_safe_log(_nz(pm_vol_m,0.0) + 1.0))
        elif name == "ln_fr":       vals.append(math.log(max(1e-6, _nz(pm_vol_m,0.0)/max(1e-6, _nz(float_m,0.0)))))
        elif name == "ln1p_pmmc":   vals.append(_safe_log(_nz(pm_dol_m,0.0)/max(1e-6, _nz(mc_m,0.0)) + 1.0))
        elif name == "catalyst":    vals.append(float(_nz(catalyst,0.0)))
        elif name == "ln_pred_day": vals.append(_safe_log(pred_day))
        else: vals.append(0.0)

    if coef is None or not vals:
        return 0.50
    return float(np.clip(logit_inv(bias + np.dot(vals, coef)), 1e-3, 1-1e-3))

def _prob_to_odds(prob: float, cuts: Dict[str, float]) -> str:
    if prob >= cuts.get("very_high",0.85): return "Very High Odds"
    if prob >= cuts.get("high",0.70):      return "High Odds"
    if prob >= cuts.get("moderate",0.55):  return "Moderate Odds"
    if prob >= cuts.get("low",0.40):       return "Low Odds"
    return "Very Low Odds"

def _prob_to_grade(prob: float, cuts: Dict[str, float]) -> str:
    if prob >= cuts.get("App",0.92): return "A++"
    if prob >= cuts.get("Ap",0.85):  return "A+"
    if prob >= cuts.get("A",0.75):   return "A"
    if prob >= cuts.get("B",0.65):   return "B"
    if prob >= cuts.get("C",0.50):   return "C"
    return "D"

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

# ============================== Add Stock ==============================
with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2,1.2,1.0])
        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL", 0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"], index=0)
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        cat = 1.0 if catalyst_flag=="Yes" else 0.0

        # Blended DayVol prediction (PredVol ≥ PMVol)
        pred_vol_m, ci68_l, ci68_u, pmfrac_pct, w_dir = predict_day_volume_blend(
            pm_vol_m, mc_m, gap_pct, atr_usd, float_m, rvol, cat
        )

        # FT probability (uses ln(PredDay))
        ft_prob = predict_ft_prob(mc_m, gap_pct, atr_usd, float_m, si_pct, rvol, pm_vol_m, pm_dol_m, cat)
        odds_cuts = st.session_state.get("ODDS_CUTS", {})
        grade_cuts = st.session_state.get("GRADE_CUTS", {})
        odds_name = _prob_to_odds(ft_prob, odds_cuts)
        level     = _prob_to_grade(ft_prob, grade_cuts)

        row = {
            "Ticker": ticker,
            "Odds": odds_name,
            "Level": level,
            "FinalScore": round(ft_prob*100.0, 2),
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PM%_implied": round(pmfrac_pct, 1),
            "Blend_w_direct": round(w_dir, 2),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']:.2f})")

    # preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c,d,e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        c.metric("Grade", l.get('Level','—'))
        d.metric("Odds", l.get('Odds','—'))
        e.metric("PredVol (M)", f"{l.get('PredVol_M',0):.2f}")
        st.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f} – {l.get('PredVol_CI68_U',0):.2f} M · "
            f"PM% implied: {l.get('PM%_implied',0):.1f}% · Blend direct={l.get('Blend_w_direct',0):.2f}"
        )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = [
            "Ticker","Odds","Level","FinalScore",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM%_implied","Blend_w_direct"
        ]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level") else 0.0
        st.dataframe(
            df[cols_to_show], use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("FT Probability %", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("PredVol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("PredVol CI68 High (M)", format="%.2f"),
                "PM%_implied": st.column_config.NumberColumn("PM% implied", format="%.1f"),
                "Blend_w_direct": st.column_config.NumberColumn("Blend weight (direct)", format="%.2f"),
            }
        )
        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )
    else:
        st.info("Add at least one stock.")
