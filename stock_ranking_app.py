# app.py — Premarket Ranking (Predictive‑Only, trains from uploaded workbook)
# ---------------------------------------------------------------------------------
# • Learns BOTH: (A) Day Volume model (log‑space ridge) and (B) FT classifier.
# • Also learns an auxiliary regression for Max Push Daily % and blends it into FT.
# • Uses robust column auto‑detection. No scikit‑learn required.
# • Numeric inputs accept commas (e.g., "5,05").
# • Keeps UI minimal: Upload → Learn → Add/Score → Ranking.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking — Predictive", layout="wide")
st.title("Premarket Stock Ranking — Predictive")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
      .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:.78rem; }
      .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
      .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
      .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 11.5px; color:#374151; }
      [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
      [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}

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

# smart column picker
_defmap = {
    "FT": ["ft", "FT"],
    "MAXPCT": ["max push daily %", "max push %", "max push daily%", "max push", "max push d %"],
    "GAP": ["gap %", "gap%", "premarket gap", "gap"],
    "ATR": ["atr", "atr $", "atr$", "atr (usd)"],
    "RVOL": ["rvol @ bo", "rvol", "relative volume"],
    "PMVOL": ["pm vol (m)", "premarket vol (m)", "pm volume (m)", "pm shares (m)"],
    "PM$": ["pm $vol (m)", "pm dollar vol (m)", "pm $ volume (m)", "pm $vol"],
    "FLOAT": ["float m shares", "public float (m)", "float (m)", "float"],
    "MCAP": ["marketcap m", "market cap (m)", "mcap m", "mcap"],
    "SI": ["si", "short interest %", "short float %", "short interest (float) %"],
    "DAILY": ["daily vol (m)", "day volume (m)", "volume (m)"],
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

# ---------- math utils ----------

def _safe_log(x: float, eps: float = 1e-8) -> float:
    x = float(x) if x is not None else 0.0
    return math.log(max(x, eps))

# Ridge linear regression (closed-form via normal equations with L2)

def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    n, k = X.shape
    I = np.eye(k)
    return np.linalg.solve(X.T @ X + l2 * I, X.T @ y)

# Logistic regression (IRLS) with L2 on weights (no penalty on intercept)

def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 60, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    w = np.zeros(k+1)
    R = np.eye(k+1); R[0,0] = 0.0; R *= l2
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0/(1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p * (1 - p)
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H = Xb.T @ WX + R
        g = Xb.T @ (y - p)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w + delta
        if np.linalg.norm(delta) < tol:
            w = w_new; break
        w = w_new
    return w[1:].astype(float), float(w[0])

def logit_predict(X: np.ndarray, coef_: np.ndarray, intercept_: float) -> np.ndarray:
    z = intercept_ + X @ coef_
    z = np.clip(z, -35, 35)
    return 1.0/(1.0 + np.exp(-z))

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")

def _load_and_learn(xls: pd.ExcelFile, sheet: str, bins: int = 4) -> None:
    raw = pd.read_excel(xls, sheet)

    # --- column mapping ---
    col = {}
    for key, cands in _defmap.items():
        pick = _pick(raw, cands)
        if pick: col[key] = pick
    if "FT" not in col:
        st.error("No 'FT' column found in sheet.")
        return
    df = pd.DataFrame()
    df["FT"] = pd.to_numeric(raw[col["FT"]], errors="coerce")

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
    if "MAXPCT" in col:
        df["maxpush_pct"] = pd.to_numeric(raw[col["MAXPCT"]], errors="coerce")

    # derived
    if {"pm_vol_m","float_m"}.issubset(df.columns):
        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"]

    # --- (A) Learn Day Volume model in log space ---
    yv = None
    if "daily_vol_m" in df.columns:
        # target: ln(DailyVol_M)
        yv = np.log(np.clip(df["daily_vol_m"].astype(float).to_numpy(), 1e-6, None))
    else:
        st.warning("No 'Daily Vol (M)' column — will use fallback formula at inference.")

    # candidate regressors for volume
    Xv_cols: List[str] = []
    for c in ["mcap_m","gap_pct","atr_usd","float_m","si_pct"]:
        if c in df.columns:
            Xv_cols.append(c)
    Xv = None
    vol_coef = None
    vol_mu = None
    vol_sigma = None

    if yv is not None and len(Xv_cols) >= 2:
        # transform: ln(MCap), ln(GapFrac), ln(ATR), ln(Float), ln(1+SI) (if present)
        def _feat_map(r):
            mc = _safe_log(r.get("mcap_m", np.nan))
            gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
            at = _safe_log(r.get("atr_usd", np.nan))
            fl = _safe_log(r.get("float_m", np.nan))
            si = math.log(max(1e-6, 1.0 + (r.get("si_pct", np.nan) or 0.0)/100.0))
            return [mc, gp, at, fl, si]
        V = df.apply(_feat_map, axis=1, result_type="expand").to_numpy(dtype=float)
        mask = np.isfinite(yv) & np.all(np.isfinite(V), axis=1)
        V, yv_fit = V[mask], yv[mask]
        if V.size and yv_fit.size:
            vol_coef = ridge_fit(V, yv_fit, l2=1.5)
            resid = yv_fit - (V @ vol_coef)
            vol_mu = float(np.mean(resid))  # bias
            vol_sigma = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.60
            st.success(f"Day Volume model trained on n={len(yv_fit)} rows.")
        else:
            st.warning("Insufficient rows to fit Day Volume model — fallback will be used.")
    else:
        st.warning("Not enough features to fit Day Volume model — fallback will be used.")

    # --- (B1) Auxiliary regression: Max Push Daily % (logit of pct/100) ---
    maxpush_coef = None
    maxpush_bias = 0.0
    maxpush_scale = 1.0
    mp_median_ft1 = 12.0
    if "maxpush_pct" in df.columns:
        y_mp_raw = pd.to_numeric(df["maxpush_pct"], errors="coerce").to_numpy(dtype=float)
        y_mp_frac = np.clip(y_mp_raw/100.0, 1e-4, 0.999)
        y_mp = np.log(y_mp_frac/(1.0 - y_mp_frac))  # logit
        # use same V matrix as for volume (if defined), else build a simple one
        if 'V' not in locals() or V is None or not V.size:
            def _feat_simple(r):
                mc = _safe_log(r.get("mcap_m", np.nan))
                gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
                at = _safe_log(r.get("atr_usd", np.nan))
                fr = math.log(max(1e-6, (r.get("pm_vol_m", np.nan) or 0.0)/max(1e-6, r.get("float_m", np.nan) or 0.0)))
                return [mc, gp, at, fr]
            Xmp = df.apply(_feat_simple, axis=1, result_type="expand").to_numpy(dtype=float)
        else:
            Xmp = V
        mask = np.all(np.isfinite(Xmp), axis=1) & np.isfinite(y_mp)
        Xmp_f, ymp_f = Xmp[mask], y_mp[mask]
        if Xmp_f.size and ymp_f.size:
            beta = ridge_fit(Xmp_f, ymp_f, l2=1.5)
            maxpush_coef = beta
            # calibration stats
            pred = Xmp_f @ beta
            maxpush_bias = float(np.mean(ymp_f - pred))
            maxpush_scale = float(np.std(ymp_f - pred, ddof=1)) if ymp_f.size > 2 else 0.6
            # class medians
            try:
                mp_ft = y_mp_raw[(df["FT"]==1).to_numpy() & mask]
                if mp_ft.size: mp_median_ft1 = float(np.median(mp_ft))
            except Exception:
                pass
            st.success(f"MaxPush% model trained on n={len(ymp_f)} rows (median FT=1 ~ {mp_median_ft1:.1f}%).")
        else:
            st.warning("Insufficient rows to fit MaxPush% model — FT will rely on classifier only.")
    else:
        st.info("No 'Max Push Daily %' found — FT will rely on classifier only.")

    # --- (B2) FT classifier features ---
    # We will use engineered features (log transforms) + optional predicted day vol and predicted maxpush.

    # build design matrix for FT
    y_ft = pd.to_numeric(df["FT"], errors="coerce").to_numpy(dtype=float)

    def _ft_feats_row(r) -> List[float]:
        # core transforms
        mc = _safe_log(r.get("mcap_m", np.nan))
        gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
        at = _safe_log(r.get("atr_usd", np.nan))
        fl = _safe_log(r.get("float_m", np.nan))
        si = math.log(max(1e-6, 1.0 + (r.get("si_pct", np.nan) or 0.0)/100.0))
        pm = _safe_log(r.get("pm_vol_m", np.nan) + 1.0)
        fr = math.log(max(1e-6, (r.get("pm_vol_m", np.nan) or 0.0)/max(1e-6, r.get("float_m", np.nan) or 0.0)))
        pmmc = _safe_log((r.get("pm_dol_m", np.nan) or 0.0) / max(1e-6, r.get("mcap_m", np.nan) or 0.0) + 1.0)
        # predicted day volume (if model exists)
        pred_ln_dvol = 0.0
        if 'vol_coef' in locals() and vol_coef is not None:
            xV = np.array([mc, gp, at, fl, si], dtype=float)
            pred_ln_dvol = float(xV @ vol_coef + (vol_mu or 0.0))
        # predicted maxpush logit (if model exists)
        pred_mp_logit = 0.0
        if 'maxpush_coef' in locals() and maxpush_coef is not None:
            if 'V' in locals() and V is not None and len(V.shape)==2 and V.shape[1]==len(maxpush_coef):
                xmp = np.array([mc, gp, at, fl, si], dtype=float) if V.shape[1]==5 else np.array([mc, gp, at, fr], dtype=float)
            else:
                xmp = np.array([mc, gp, at, fr], dtype=float)
            pred_mp_logit = float(xmp @ maxpush_coef + (maxpush_bias or 0.0))
        return [mc, gp, at, fl, si, pm, fr, pmmc, pred_ln_dvol, pred_mp_logit]

    Xft = df.apply(_ft_feats_row, axis=1, result_type="expand").to_numpy(dtype=float)
    mask_ft = np.isfinite(y_ft) & np.all(np.isfinite(Xft), axis=1)
    Xft_fit, yft_fit = Xft[mask_ft], y_ft[mask_ft]

    ft_coef = None
    ft_bias = 0.0
    if Xft_fit.size and yft_fit.size and np.unique(yft_fit).size == 2:
        ft_coef, ft_bias = logit_fit(Xft_fit, yft_fit, l2=1.0, max_iter=80, tol=1e-6)
        st.success(f"FT classifier trained on n={len(yft_fit)} rows.")
    else:
        st.error("Unable to train FT classifier (need binary FT and valid features).")

    # Calibration for grade/odds
    if ft_coef is not None:
        p_cal = logit_predict(Xft_fit, ft_coef, ft_bias)
        def _q(p):
            return float(np.quantile(p_cal, p)) if p_cal.size else 0.5
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

    # Save all learned artifacts
    st.session_state.MODELS = {
        "vol_coef": vol_coef,
        "vol_mu": vol_mu,
        "vol_sigma": vol_sigma,
        "maxpush_coef": maxpush_coef,
        "maxpush_bias": maxpush_bias,
        "maxpush_scale": maxpush_scale,
        "mp_median_ft1": mp_median_ft1,
        "ft_coef": ft_coef,
        "ft_bias": ft_bias,
    }
    st.session_state.ODDS_CUTS = odds_cuts
    st.session_state.GRADE_CUTS = grade_cuts
    st.success("Learning complete.")

# Sheet controls (defaults for your workbook)
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
col1, col2 = st.columns([1,1])
with col1:
    learn_btn = st.button("Learn from uploaded sheet", use_container_width=True)
with col2:
    # Optional fast path if file is on disk (e.g., /mnt/data/PMH Database.xlsx)
    local_path = st.text_input("Or read local path (server)", "")
    learn_local_btn = st.button("Learn from local path", use_container_width=True)

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

if learn_local_btn:
    try:
        if not os.path.isfile(local_path):
            st.error("Local path not found.")
        else:
            xls = pd.ExcelFile(local_path)
            if sheet_name not in xls.sheet_names:
                st.error(f"Sheet '{sheet_name}' not found. Available: {xls.sheet_names}")
            else:
                _load_and_learn(xls, sheet_name)
    except Exception as e:
        st.error(f"Learning failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"]) 

# ============================== Inference helpers ==============================

def predict_day_volume_m(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float) -> Tuple[float, float, float]:
    """Return (pred_m, ci68_lo, ci68_hi). If model missing, fallback to simple formula."""
    M = st.session_state.MODELS or {}
    coef = M.get("vol_coef")
    if coef is not None:
        mc = _safe_log(mc_m)
        gp = _safe_log((gap_pct or 0.0)/100.0)
        at = _safe_log(atr_usd)
        fl = _safe_log(float_m)
        si = math.log(max(1e-6, 1.0 + (si_pct or 0.0)/100.0))
        x = np.array([mc, gp, at, fl, si], dtype=float)
        lnY = float(x @ coef + (M.get("vol_mu") or 0.0))
        pred = float(math.exp(lnY))
        sigma = float(M.get("vol_sigma") or 0.60)
        lo = pred * math.exp(-1.0*sigma)
        hi = pred * math.exp(+1.0*sigma)
        return pred, lo, hi
    # fallback from your prior heuristic
    e=1e-6
    ln_y = 3.1435 + 0.1608*math.log(max(mc_m,e)) + 0.6704*math.log(max(gap_pct/100.0,e)) - 0.3878*math.log(max(atr_usd,e))
    pred = math.exp(ln_y)
    sigma=0.60
    return pred, pred*math.exp(-sigma), pred*math.exp(+sigma)


def predict_maxpush_pct(mc_m: float, gap_pct: float, atr_usd: float, pm_vol_m: float, float_m: float) -> Optional[float]:
    M = st.session_state.MODELS or {}
    beta = M.get("maxpush_coef")
    if beta is None:
        return None
    mc = _safe_log(mc_m)
    gp = _safe_log((gap_pct or 0.0)/100.0)
    at = _safe_log(atr_usd)
    fr = math.log(max(1e-6, pm_vol_m/max(1e-6, float_m)))
    # choose dimensionality by learned beta
    if len(beta)==5:
        # used [mc,gp,at,fl,si] during train; approximate with fl,si as zeros in live
        x = np.array([mc,gp,at,0.0,0.0], dtype=float)
    else:
        x = np.array([mc,gp,at,fr], dtype=float)
    z = float(x @ beta + (M.get("maxpush_bias") or 0.0))
    p = 1.0/(1.0 + math.exp(-z))
    return float(p*100.0)


def predict_ft_prob(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float,
                    pm_vol_m: float, pm_dol_m: float) -> float:
    M = st.session_state.MODELS or {}
    coef = M.get("ft_coef")
    if coef is None:
        return 0.50
    # build features consistent with training
    mc = _safe_log(mc_m)
    gp = _safe_log((gap_pct or 0.0)/100.0)
    at = _safe_log(atr_usd)
    fl = _safe_log(float_m)
    si = math.log(max(1e-6, 1.0 + (si_pct or 0.0)/100.0))
    pm = _safe_log(pm_vol_m + 1.0)
    fr = math.log(max(1e-6, pm_vol_m/max(1e-6, float_m)))
    pmmc = _safe_log((pm_dol_m or 0.0) / max(1e-6, mc_m) + 1.0)
    # predicted auxiliaries
    pred_ln_dvol = 0.0
    if M.get("vol_coef") is not None:
        xV = np.array([mc,gp,at,fl,si], dtype=float)
        pred_ln_dvol = float(xV @ M["vol_coef"] + (M.get("vol_mu") or 0.0))
    pred_mp_logit = 0.0
    mp_pred_pct = predict_maxpush_pct(mc_m, gap_pct, atr_usd, pm_vol_m, float_m)
    if mp_pred_pct is not None:
        q = np.clip((mp_pred_pct/100.0), 1e-4, 0.999)
        pred_mp_logit = math.log(q/(1.0-q))
    x = np.array([mc,gp,at,fl,si,pm,fr,pmmc,pred_ln_dvol,pred_mp_logit], dtype=float)
    p_cls = float(logit_predict(x[None,:], coef, M.get("ft_bias", 0.0))[0])

    # Blend with push‑implied probability using median FT=1 as soft threshold
    p_blend = p_cls
    if mp_pred_pct is not None:
        thresh = float(M.get("mp_median_ft1", 12.0))  # %
        scale  = max(2.0, float(M.get("maxpush_scale", 6.0))*8.0)  # softer
        p_push = 1.0/(1.0 + math.exp(-(mp_pred_pct - thresh)/scale))
        # convex blend leaning on classifier
        p_blend = 0.75*p_cls + 0.25*p_push
    return float(np.clip(p_blend, 1e-3, 1-1e-3))


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
            notes = st.text_input("Notes / Catalyst (optional)", "")
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        models = st.session_state.MODELS or {}
        # predictions
        pred_vol_m, ci68_l, ci68_u = predict_day_volume_m(mc_m, gap_pct, atr_usd, float_m, si_pct)
        ft_prob = predict_ft_prob(mc_m, gap_pct, atr_usd, float_m, si_pct, pm_vol_m, pm_dol_m)
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
            "Notes": notes,
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
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M"
        )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PredVol_M","PredVol_CI68_L","PredVol_CI68_U","Notes"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level","Notes") else 0.0
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
                "Notes": st.column_config.TextColumn("Notes"),
            }
        )
        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )
    else:
        st.info("Add at least one stock.")
