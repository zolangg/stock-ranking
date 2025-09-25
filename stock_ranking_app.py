# app.py — Premarket Ranking (Predictive-Only, trains from uploaded workbook)
# ---------------------------------------------------------------------------------
# • Learns: (A) PM%-of-Day model (logit ridge) → Predicted Day Volume + CI (and never below PM).
# • (B) FT classifier (logistic IRLS) using dataset vars + ln(PredDay).
# • (C) Optional Max Push Daily % regression; blended (20%) into FT.
# • Robust column auto-detection matching the Legend.
# • Numeric inputs accept commas (e.g., "5,05").
# • Minimal UI: Upload → Learn → Add/Score → Ranking.

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

# smart column picker (aligned with Legend)
_DEF = {
    "FT": ["FT"],
    "MAXPCT": ["Max Push Daily %", "Max Push Daily (%)", "Max Push %"],
    "GAP": ["Gap %", "Gap"],
    "ATR": ["ATR $", "ATR", "Daily ATR"],  # include "Daily ATR" used in Legend
    "RVOL": ["RVOL @ BO", "RVOL", "Relative Volume"],
    "PMVOL": ["PM Vol (M)", "Premarket Vol (M)"],
    "PM$": ["PM $Vol (M)", "PM Dollar Vol (M)"],  # optional, handled gracefully
    "FLOAT": ["Float M Shares", "Public Float (M)", "Float (M)"],
    "MCAP": ["MarketCap M", "Market Cap (M)"],
    "SI": ["Short Interest %"],
    "DAILY": ["Daily Vol (M)", "Day Volume (M)"],
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

# ---------- math & models ----------
def _safe_log(x: float, eps: float = 1e-8) -> float:
    x = float(x) if x is not None else 0.0
    return math.log(max(x, eps))

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

col1, col2 = st.columns([1,1])
with col1:
    learn_btn = st.button("Learn from uploaded sheet", use_container_width=True)
with col2:
    local_path = st.text_input("Or read local path (server)", "")
    learn_local_btn = st.button("Learn from local path", use_container_width=True)

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
    df["FT"] = pd.to_numeric(raw[col["FT"]], errors="coerce").astype(float)

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
        cat_raw = pd.to_numeric(raw[col["CAT"]], errors="coerce")
        df["catalyst"] = cat_raw.clip(0,1)

    # derived
    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"].replace(0, np.nan)
    if {"pm_vol_m","float_m"}.issubset(df.columns):
        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

    # ---------- (A) Learn PM%-of-Day model (logit of pm_pct_daily/100) ----------
    pmcoef = None; pmbias = 0.0; pmsigma = 0.60; pm_base = 0.139  # base ≈ dataset median 13.9%
    if "pm_pct_daily" in df.columns and df["pm_pct_daily"].notna().any():
        y_raw = pd.to_numeric(df["pm_pct_daily"], errors="coerce").astype(float)
        y_frac = np.clip(y_raw/100.0, 0.005, 0.95)   # realistic band
        y = np.log(y_frac/(1.0 - y_frac))           # logit

        # features: ln mcap, ln gap_frac, ln atr, ln float, ln(1+rvol), catalyst
        def _feat_pm(r):
            mc = _safe_log(r.get("mcap_m", np.nan))
            gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
            at = _safe_log(r.get("atr_usd", np.nan))
            fl = _safe_log(r.get("float_m", np.nan))
            rv = _safe_log(1.0 + (r.get("rvol", np.nan) or 0.0))
            ca = float(r.get("catalyst", 0.0) or 0.0)
            return [mc,gp,at,fl,rv,ca]

        X = df.apply(_feat_pm, axis=1, result_type="expand").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        Xf, yf = X[mask], y[mask]
        if Xf.size and yf.size:
            pmcoef = ridge_fit(Xf, yf, l2=1.2)
            resid = yf - (Xf @ pmcoef)
            pmbias = float(np.mean(resid))
            pmsigma = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.60
            pm_base = float(np.median(y_raw[mask]))/100.0
            st.success(f"PM% model trained on n={len(yf)} rows (median≈{pm_base*100:.1f}%).")
        else:
            st.warning("Insufficient rows to fit PM% model — will fall back to median baseline.")
    else:
        st.warning("No 'PM Vol % of Daily' available — fall back to median baseline.")

    # ---------- (B1) Max Push Daily % regression (optional) ----------
    mpcoef = None; mpbias = 0.0; mpscale = 1.0; mp_med_ft1 = 12.0
    if "MAXPCT" in col:
        y_mp_raw = pd.to_numeric(raw[col["MAXPCT"]], errors="coerce").astype(float)
        if np.isfinite(y_mp_raw).any():
            y_mp_frac = np.clip(y_mp_raw/100.0, 1e-4, 0.999)
            y_mp = np.log(y_mp_frac/(1.0 - y_mp_frac))
            def _feat_mp(r):
                mc = _safe_log(r.get("mcap_m", np.nan))
                gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
                at = _safe_log(r.get("atr_usd", np.nan))
                fr = math.log(max(1e-6, (r.get("pm_vol_m", np.nan) or 0.0)/max(1e-6, r.get("float_m", np.nan) or 0.0)))
                ca = float(r.get("catalyst", 0.0) or 0.0)
                return [mc,gp,at,fr,ca]
            Xmp = df.apply(_feat_mp, axis=1, result_type="expand").to_numpy(dtype=float)
            mask = np.all(np.isfinite(Xmp), axis=1) & np.isfinite(y_mp)
            Xf, yf = Xmp[mask], y_mp[mask]
            if Xf.size and yf.size:
                mpcoef = ridge_fit(Xf, yf, l2=1.2)
                pred = Xf @ mpcoef
                mpbias = float(np.mean(yf - pred))
                mpscale = float(np.std(yf - pred, ddof=1)) if yf.size>2 else 0.6
                try:
                    mp_ft = y_mp_raw[(df["FT"]==1).to_numpy() & mask]
                    if mp_ft.size: mp_med_ft1 = float(np.median(mp_ft))
                except Exception:
                    pass
                st.success(f"MaxPush% model trained on n={len(yf)} rows (median FT=1 ≈ {mp_med_ft1:.1f}%).")
            else:
                st.warning("Insufficient rows to fit MaxPush% model — FT will rely on classifier only.")
        else:
            st.info("Max Push Daily % present but empty.")
    else:
        st.info("No 'Max Push Daily %' column.")

    # ---------- (B2) FT classifier ----------
    def _pm_pct_predict_row(r) -> float:
        if pmcoef is None:
            return float(np.clip(0.139, 0.02, 0.95))  # median baseline (~13.9%)
        mc = _safe_log(r.get("mcap_m", np.nan))
        gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
        at = _safe_log(r.get("atr_usd", np.nan))
        fl = _safe_log(r.get("float_m", np.nan))
        rv = _safe_log(1.0 + (r.get("rvol", np.nan) or 0.0))
        ca = float(r.get("catalyst", 0.0) or 0.0)
        z = float(np.dot([mc,gp,at,fl,rv,ca], pmcoef) + pmbias)
        p = float(1.0/(1.0 + math.exp(-z)))
        return float(np.clip(p, 0.02, 0.95))

    y_ft = df["FT"].to_numpy(dtype=float)

    def _ft_feats(r):
        mc = _safe_log(r.get("mcap_m", np.nan))
        gp = _safe_log((r.get("gap_pct", np.nan) or 0.0)/100.0)
        at = _safe_log(r.get("atr_usd", np.nan))
        fl = _safe_log(r.get("float_m", np.nan))
        si = math.log(max(1e-6, 1.0 + (r.get("si_pct", np.nan) or 0.0)/100.0))
        rv = _safe_log(1.0 + (r.get("rvol", np.nan) or 0.0))
        pm = _safe_log(r.get("pm_vol_m", np.nan) + 1.0)
        fr = math.log(max(1e-6, (r.get("pm_vol_m", np.nan) or 0.0)/max(1e-6, r.get("float_m", np.nan) or 0.0)))
        pmmc = _safe_log((r.get("pm_dol_m", np.nan) or 0.0) / max(1e-6, r.get("mcap_m", np.nan) or 0.0) + 1.0)
        ca = float(r.get("catalyst", 0.0) or 0.0)
        p_pct = _pm_pct_predict_row(r)                # fraction
        pred_day = max(1e-6, (r.get("pm_vol_m", 0.0) or 0.0) / max(1e-6, p_pct))  # ensure >= PM
        pred_day = max(pred_day, float(r.get("pm_vol_m", 0.0) or 0.0))
        ln_pred_day = _safe_log(pred_day)
        return [mc,gp,at,fl,si,rv,pm,fr,pmmc,ca,ln_pred_day]

    Xft = df.apply(_ft_feats, axis=1, result_type="expand").to_numpy(dtype=float)
    mask = np.isfinite(y_ft) & np.all(np.isfinite(Xft), axis=1)
    Xf, yf = Xft[mask], y_ft[mask]

    ft_coef = None; ft_bias = 0.0
    if Xf.size and yf.size and np.unique(yf).size == 2:
        ft_coef, ft_bias = logit_fit(Xf, yf, l2=1.0, max_iter=100, tol=1e-6)
        st.success(f"FT classifier trained on n={len(yf)} rows.")
    else:
        st.error("Unable to train FT classifier (need binary FT and valid features).")

    # Calibration for grade/odds
    if ft_coef is not None:
        p_cal = logit_inv(ft_bias + Xf @ ft_coef)
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
        "pmcoef": pmcoef,
        "pmbias": pmbias,
        "pmsigma": pmsigma,
        "pm_base": pm_base,
        # Max push
        "mpcoef": mpcoef,
        "mpbias": mpbias,
        "mpscale": mpscale,
        "mp_med_ft1": mp_med_ft1,
        # FT
        "ft_coef": ft_coef,
        "ft_bias": ft_bias,
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

# ============================== Inference helpers ==============================
def _pm_pct_predict(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, rvol: float, catalyst: float) -> Tuple[float,float,float]:
    """Predict PM% of day (as fraction) with CI68 using learned model or baseline.
       Returns (p, lo, hi) in FRAC (0..1)."""
    A = st.session_state.ARTIFACTS or {}
    pmcoef = A.get("pmcoef")
    pmbias = float(A.get("pmbias") or 0.0)
    pmsigma = float(A.get("pmsigma") or 0.60)
    base = float(A.get("pm_base") or 0.139)
    if pmcoef is None:
        p = base
    else:
        mc = _safe_log(mc_m)
        gp = _safe_log((gap_pct or 0.0)/100.0)
        at = _safe_log(atr_usd)
        fl = _safe_log(float_m)
        rv = _safe_log(1.0 + (rvol or 0.0))
        ca = float(catalyst or 0.0)
        z = float(np.dot([mc,gp,at,fl,rv,ca], pmcoef) + pmbias)
        p = 1.0/(1.0 + math.exp(-z))
    p = float(np.clip(p, 0.02, 0.95))
    # CI68 in logit space
    logit = math.log(p/(1.0-p))
    lo = 1.0/(1.0+math.exp(-(logit - pmsigma)))
    hi = 1.0/(1.0+math.exp(-(logit + pmsigma)))
    lo = float(np.clip(lo, 0.01, 0.98))
    hi = float(np.clip(hi, 0.01, 0.98))
    return p, lo, hi

def predict_day_volume_from_pm(pm_vol_m: float, p_pm: float, lo: float, hi: float) -> Tuple[float,float,float]:
    """Invert PM% to get day volume. Guarantees PredVol ≥ PM volume. Returns (pred, ci68_lo, ci68_hi)."""
    p = max(1e-6, min(0.95, p_pm))
    pred = pm_vol_m / p if p > 0 else float("nan")
    pred = max(pred, pm_vol_m)  # never below PM volume
    lo_v = max(pm_vol_m, pm_vol_m / max(1e-6, hi))
    hi_v = max(pm_vol_m, pm_vol_m / max(1e-6, lo))
    return float(pred), float(lo_v), float(hi_v)

def predict_ft_prob(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float,
                    rvol: float, pm_vol_m: float, pm_dol_m: float, catalyst: float) -> float:
    A = st.session_state.ARTIFACTS or {}
    coef = A.get("ft_coef")
    bias = float(A.get("ft_bias") or 0.0)

    # PM%-derived day volume
    p_pm, p_lo, p_hi = _pm_pct_predict(mc_m, gap_pct, atr_usd, float_m, rvol, catalyst)
    pred_vol_m, _, _ = predict_day_volume_from_pm(pm_vol_m, p_pm, p_lo, p_hi)

    # features for classifier (match training):
    mc = _safe_log(mc_m)
    gp = _safe_log((gap_pct or 0.0)/100.0)
    at = _safe_log(atr_usd)
    fl = _safe_log(float_m)
    si = math.log(max(1e-6, 1.0 + (si_pct or 0.0)/100.0))
    rv = _safe_log(1.0 + (rvol or 0.0))
    pm = _safe_log(pm_vol_m + 1.0)
    fr = math.log(max(1e-6, pm_vol_m/max(1e-6, float_m)))
    pmmc = _safe_log((pm_dol_m or 0.0) / max(1e-6, mc_m) + 1.0)
    ca = float(catalyst or 0.0)
    ln_pred_day = _safe_log(pred_vol_m)

    x = np.array([mc,gp,at,fl,si,rv,pm,fr,pmmc,ca,ln_pred_day], dtype=float)

    if coef is None:
        p_cls = 0.50
    else:
        p_cls = float(logit_inv(bias + x @ coef))

    # Optional blend with MaxPush model (small weight)
    mpcoef = A.get("mpcoef"); mpbias = float(A.get("mpbias") or 0.0)
    mpscale = float(A.get("mpscale") or 6.0)
    mp_med = float(A.get("mp_med_ft1") or 12.0)
    p_blend = p_cls
    if mpcoef is not None:
        z_mp = float(np.dot([mc,gp,at,fr,ca], mpcoef) + mpbias)
        mp_pct = float(1.0/(1.0+math.exp(-z_mp)))*100.0
        p_push = 1.0/(1.0 + math.exp(-(mp_pct - mp_med)/max(2.0, mpscale*8.0)))
        p_blend = 0.80*p_cls + 0.20*p_push

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
            notes = st.text_input("Notes (optional)", "")
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        cat = 1.0 if catalyst_flag=="Yes" else 0.0
        # PM% → Predicted Day Volume (with CI; PredVol >= PM)
        p_pm, p_lo, p_hi = _pm_pct_predict(mc_m, gap_pct, atr_usd, float_m, rvol, cat)
        pred_vol_m, ci68_l, ci68_u = predict_day_volume_from_pm(pm_vol_m, p_pm, p_lo, p_hi)

        # FT probability
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
            "PM%_pred": round(p_pm*100.0, 1),
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
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · PM% (pred): {l.get('PM%_pred',0):.1f}%"
        )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM%_pred","Notes"]
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
                "PM%_pred": st.column_config.NumberColumn("Predicted PM% of Day", format="%.1f"),
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
