# app.py — Premarket Ranking (Trained DayVol + FT; A/Ap/App fixed; Odds aligned)
# ------------------------------------------------------------------------------
# • DayVol predictor is TRAINED on your workbook (ridge on ln(Daily Vol M)), with learned σ.
#   Fallback to your fixed direct model if training is unavailable. PredVol ≥ PM Vol clamp.
# • FT classifier uses NO pmfrac; features:
#   ln1p_pmvol, ln_gapf, catalyst, ln_atr, ln_mcap, ln1p_rvol, ln1p_pmmc, ln1p_si.
# • Grades: A≥0.90, A+≥0.97, A++≥0.99 (fixed). Below 0.90 → B/C/D via exponential warp of train preds.
# • Odds: Very High≥0.90, High≥0.75, Moderate≥0.60, Low≥0.45, else Very Low.
# • UI: Ranking + Markdown + delete rows + clear + downloads.

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
st.set_page_config(page_title="Premarket Ranking — Trained DayVol + FT", layout="wide")
st.title("Premarket Ranking — Trained DayVol + FT")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Session ==============================
if "ART" not in st.session_state: st.session_state.ART = {}   # FT artifacts
if "DVP" not in st.session_state: st.session_state.DVP = {}   # DayVol predictor artifacts
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

# ============================== Helpers ==============================
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
    except: return float(fallback)

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

# ============================== Legend-driven mapping ==============================
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
    "DAILY":  ["Daily Vol (M)","Day Volume (M)","Volume (M)"],   # target for DayVol train
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

# ============================== Fixed direct DayVol (fallback) ==============================
def predict_dayvol_m_fixed(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Returns **millions of shares**.
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    y = math.exp(ln_y)
    return float(max(0.0, y))

# ============================== Ridge & Logistic ==============================
def ridge_fit(Z: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Ridge regression with intercept on standardized Z (already z-scored).
    Returns (coef_z, intercept).
    """
    n, k = Z.shape
    # add intercept column
    Xb = np.concatenate([np.ones((n,1)), Z], axis=1)
    I = np.eye(k+1); I[0,0] = 0.0  # no penalty on intercept
    w = np.linalg.solve(Xb.T @ Xb + l2*I, Xb.T @ y)
    return w[1:].astype(float), float(w[0])

def logit_fit_weighted(Z: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                       l2: float = 1.0, max_iter: int = 120, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    n, k = Z.shape
    Xb = np.concatenate([np.ones((n,1)), Z], axis=1)
    w = np.zeros(k+1)
    R = np.eye(k+1); R[0,0] = 0.0; R *= l2
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0/(1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p*(1-p)*sample_w
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H = Xb.T @ WX + R
        g = Xb.T @ ((y - p) * sample_w)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w = w + delta
        if np.linalg.norm(delta) < tol: break
    return w[1:].astype(float), float(w[0])

def logit_inv(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0/(1.0 + np.exp(-z))

# ============================== Cuts & Labels ==============================
def _inv_warp_p(sw: float, alpha: float = 2.6) -> float:
    sw = float(np.clip(sw, 0.0, 1.0))
    return float(1.0 - (1.0 - sw) ** (1.0 / max(1e-9, alpha)))

def make_grade_cuts() -> dict:
    """
    Fixed A-zone thresholds:
      • A   ≥ 0.90
      • A+  ≥ 0.97
      • A++ ≥ 0.99
    Below 0.90 → B/C/D via exponential warp of training predictions.
    """
    A_cut, Ap_cut, App_cut = 0.90, 0.97, 0.99

    p_cal = st.session_state.get("TRAIN_PREDS", np.array([]))
    if p_cal.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.75, "C": 0.55}

    sub = np.asarray(p_cal[(p_cal < A_cut) & np.isfinite(p_cal)])
    if sub.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.80, "C": 0.60}

    sub.sort()
    alpha = 2.6
    b_pos = _inv_warp_p(0.80, alpha)  # B/C boundary within sub-90 pool
    c_pos = _inv_warp_p(0.50, alpha)  # C/D boundary
    B_cut = float(np.quantile(sub, b_pos))
    C_cut = float(np.quantile(sub, c_pos))

    B_cut = min(B_cut, 0.89)
    if not np.isfinite(B_cut): B_cut = 0.75
    if not np.isfinite(C_cut): C_cut = 0.55
    if C_cut >= B_cut: C_cut = max(0.10, B_cut - 0.02)

    return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": B_cut, "C": C_cut}

def make_odds_cuts() -> dict:
    return {"very_high": 0.90, "high": 0.75, "moderate": 0.60, "low": 0.45}

def _prob_to_odds(p: float, cuts: Dict[str,float]) -> str:
    if p >= cuts.get("very_high",0.90): return "Very High Odds"
    if p >= cuts.get("high",0.75):      return "High Odds"
    if p >= cuts.get("moderate",0.60):  return "Moderate Odds"
    if p >= cuts.get("low",0.45):       return "Low Odds"
    return "Very Low Odds"

def _prob_to_grade(p: float, cuts: Dict[str,float]) -> str:
    if p >= cuts.get("App",0.99): return "A++"   # Very High Odds
    if p >= cuts.get("Ap",0.97):  return "A+"    # Very High Odds
    if p >= cuts.get("A",0.90):   return "A"     # High Odds
    if p >= cuts.get("B",0.75):   return "B"     # Moderate Odds
    if p >= cuts.get("C",0.55):   return "C"     # Low Odds
    return "D"                                   # Very Low Odds

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn from uploaded sheet", use_container_width=True)

def _learn(xls: pd.ExcelFile, sheet: str) -> None:
    raw = pd.read_excel(xls, sheet)

    # map columns
    col = {}
    for k, cands in _DEF.items():
        p = _pick(raw, cands)
        if p: col[k] = p
    if "FT" not in col:
        st.error("No 'FT' column found.")
        return

    df = pd.DataFrame()
    df["FT"] = (pd.to_numeric(raw[col["FT"]], errors="coerce").fillna(0.0) >= 0.5).astype(float)

    def _add(name, key):
        if key in col:
            df[name] = pd.to_numeric(raw[col[key]], errors="coerce")

    _add("gap_pct","GAP")
    _add("atr_usd","ATR")
    _add("rvol","RVOL")
    _add("pm_vol_m","PMVOL")
    _add("pm_dol_m","PM$")
    _add("float_m","FLOAT")
    _add("mcap_m","MCAP")
    _add("si_pct","SI")
    _add("daily_vol_m","DAILY")
    if "CAT" in col: df["catalyst"] = _parse_catalyst_col(raw[col["CAT"]])
    else: df["catalyst"] = 0.0

    # ---------------- DayVol TRAIN (ridge on ln(Daily Vol M)) ----------------
    dv_feats: List[Tuple[str, callable]] = []
    # features: ln_mcap, ln_gapf, ln_atr, catalyst, ln1p_rvol, ln_float (use what exists)
    if "mcap_m" in df.columns:    dv_feats.append(("ln_mcap",   lambda r: _safe_log(r.get("mcap_m"))))
    if "gap_pct" in df.columns:   dv_feats.append(("ln_gapf",   lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0)))
    if "atr_usd" in df.columns:   dv_feats.append(("ln_atr",    lambda r: _safe_log(r.get("atr_usd"))))
    if "catalyst" in df.columns:  dv_feats.append(("catalyst",  lambda r: float(_nz(r.get("catalyst"),0.0))))
    if "rvol" in df.columns:      dv_feats.append(("ln1p_rvol", lambda r: _safe_log(_nz(r.get("rvol"),0.0) + 1.0)))
    if "float_m" in df.columns:   dv_feats.append(("ln_float",  lambda r: _safe_log(r.get("float_m"))))

    DVP = {"trained": False, "sigma_ln": 0.60}  # default sigma
    if "daily_vol_m" in df.columns and df["daily_vol_m"].notna().sum() >= 20 and len(dv_feats) >= 3:
        y_ln = np.log(np.clip(df["daily_vol_m"].to_numpy(float), 1e-6, None))
        X = np.vstack([[f(df.iloc[i].to_dict()) for _, f in dv_feats] for i in range(len(df))]).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1); sd[sd==0] = 1.0
        Z = (X - mu) / sd; Z = np.clip(Z, -4.0, 4.0)

        coef_z, intercept = ridge_fit(Z, y_ln, l2=1.0)
        yhat_ln = intercept + Z @ coef_z
        resid = y_ln - yhat_ln
        sigma_ln = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.60

        DVP = {
            "trained": True,
            "feat_names": [n for n,_ in dv_feats],
            "mu": mu, "sd": sd,
            "coef_z": coef_z, "intercept": float(intercept),
            "sigma_ln": sigma_ln
        }
        st.success(f"DayVol model trained (n={len(y_ln)}; σ_ln≈{sigma_ln:.2f}; feats: {', '.join(DVP['feat_names'])}).")
    else:
        st.info("DayVol training unavailable — will use fixed direct model with generic σ_ln=0.60.")

    # ---------------- FT TRAIN (no pmfrac) ----------------
    ft_feats: List[Tuple[str, callable]] = [
        ("ln1p_pmvol",  lambda r: _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)),
        ("ln_gapf",     lambda r: _safe_log(_nz(r.get("gap_pct"),0.0) / 100.0)),
        ("catalyst",    lambda r: float(_nz(r.get("catalyst"),0.0))),
        ("ln_atr",      lambda r: _safe_log(r.get("atr_usd"))),
        ("ln_mcap",     lambda r: _safe_log(r.get("mcap_m"))),
        ("ln1p_rvol",   lambda r: _safe_log(_nz(r.get("rvol"),0.0) + 1.0)),
        ("ln1p_pmmc",   lambda r: _safe_log(_nz(r.get("pm_dol_m"),0.0) / max(1e-6, _nz(r.get("mcap_m"),0.0)) + 1.0)),
        ("ln1p_si",     lambda r: _safe_log(_nz(r.get("si_pct"),0.0) + 1.0)),
    ]
    Xft = np.vstack([[f(df.iloc[i].to_dict()) for _, f in ft_feats] for i in range(len(df))]).astype(float)
    Xft = np.nan_to_num(Xft, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["FT"].to_numpy(dtype=float)

    mu = Xft.mean(axis=0); sd = Xft.std(axis=0, ddof=1); sd[sd==0] = 1.0
    Z = (Xft - mu) / sd; Z = np.clip(Z, -3.0, 3.0)

    # balance weights
    p1 = float(np.mean(y)) if y.size else 0.5
    w1 = 0.5 / max(1e-9, p1)
    w0 = 0.5 / max(1e-9, 1.0 - p1)
    sample_w = np.where(y>0.5, w1, w0).astype(float)

    if Z.shape[0] >= 12 and np.unique(y).size == 2:
        coef_z, bias = logit_fit_weighted(Z, y, sample_w, l2=1.0, max_iter=140, tol=1e-6)
        st.success("FT model trained (n={}; base FT≈{:.2f}).".format(Z.shape[0], p1))
    else:
        coef_z, bias = np.zeros(Z.shape[1]), 0.0
        st.error("Unable to train FT (need ≥12 rows and both classes). Using neutral 50%.")

    p_cal = logit_inv(bias + Z @ coef_z)
    st.session_state.TRAIN_PREDS = p_cal
    st.session_state.GRADE_CUTS = make_grade_cuts()
    st.session_state.ODDS_CUTS  = make_odds_cuts()

    st.session_state.ART = {
        "feat_names":[n for n,_ in ft_feats], "mu":mu, "sd":sd, "coef_z":coef_z, "bias":bias
    }
    st.session_state.DVP = DVP
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
                _learn(xls, sheet_name)
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Inference ==============================
def predict_dayvol_trained(mcap_m: float, gap_pct: float, atr_usd: float,
                           rvol: float, float_m: float, catalyst: float) -> Tuple[float,float,float,bool]:
    """
    Returns (pred_M, CI68_low, CI68_high, used_trained_model)
    Uses trained ridge model if available; otherwise falls back to fixed direct model with σ=0.60.
    """
    DVP = st.session_state.get("DVP", {}) or {}
    used_trained = False

    if DVP.get("trained"):
        names = DVP.get("feat_names", [])
        mu = DVP.get("mu"); sd = DVP.get("sd")
        coef = DVP.get("coef_z"); intercept = float(DVP.get("intercept", 0.0))
        sigma_ln = float(DVP.get("sigma_ln", 0.60))
        if names and mu is not None and sd is not None and coef is not None:
            fmap = {
                "ln_mcap":   lambda: _safe_log(mcap_m),
                "ln_gapf":   lambda: _safe_log(_nz(gap_pct,0.0) / 100.0),
                "ln_atr":    lambda: _safe_log(atr_usd),
                "catalyst":  lambda: float(_nz(catalyst,0.0)),
                "ln1p_rvol": lambda: _safe_log(_nz(rvol,0.0) + 1.0),
                "ln_float":  lambda: _safe_log(float_m),
            }
            vals = [float(fmap[n]()) for n in names]
            Z = (np.array(vals) - mu) / sd
            Z = np.clip(Z, -4.0, 4.0)
            ln_pred = float(intercept + np.dot(Z, coef))
            pred = float(max(0.0, math.exp(ln_pred)))
            lo = float(pred * math.exp(-1.0 * sigma_ln))
            hi = float(pred * math.exp(+1.0 * sigma_ln))
            used_trained = True
            return pred, lo, hi, used_trained

    # Fallback to fixed model
    pred = predict_dayvol_m_fixed(mcap_m, gap_pct, atr_usd)
    sigma_ln = 0.60
    lo = float(pred * math.exp(-1.0 * sigma_ln))
    hi = float(pred * math.exp(+1.0 * sigma_ln))
    return pred, lo, hi, used_trained

def predict_ft_prob(gap_pct: float, pm_vol_m: float, catalyst: float,
                    atr_usd: float, mcap_m: float, rvol: float, pm_dol_m: float,
                    si_pct: float) -> float:
    A = st.session_state.ART or {}
    names = A.get("feat_names") or []; mu = A.get("mu"); sd = A.get("sd")
    coef = A.get("coef_z"); bias = float(A.get("bias", 0.0))
    if not names or coef is None or mu is None or sd is None: return 0.50

    feat_map = {
        "ln1p_pmvol":  lambda: _safe_log(_nz(pm_vol_m,0.0) + 1.0),
        "ln_gapf":     lambda: _safe_log(_nz(gap_pct,0.0)/100.0),
        "catalyst":    lambda: float(_nz(catalyst,0.0)),
        "ln_atr":      lambda: _safe_log(atr_usd),
        "ln_mcap":     lambda: _safe_log(mcap_m),
        "ln1p_rvol":   lambda: _safe_log(_nz(rvol,0.0) + 1.0),
        "ln1p_pmmc":   lambda: _safe_log(_nz(pm_dol_m,0.0)/max(1e-6,_nz(mcap_m,0.0)) + 1.0),
        "ln1p_si":     lambda: _safe_log(_nz(si_pct,0.0) + 1.0),
    }
    vals = [float(feat_map[n]()) for n in names]
    Z = (np.array(vals) - mu) / sd
    Z = np.clip(Z, -3.0, 3.0)
    return float(np.clip(logit_inv(bias + np.dot(Z, coef)), 1e-3, 1-1e-3))

def _odds_and_grade(p: float) -> Tuple[str, str]:
    odds_cuts  = st.session_state.get("ODDS_CUTS", make_odds_cuts())
    grade_cuts = st.session_state.get("GRADE_CUTS", make_grade_cuts())
    return _prob_to_odds(p, odds_cuts), _prob_to_grade(p, grade_cuts)

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add / Score", "📊 Ranking"])

# ============================== Add / Score ==============================
with tab_add:
    # Training status badges
    _ft_ok  = bool(st.session_state.ART)
    _dvp_ok = bool(st.session_state.DVP.get("trained", False))
    st.markdown(
        f'<div style="margin:.25rem 0 1rem 0;">'
        f'  <span style="display:inline-block;padding:.18rem .55rem;border-radius:999px;'
        f'         border:1px solid {_ft_ok and "#16a34a" or "#ef4444"};'
        f'         background:{_ft_ok and "#ecfdf5" or "#fef2f2"};'
        f'         color:{_ft_ok and "#166534" or "#991b1b"}; font-weight:600; font-size:.82rem; margin-right:.4rem;">'
        f'    FT model: {_ft_ok and "trained" or "NOT trained"}'
        f'  </span>'
        f'  <span style="display:inline-block;padding:.18rem .55rem;border-radius:999px;'
        f'         border:1px solid {_dvp_ok and "#16a34a" or "#94a3b8"};'
        f'         background:{_dvp_ok and "#ecfdf5" or "#f1f5f9"};'
        f'         color:{_dvp_ok and "#166534" or "#334155"}; font-weight:600; font-size:.82rem;">'
        f'    DayVol model: {_dvp_ok and "trained" or "fixed fallback"}'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2,1.2,1.0])
        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %",                    0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"], index=0)
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        if not _ft_ok:
            st.error("Train the FT model first (Upload → Learn).")
        else:
            cat = 1.0 if catalyst_flag=="Yes" else 0.0

            # DayVol (trained if available; else fixed); clamp to PM
            pred_vol_m, ci68_l, ci68_u, used_trained = predict_dayvol_trained(
                mcap_m=mc_m, gap_pct=gap_pct, atr_usd=atr_usd,
                rvol=rvol, float_m=float_m, catalyst=cat
            )
            pred_vol_m = max(pred_vol_m, _nz(pm_vol_m, 0.0))
            # when clamped, CI can be below pred; that's fine for display—keep as model CI

            # FT probability
            ft_prob = predict_ft_prob(
                gap_pct=gap_pct, pm_vol_m=pm_vol_m, catalyst=cat,
                atr_usd=atr_usd, mcap_m=mc_m, rvol=rvol, pm_dol_m=pm_dol_m,
                si_pct=si_pct
            )
            odds, grade = _odds_and_grade(ft_prob)

            row = {
                "Ticker": ticker,
                "Odds": odds,
                "Level": grade,
                "FinalScore": round(ft_prob*100.0, 2),

                "PredVol_M": round(pred_vol_m, 2),
                "PredVol_CI68_L": round(ci68_l, 2),
                "PredVol_CI68_U": round(ci68_u, 2),

                "PM%_of_Pred": round(100.0 * _nz(pm_vol_m,0.0) / max(1e-6, pred_vol_m), 1),
                "PM$ / MC_%": round(100.0 * _nz(pm_dol_m,0.0) / max(1e-6, _nz(mc_m,0.0)), 1),
                "DayVolModel": ("Trained" if used_trained else "Fixed"),

                # raw inputs for CSV/debug
                "_MCap_M": mc_m, "_Gap_%": gap_pct, "_ATR_$": atr_usd, "_PM_M": pm_vol_m,
                "_Float_M": float_m, "_SI_%": si_pct, "_RVOL": rvol, "_PM$_M": pm_dol_m, "_Catalyst": cat,
            }
            st.session_state.rows.append(row)
            st.session_state.last = row
            st.session_state.flash = f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})"
            _rerun()

    # Preview card
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
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"PM % of Pred: {l.get('PM%_of_Pred',0):.1f}% · PM $/MC: {l.get('PM$ / MC_%',0):.1f}% · "
            f"DayVol model: {l.get('DayVolModel','—')}"
        )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols = ["Ticker","Odds","Level","FinalScore",
                "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM%_of_Pred","PM$ / MC_%","DayVolModel"]
        for c in cols:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level","DayVolModel") else 0.0

        st.dataframe(
            df[cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("FT Probability %", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("PredVol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("PredVol CI68 High (M)", format="%.2f"),
                "PM%_of_Pred": st.column_config.NumberColumn("PM % of Pred", format="%.1f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "DayVolModel": st.column_config.TextColumn("DayVol Model"),
            }
        )

        # Delete rows (top 12)
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

        # Downloads
        st.download_button(
            "Download CSV (Ranking)",
            df[cols].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        # Markdown table
        st.markdown("### 📋 Ranking (Markdown view)")
        md_text = df_to_markdown_table(df, cols)
        st.code(md_text, language="markdown")
        st.download_button(
            "Download Markdown",
            md_text.encode("utf-8"),
            "ranking.md", "text/markdown", use_container_width=True
        )

        c1, _ = st.columns([0.25,0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                _rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add / Score** tab.")
