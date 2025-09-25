# app.py — Premarket Stock Ranking — Direct Log Model (with robust catalyst + unit diagnostics)
# ---------------------------------------------------------------------------------------------
# • Day Volume prediction uses ONLY your direct log-linear model (millions), clamped: PredVol ≥ PM Vol.
# • FT classifier learned from the workbook; features include ln(PredDay) + context. Optional non-neg on ln1p_pmvol.
# • Robust catalyst parsing (Yes/No/True/False/1/0/"pr"/"news").
# • Dataset unit diagnostics right after learning (quick sanity flags).
# • UI: Ranking + Markdown (both downloadable), delete rows (top 12), Clear. No local paths, no notes.

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
st.set_page_config(page_title="Premarket Stock Ranking — Direct Log Model", layout="wide")
st.title("Premarket Stock Ranking — Direct Log Model")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Sidebar (confidence only) ==============================
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (CI68)", 0.10, 1.50, 0.60, 0.01,
                             help="Std dev of residuals in ln(DayVol). 0.60 is a sensible default.")

# Optional: do not allow negative weight on ln1p_pmvol in FT (prevents 'more PM flow -> lower FT').
ENFORCE_NONNEG_FLOW = st.sidebar.checkbox("FT: enforce non-negative coef on ln1p_pmvol", True)

# ============================== Session State ==============================
if "ARTIFACTS" not in st.session_state: st.session_state.ARTIFACTS = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ============================== Helpers ==============================
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip().replace(" ", "").replace("’","").replace("'","")
    if s == "": return None
    if "," in s and "." not in s: s = s.replace(",", ".")
    else: s = s.replace(",", "")
    try: return float(s)
    except Exception: return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"; s = st.text_input(label, fmt.format(value), key=key, help=help)
    v = _parse_local_float(s)
    if v is None: return float(value)
    v = max(min_value, v); 
    if max_value is not None: v = min(max_value, v)
    return float(v)

def _nz(x, fallback=0.0):
    try:
        xx = float(x)
        return xx if np.isfinite(xx) else float(fallback)
    except Exception:
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

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ============================== Legend-driven column mapping ==============================
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
    "DAILY": ["Daily Vol (M)", "Day Volume (M)", "Volume (M)"],          # optional (diagnostics)
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

# ============================== Catalyst parsing (robust) ==============================
def _parse_catalyst_col(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    positives = {"1","true","yes","y","t"}
    negatives = {"0","false","no","n","f",""}
    out = []
    for v in s.fillna(""):
        if v in positives: out.append(1.0)
        elif v in negatives: out.append(0.0)
        elif "pr" in v or "news" in v or "catalyst" in v: out.append(1.0)
        else:
            try: out.append(float(v))
            except: out.append(0.0)
    return pd.Series(out, dtype=float).clip(0,1)

# ============================== Direct log-linear DayVol (ONLY) ==============================
def predict_day_volume_m_direct(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Returns Y in **millions of shares**.
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    y = math.exp(ln_y)
    return float(max(0.0, y))

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float) -> Tuple[float,float]:
    if pred_m <= 0: return 0.0, 0.0
    return float(pred_m * math.exp(-z*sigma_ln)), float(pred_m * math.exp(z*sigma_ln))

# ============================== Logistic with L2 + class balance (+ optional nonneg) ==============================
def logit_fit_weighted(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                       l2: float = 1.0, max_iter: int = 100, tol: float = 1e-6,
                       nonneg_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
    """IRLS with L2 and sample weights; optionally projects selected coef to ≥0 each step."""
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
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
        if nonneg_idx:
            for j in nonneg_idx:
                jj = j + 1  # skip intercept
                if w[jj] < 0: w[jj] = 0.0
        if np.linalg.norm(delta) < tol: break
    return w[1:].astype(float), float(w[0])

def logit_inv(z: float) -> float:
    z = float(np.clip(z, -35, 35))
    return 1.0/(1.0 + math.exp(-z))

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn from uploaded sheet", use_container_width=True)

def _load_and_learn(xls: pd.ExcelFile, sheet: str) -> None:
    raw = pd.read_excel(xls, sheet)

    # --- column mapping ---
    col = {}
    for key, cands in _DEF.items():
        pick = _pick(raw, cands)
        if pick: col[key] = pick
    if "FT" not in col:
        st.error("No 'FT' column found in sheet.")
        return

    df = pd.DataFrame()
    ft_series = pd.to_numeric(raw[col["FT"]], errors="coerce")
    df["FT"] = (ft_series.fillna(0.0) >= 0.5).astype(float)

    def _add(colname: str, key: str): 
        if key in col: df[colname] = pd.to_numeric(raw[col[key]], errors="coerce")

    _add("gap_pct", "GAP"); _add("atr_usd", "ATR"); _add("rvol", "RVOL")
    _add("pm_vol_m", "PMVOL"); _add("pm_dol_m", "PM$")
    _add("float_m", "FLOAT"); _add("mcap_m", "MCAP"); _add("si_pct", "SI")
    _add("daily_vol_m", "DAILY")

    if "CAT" in col: df["catalyst"] = _parse_catalyst_col(raw[col["CAT"]])
    else: df["catalyst"] = 0.0

    if "MAXPCT" in col:
        df["max_push_pct"] = pd.to_numeric(raw[col["MAXPCT"]], errors="coerce")

    # Derived diagnostics
    if {"pm_vol_m","float_m"}.issubset(df.columns):
        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

    # ---------- Dataset unit diagnostics ----------
    with st.expander("Dataset checks (medians & sanity flags)"):
        med = {c: float(np.nanmedian(df[c])) for c in ["mcap_m","float_m","pm_vol_m","gap_pct","atr_usd"] if c in df.columns}
        st.write("Medians (from uploaded):", med)
        flags = []
        if med.get("mcap_m", 0) > 50000: flags.append("⚠️ Median MCap > $50B — ensure units are **millions**.")
        if med.get("float_m", 0) > 10000: flags.append("⚠️ Median Float > 10,000M — check units.")
        if med.get("pm_vol_m", 0) > 1000: flags.append("⚠️ Median PM Vol > 1,000M — check units.")
        if med.get("gap_pct", 0) < 1 and med.get("gap_pct", 0) > 0: flags.append("⚠️ Gap % looks like a fraction (e.g., 0.23). Enter as percent (23).")
        if med.get("atr_usd", 0) > 20: flags.append("⚠️ ATR > $20 — double-check source/units.")
        if flags: st.warning("\n".join(flags))
        else: st.info("No obvious unit issues detected.")

    # ---------- Build ln(PredDay) using DIRECT model (clamp to PMVol) ----------
    pred_day_direct = []
    for i in range(len(df)):
        r = df.iloc[i]
        pred_m = predict_day_volume_m_direct(_nz(r.get("mcap_m")), _nz(r.get("gap_pct")), _nz(r.get("atr_usd")))
        pm_m   = _nz(r.get("pm_vol_m"), 0.0)
        pred_day_direct.append(max(pm_m, pred_m))
    df["pred_day_m"] = np.array(pred_day_direct, dtype=float)

    # ===== Train FT classifier =====
    def _ln1p_pmvol(r): return _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)
    def _ln_fr(r):
        pm = _nz(r.get("pm_vol_m"),0.0); fl = _nz(r.get("float_m"),0.0)
        return math.log(max(1e-6, pm / max(1e-6, fl)))
    def _ln1p_pmmc(r):
        pm_dol = _nz(r.get("pm_dol_m"),0.0); mc = _nz(r.get("mcap_m"),0.0)
        return _safe_log(pm_dol / max(1e-6, mc) + 1.0)
    def _scaled_maxpush(r):
        v = _nz(r.get("max_push_pct"), np.nan)
        return (v/100.0) if np.isfinite(v) else 0.0

    ft_candidates = [
        ("ln_mcap",      lambda r: _safe_log(r.get("mcap_m")),                         ["mcap_m"]),
        ("ln_gapf",      lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0),         ["gap_pct"]),
        ("ln_atr",       lambda r: _safe_log(r.get("atr_usd")),                        ["atr_usd"]),
        ("ln_float",     lambda r: _safe_log(r.get("float_m")),                        ["float_m"]),
        ("ln1p_rvol",    lambda r: _safe_log(1.0 + _nz(r.get("rvol"),0.0)),            ["rvol"]),
        ("ln1p_pmvol",   _ln1p_pmvol,                                                  ["pm_vol_m"]),
        ("ln_fr",        _ln_fr,                                                       ["pm_vol_m","float_m"]),
        ("ln1p_pmmc",    _ln1p_pmmc,                                                   ["pm_dol_m","mcap_m"]),
        ("catalyst",     lambda r: float(_nz(r.get("catalyst"),0.0)),                  ["catalyst"]),
        ("ln_pred_day",  lambda r: _safe_log(r.get("pred_day_m")),                     ["pred_day_m"]),
        ("maxpush_s",    _scaled_maxpush,                                              []),  # optional, 0 if missing
    ]
    ft_use = [(n,f,req) for (n,f,req) in ft_candidates if all(k in df.columns for k in req) or n=="maxpush_s"]

    if not ft_use:
        st.error("No usable FT features found.")
        return

    Xft = np.vstack([[f(df.iloc[i].to_dict()) for (n,f,_) in ft_use] for i in range(len(df))]).astype(float)
    Xft = np.nan_to_num(Xft, nan=0.0, posinf=0.0, neginf=0.0)
    y_ft = (np.nan_to_num(df["FT"].to_numpy(dtype=float), nan=0.0) >= 0.5).astype(float)
    feat_names = [n for (n,_,_) in ft_use]

    # Standardize + clip (stabilize fit)
    mu = Xft.mean(axis=0)
    sd = Xft.std(axis=0, ddof=1); sd[sd == 0] = 1.0
    Z = (Xft - mu) / sd
    Z = np.clip(Z, -3.0, 3.0)

    # Class weights (balance prevalence)
    p1 = float(np.mean(y_ft)) if y_ft.size else 0.5
    w1 = 0.5 / max(1e-9, p1); w0 = 0.5 / max(1e-9, 1.0 - p1)
    sample_w = np.where(y_ft > 0.5, w1, w0).astype(float)

    # Optional non-negative constraint on ln1p_pmvol
    nonneg_idx = []
    if ENFORCE_NONNEG_FLOW and "ln1p_pmvol" in feat_names:
        nonneg_idx.append(feat_names.index("ln1p_pmvol"))

    ft_coef_z, ft_bias = logit_fit_weighted(Z, y_ft, sample_w, l2=1.0, max_iter=140, tol=1e-6, nonneg_idx=nonneg_idx)

    # Save artifacts
    st.session_state.ARTIFACTS = {
        "feat_names": feat_names, "mu": mu, "sd": sd,
        "ft_coef_z": ft_coef_z, "ft_bias": ft_bias,
    }

    # Calibration thresholds from train preds (with floors)
    p_cal = 1.0/(1.0 + np.exp(-(ft_bias + Z @ ft_coef_z)))
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
    st.session_state.ODDS_CUTS = odds_cuts
    st.session_state.GRADE_CUTS = grade_cuts

    # Quick coefficients table
    coefs_tbl = pd.DataFrame({
        "feature": feat_names,
        "coef_z":  ft_coef_z
    }).sort_values("coef_z", ascending=False, key=np.abs)
    st.dataframe(coefs_tbl, use_container_width=True, hide_index=True)

    st.success(f"FT classifier trained (n={Z.shape[0]}; feats: {', '.join(feat_names)}; base FT≈{p1:.2f}).")
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

# ============================== Inference ==============================
def predict_ft_prob_direct(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float,
                           rvol: float, pm_vol_m: float, pm_dol_m: float, catalyst: float,
                           maxpush_pct_live: Optional[float] = None) -> float:
    A = st.session_state.ARTIFACTS or {}
    names = A.get("feat_names") or []; mu = A.get("mu"); sd = A.get("sd")
    coef_z = A.get("ft_coef_z"); bias = float(A.get("ft_bias") or 0.0)
    if not names or coef_z is None or mu is None or sd is None: return 0.50

    # Predicted Day Vol from DIRECT model (clamped to PMVol)
    pred_day_m = max(_nz(pm_vol_m,0.0), predict_day_volume_m_direct(mc_m, gap_pct, atr_usd))

    def _ln1p_pmvol(): return _safe_log(_nz(pm_vol_m,0.0) + 1.0)
    def _ln_fr():
        pm = _nz(pm_vol_m,0.0); fl = _nz(float_m,0.0)
        return math.log(max(1e-6, pm / max(1e-6, fl)))
    def _ln1p_pmmc():
        pm_dol = _nz(pm_dol_m,0.0); mc = _nz(mc_m,0.0)
        return _safe_log(pm_dol / max(1e-6, mc) + 1.0)
    def _maxpush_s():
        v = _nz(maxpush_pct_live, np.nan)
        return (v/100.0) if np.isfinite(v) else 0.0

    feat_map = {
        "ln_mcap":     lambda: _safe_log(mc_m),
        "ln_gapf":     lambda: _safe_log(_nz(gap_pct,0.0)/100.0),
        "ln_atr":      lambda: _safe_log(atr_usd),
        "ln_float":    lambda: _safe_log(float_m),
        "ln1p_rvol":   lambda: _safe_log(1.0 + _nz(rvol,0.0)),
        "ln1p_pmvol":  _ln1p_pmvol,
        "ln_fr":       _ln_fr,
        "ln1p_pmmc":   _ln1p_pmmc,
        "catalyst":    lambda: float(_nz(catalyst,0.0)),
        "ln_pred_day": lambda: _safe_log(pred_day_m),
        "maxpush_s":   _maxpush_s,
    }
    vals_raw = [float(feat_map[n]()) for n in names]
    Z = (np.array(vals_raw) - mu) / sd
    Z = np.clip(Z, -3.0, 3.0)

    return float(np.clip(1.0/(1.0 + np.exp(-(bias + np.dot(Z, coef_z)))), 1e-3, 1-1e-3))

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

def _odds_and_grade(p: float) -> Tuple[str,str]:
    return _prob_to_odds(p, st.session_state.get("ODDS_CUTS", {})), _prob_to_grade(p, st.session_state.get("GRADE_CUTS", {}))

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
            gap_pct  = input_float("Gap %",                     0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"], index=0)
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        cat = 1.0 if catalyst_flag=="Yes" else 0.0

        # Direct model → Predicted Day Volume (millions), clamp to PM
        pred_vol_m = predict_day_volume_m_direct(mc_m, gap_pct, atr_usd)
        pred_vol_m = max(pred_vol_m, _nz(pm_vol_m, 0.0))

        # Confidence bands (68%)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)

        # FT probability
        ft_prob = predict_ft_prob_direct(mc_m, gap_pct, atr_usd, float_m, si_pct, rvol, pm_vol_m, pm_dol_m, cat)
        odds_name, level = _odds_and_grade(ft_prob)

        row = {
            "Ticker": ticker,
            "Odds": odds_name,
            "Level": level,
            "FinalScore": round(ft_prob*100.0, 2),

            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),

            # Display helpers
            "PM%_of_Pred": round(100.0 * _nz(pm_vol_m,0.0) / max(1e-6, pred_vol_m), 1),
            "PM$ / MC_%": round(100.0 * _nz(pm_dol_m,0.0) / max(1e-6, _nz(mc_m,0.0)), 1),

            # raw inputs for CSV (kept hidden in UI)
            "_MCap_M": mc_m, "_Gap_%": gap_pct, "_ATR_$": atr_usd, "_PM_M": pm_vol_m,
            "_Float_M": float_m, "_SI_%": si_pct, "_RVOL": rvol, "_PM$_M": pm_dol_m, "_Catalyst": cat,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

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
            f"PM % of Pred: {l.get('PM%_of_Pred',0):.1f}% · PM $/MC: {l.get('PM$ / MC_%',0):.1f}%"
        )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "FinalScore",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM%_of_Pred","PM$ / MC_%"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level") else 0.0

        st.dataframe(
            df[cols_to_show],
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
            }
        )

        # Delete rows (top 12 quick buttons)
        st.markdown("#### Delete rows")
        del_cols = st.columns(4)
        head12 = df.head(12).reset_index(drop=True)
        for i, r in head12.iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    do_rerun()

        # Download CSV (Ranking view)
        st.download_button(
            "Download CSV (Ranking)",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        # Markdown table (below ranking) + download + clear
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
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
