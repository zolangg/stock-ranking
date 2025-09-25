# app.py — Premarket Ranking (Direct Log Vol + FT WITHOUT pmfrac & fr)
# -------------------------------------------------------------------
# • Day Volume: your direct log-linear model (millions), clamped to ≥ 0; CI via sidebar σ.
# • FT Classifier: logistic (L2), class-balanced, standardized (z-scores), clipped ±3σ,
#   **NO pmfrac features** and **NO ln_fr**. Flow signal = ln1p_pmvol only.
#   Features (if available): ln_gapf, ln_mcap, ln_atr, ln_float, ln1p_rvol, ln1p_pmvol, catalyst, optional maxpush_s.
# • Intercept calibration to match training prevalence.
# • UI: Add ➜ Ranking (+ Markdown view), Download, Delete rows (top 12), Clear.
# • Catalyst yes/no supported. PredVol shown but NOT used by FT.

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
st.set_page_config(page_title="Premarket Stock Ranking — FT (no pmfrac / no fr)", layout="wide")
st.title("Premarket Stock Ranking — FT (no pmfrac / no fr)")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Sidebar ==============================
st.sidebar.header("Predicted Day Volume — CI (display only)")
sigma_ln = st.sidebar.slider("Log-space σ for PredVol CI68", 0.10, 1.50, 0.60, 0.01,
                             help="Std dev of residuals in ln(DayVol). 0.60 ≈ typical.")

st.sidebar.header("FT Flow Sensitivity")
flow_sens = st.sidebar.slider("Flow sensitivity (ln1p_pmvol)", 0.5, 1.5, 1.00, 0.05,
                              help="Scales the ln1p_pmvol feature after z-scoring.")
st.session_state["FLOW_SENS"] = float(flow_sens)

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
            if isinstance(v, float): cells.append(f"{v:.2f}")
            else: cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ============================== Legend-driven column mapping ==============================
_DEF = {
    "FT": ["FT"],
    "MAXPCT": ["Max Push Daily %", "Max Push Daily (%)", "Max Push %"],  # optional
    "GAP": ["Gap %", "Gap", "Premarket Gap"],
    "ATR": ["Daily ATR", "ATR $", "ATR", "ATR (USD)", "ATR$"],
    "RVOL": ["RVOL @ BO", "RVOL", "Relative Volume"],
    "PMVOL": ["PM Vol (M)", "Premarket Vol (M)", "PM Volume (M)", "PM Shares (M)"],
    "PM$": ["PM $Vol (M)", "PM Dollar Vol (M)", "PM $ Volume (M)", "PM $Vol"],  # optional
    "FLOAT": ["Float M Shares", "Public Float (M)", "Float (M)", "Float"],
    "MCAP": ["MarketCap M", "Market Cap (M)", "MCap M", "MCap"],
    "SI": ["Short Interest %", "Short Float %", "Short Interest (Float) %", "SI"],
    "DAILY": ["Daily Vol (M)", "Day Volume (M)", "Volume (M)"],  # optional (diagnostics only)
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

# ============================== Day Volume — Direct log-linear (display only) ==============================
def predict_day_volume_m_direct(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Returns Y in **millions of shares**
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
    low  = pred_m * math.exp(-z * sigma_ln)
    high = pred_m * math.exp( z * sigma_ln)
    return float(low), float(high)

# ============================== Logistic with L2 + class weights ==============================
def logit_fit_l2_weighted(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                          l2: float = 1.2, max_iter: int = 120, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Logistic regression via IRLS with L2 on weights (not intercept) and sample weights."""
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    w = np.zeros(k+1)
    R = np.eye(k+1); R[0,0] = 0.0; R *= float(l2)
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0/(1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p*(1-p) * sample_w
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H = Xb.T @ WX + R
        g = Xb.T @ ((y - p) * sample_w)
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

def _clip_z(z: np.ndarray, lim: float = 3.0) -> np.ndarray:
    return np.clip(z, -abs(lim), abs(lim))

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
    # Force-binary FT
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
    _add("daily_vol_m", "DAILY")  # optional (not used by FT)
    if "CAT" in col:
        df["catalyst"] = pd.to_numeric(raw[col["CAT"]], errors="coerce").clip(0,1).fillna(0.0)
    else:
        df["catalyst"] = 0.0
    if "MAXPCT" in col:
        df["max_push_pct"] = pd.to_numeric(raw[col["MAXPCT"]], errors="coerce")

    # ================= FT CLASSIFIER (NO pmfrac / NO fr) =================
    # Feature builders
    def _ln_gapf(r):    return _safe_log(_nz(r.get("gap_pct"),0.0)/100.0)
    def _ln_mcap(r):    return _safe_log(r.get("mcap_m"))
    def _ln_atr(r):     return _safe_log(r.get("atr_usd"))
    def _ln_float(r):   return _safe_log(r.get("float_m"))
    def _ln1p_rvol(r):  return _safe_log(1.0 + _nz(r.get("rvol"),0.0))
    def _ln1p_pmvol(r): return _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)
    def _catalyst(r):   return float(_nz(r.get("catalyst"),0.0))
    def _maxpush_s(r):
        v = _nz(r.get("max_push_pct"), np.nan)
        return (v/100.0) if np.isfinite(v) else 0.0

    # Select features (no ln_pmvol_f, no ln_fr)
    F_LIST = [
        ("ln_gapf", _ln_gapf, ["gap_pct"]),
        ("ln_mcap", _ln_mcap, ["mcap_m"]),
        ("ln_atr", _ln_atr, ["atr_usd"]),
        ("ln_float", _ln_float, ["float_m"]),
        ("ln1p_rvol", _ln1p_rvol, ["rvol"]),
        ("ln1p_pmvol", _ln1p_pmvol, ["pm_vol_m"]),  # the ONLY flow signal
        ("catalyst", _catalyst, ["catalyst"]),
        ("maxpush_s", _maxpush_s, []),              # optional
    ]
    use_feats = [(n,f,req) for (n,f,req) in F_LIST if all(k in df.columns for k in req) or n=="maxpush_s"]
    if not use_feats:
        st.error("No usable FT features found.")
        return

    # Design matrix
    X_raw = np.vstack([[f(df.iloc[i].to_dict()) for (n,f,_) in use_feats] for i in range(len(df))]).astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_ft = (np.nan_to_num(df["FT"].to_numpy(dtype=float), nan=0.0) >= 0.5).astype(float)
    feat_names = [n for (n,_,_) in use_feats]

    # Standardize
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0, ddof=1); sd[sd == 0] = 1.0
    X = (X_raw - mu) / sd

    # Clip z-scores
    X = _clip_z(X, 3.0)
    # Flow sensitivity on ln1p_pmvol only
    for j, name in enumerate(feat_names):
        if name == "ln1p_pmvol":
            X[:, j] = _clip_z(X[:, j], 2.5) * st.session_state.get("FLOW_SENS", flow_sens)

    # Class weights (prevalence balancing)
    p1 = float(np.mean(y_ft)) if y_ft.size else 0.5
    w1 = 0.5 / max(1e-9, p1); w0 = 0.5 / max(1e-9, 1.0 - p1)
    sample_w = np.where(y_ft > 0.5, w1, w0).astype(float)

    # Fit
    ft_coef = None; ft_bias = 0.0
    if X.shape[0] >= 12 and np.unique(y_ft).size == 2 and X.shape[1] > 0:
        ft_coef, ft_bias = logit_fit_l2_weighted(X, y_ft, sample_w, l2=1.2, max_iter=140, tol=1e-6)

        # Intercept calibration (match base rate)
        p_hat = logit_inv(ft_bias + X @ ft_coef)
        p_hat = np.clip(p_hat, 1e-4, 1-1e-4)
        p_bar = float(np.mean(y_ft)) if y_ft.size else 0.5
        p_bar = float(np.clip(p_bar, 1e-4, 1-1e-4))
        corr = math.log(p_bar/(1.0 - p_bar)) - math.log(np.mean(p_hat)/(1.0 - np.mean(p_hat)))
        ft_bias = float(ft_bias + corr)

        st.success(f"FT classifier trained (n={X.shape[0]}; feats: {', '.join(feat_names)}; base rate≈{p_bar:.2f}).")
    else:
        st.error("Unable to train FT classifier (need ≥12 rows, both classes, and valid features).")

    # Cuts (quantile-based with floors)
    if ft_coef is not None:
        p_cal = logit_inv(ft_bias + X @ ft_coef)
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
        "ft_coef": ft_coef, "ft_bias": ft_bias,
        "feat_names": feat_names, "mu": mu, "sd": sd,
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

# ============================== Inference (FT with ln1p_pmvol only for flow) ==============================
def predict_ft_prob_no_pmfrac(mc_m: float, gap_pct: float, atr_usd: float, float_m: float, si_pct: float,
                              rvol: float, pm_vol_m: float, pm_dol_m: float, catalyst: float) -> float:
    A = st.session_state.ARTIFACTS or {}
    coef = A.get("ft_coef"); bias = float(A.get("ft_bias") or 0.0)
    names = A.get("feat_names") or []
    mu = A.get("mu"); sd = A.get("sd")
    if coef is None or mu is None or sd is None or not names:
        return 0.50

    # Feature builders — mirror training exactly
    def _ln_gapf():    return _safe_log(_nz(gap_pct,0.0)/100.0)
    def _ln_mcap():    return _safe_log(mc_m)
    def _ln_atr():     return _safe_log(atr_usd)
    def _ln_float():   return _safe_log(float_m)
    def _ln1p_rvol():  return _safe_log(1.0 + _nz(rvol,0.0))
    def _ln1p_pmvol(): return _safe_log(_nz(pm_vol_m,0.0) + 1.0)
    def _catalyst():   return float(_nz(catalyst,0.0))
    def _maxpush_s():  return 0.0  # no live value; safe default

    feat_lookup = {
        "ln_gapf": _ln_gapf,
        "ln_mcap": _ln_mcap,
        "ln_atr": _ln_atr,
        "ln_float": _ln_float,
        "ln1p_rvol": _ln1p_rvol,
        "ln1p_pmvol": _ln1p_pmvol,
        "catalyst": _catalyst,
        "maxpush_s": _maxpush_s,
    }
    vals_raw = [float(feat_lookup[n]()) for n in names]
    X = (np.array(vals_raw, dtype=float) - mu) / sd

    # Same clipping and flow scaling
    X = _clip_z(X, 3.0)
    for j, name in enumerate(names):
        if name == "ln1p_pmvol":
            X[j] = np.clip(X[j], -2.5, 2.5) * st.session_state.get("FLOW_SENS", flow_sens)

    p = float(logit_inv(bias + np.dot(X, coef)))
    return float(np.clip(p, 1e-3, 1-1e-3))

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

        # Predicted day volume (display only)
        pred_vol_m = predict_day_volume_m_direct(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)

        # FT probability (NO pmfrac, NO fr)
        ft_prob = predict_ft_prob_no_pmfrac(mc_m, gap_pct, atr_usd, float_m, si_pct, rvol, pm_vol_m, pm_dol_m, cat)

        odds_cuts = st.session_state.get("ODDS_CUTS", {"very_high":0.85,"high":0.70,"moderate":0.55,"low":0.40})
        grade_cuts = st.session_state.get("GRADE_CUTS", {"App":0.92,"Ap":0.85,"A":0.75,"B":0.65,"C":0.50})
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

            # Diagnostics (display-only fields)
            "PM%_of_Pred": round(100.0 * _nz(pm_vol_m,0.0) / max(1e-6, pred_vol_m), 1),
            "PM$ / MC_%": round(100.0 * _nz(pm_dol_m,0.0) / max(1e-6, _nz(mc_m,0.0)), 1),

            # raw inputs for CSV
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

        # Download CSV
        st.download_button(
            "Download CSV (Ranking)",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        # Markdown table view + download
        st.markdown("### 📋 Ranking (Markdown view)")
        md_text = df_to_markdown_table(df, cols_to_show)
        st.code(md_text, language="markdown")
        st.download_button(
            "Download Markdown",
            md_text.encode("utf-8"),
            "ranking.md", "text/markdown", use_container_width=True
        )

        # Clear Ranking
        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
