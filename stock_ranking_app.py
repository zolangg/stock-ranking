# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:.78rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 11.5px; color:#374151; }
  ul { margin: 4px 0 0 0; padding-left: 16px; }
  li { margin-bottom: 2px; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Sidebar (Curves only) ==============================
st.sidebar.header("Curves")
BINS = st.sidebar.slider("Curve bins (histogram)", min_value=1, max_value=10, value=4, step=1)
show_baseline = st.sidebar.checkbox("Curves: show baseline", True)
plot_all_curves = st.sidebar.checkbox("Curves: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or plot one variable",
    ["gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m","fr_x",
     "pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"]
)

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}   # var -> model dict
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}
if "ODDS_CUTS" not in st.session_state: st.session_state.ODDS_CUTS = {}
if "GRADE_CUTS" not in st.session_state: st.session_state.GRADE_CUTS = {}
if "STACK_KEYS" not in st.session_state: st.session_state.STACK_KEYS = []
if "STACK_COEF" not in st.session_state: st.session_state.STACK_COEF = None
if "STACK_BIAS" not in st.session_state: st.session_state.STACK_BIAS = 0.0

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

def _fmt_value(v: float) -> str:
    if v is None or not np.isfinite(v): return "—"
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 1:    return f"{v:.2f}"
    return f"{v:.3f}"

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

# ================= Predicted Day Volume (for PM% fallback) =================
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """ ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$) """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

# ================= Rank-hist learning (DB-derived) =================
STRETCH_EPS = 0.10
CLASS_LIFT = 0.08  # checklist Good/Risk offset vs local baseline

def moving_average(y: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1: return y
    pad = w//2
    ypad = np.pad(y, (pad,pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')

def stretch_curve_to_unit(p: np.ndarray, base_p: float) -> np.ndarray:
    eps = STRETCH_EPS
    p = np.asarray(p, dtype=float)
    pmin, pmax = float(np.nanmin(p)), float(np.nanmax(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return np.full_like(p, base_p)
    scale = max(1e-9, (pmax - pmin))
    p_stretched = eps + (1.0 - 2.0*eps) * (p - pmin) / scale
    return np.clip(p_stretched, 1e-6, 1.0 - 1.0e-6)

# ---------- Local baseline smoothing ----------
def _smooth_local_baseline(centers: np.ndarray, p_curve: np.ndarray, support: np.ndarray, bandwidth: float = 0.22) -> np.ndarray:
    c = centers.astype(float)
    p = p_curve.astype(float)
    n = (support.astype(float) + 1e-9)
    diffs = (c[:, None] - c[None, :]) / max(1e-6, bandwidth)
    w = np.exp(-0.5 * diffs**2) * n[None, :]
    num = (w * p[None, :]).sum(axis=1)
    den = w.sum(axis=1)
    pb = np.where(den > 0, num / den, np.nan)
    pb = pd.Series(pb).interpolate(limit_direction="both").fillna(np.nanmean(pb)).to_numpy()
    return pb

def rank_hist_model(x: pd.Series, y: pd.Series, bins: int) -> Optional[Dict[str,Any]]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 40 or y.nunique() != 2:
        return None

    # ranks & bins
    ranks = x.rank(pct=True)
    B = int(bins)
    edges = np.linspace(0, 1, B + 1)
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, B-1)

    # counts
    total = np.bincount(idx, minlength=B)
    ft    = np.bincount(idx[y==1], minlength=B)
    p0_global = float(y.mean())

    # --- Laplace/Bayesian smoothing toward global baseline ---
    # kappa controls pull strength; increase when data is tiny
    kappa = max(6.0, 0.1 * len(x) / max(1, B))  # heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        p_bin_raw = np.where(total>0, ft/total, np.nan)
        p_bin = (ft + kappa * p0_global) / (total + kappa)

    # Smooth a touch for B>2
    if B > 2:
        p_series = pd.Series(p_bin).interpolate(limit_direction="both")
        p_fill   = p_series.fillna(p_series.mean()).to_numpy()
        p_smooth = moving_average(p_fill, w=3)
    else:
        # For 2 bins, keep the smoothed values we just computed
        p_smooth = p_bin

    centers = (edges[:-1] + edges[1:]) / 2.0
    p_base_var = float(np.average(p_smooth, weights=(total + 1e-9)))

    # --- When B==2: build a linear, directional curve across rank ---
    if B == 2:
        p_low, p_high = float(p_smooth[0]), float(p_smooth[1])
        # slope across rank 0..1, centered on per-variable baseline
        # linear map: p(r) = pb + (p_high - p_low) * (r - 0.5)
        # clamp to reasonable prob range to avoid extremes in tiny data
        p_line = p_base_var + (p_high - p_low) * (centers - 0.5)
        p_line = np.clip(p_line, 0.05, 0.95)
        p_ready = p_line
    else:
        p_ready = p_smooth

    # Stretch to epsilon band to avoid degeneracy, but milder when B==2
    eps_before = STRETCH_EPS
    eps_use = 0.08 if B == 2 else eps_before
    pmin, pmax = float(np.min(p_ready)), float(np.max(p_ready))
    if np.isfinite(pmin) and np.isfinite(pmax) and pmax > pmin:
        scale = (pmax - pmin)
        p_use = eps_use + (1.0 - 2.0*eps_use) * (p_ready - pmin) / scale
    else:
        p_use = np.full_like(p_ready, p_base_var)
    p_use = np.clip(p_use, 1e-6, 1 - 1e-6)

    # local baseline curve (for checklist comparisons)
    pb_curve = _smooth_local_baseline(centers, p_use, total, bandwidth=0.30 if B==2 else 0.22)

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    return {
        "edges": edges,
        "centers": centers,
        "support": total,
        "p_raw": p_use,            # already final for use
        "p0_global": p0_global,
        "p_base_var": p_base_var,
        "pb_curve": pb_curve,
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

def value_to_prob(var_key: str, model: Dict[str,Any], x_val: float) -> float:
    if model is None or not np.isfinite(x_val): return 0.5
    pr, vals = model["quantiles"]["pr"], model["quantiles"]["vals"]
    if x_val <= vals.min(): r = 0.0
    elif x_val >= vals.max(): r = 1.0
    else:
        idx = np.searchsorted(vals, x_val)
        i0 = max(1, min(idx, len(vals)-1))
        x0, x1 = vals[i0-1], vals[i0]
        p0, p1 = pr[i0-1], pr[i0]
        t = (x_val - x0) / (x1 - x0) if x1 != x0 else 0.0
        r = float(p0 + t*(p1 - p0))
    centers = model["centers"]; p = model["p"]
    j = int(np.clip(np.searchsorted(centers, r), 0, len(centers)-1))
    p_local = float(p[j])
    return float(np.clip(p_local, 1e-6, 1-1e-6))

# ---------- Prior & simple anchors (kept) ----------
def si_directional_prior(x: float) -> Optional[float]:
    if not np.isfinite(x): return None
    k = 0.25; x0 = 10.0
    base = 0.25 + 0.65 / (1 + math.exp(-k * (x - x0)))
    return float(np.clip(base, 0.20, 0.90))

def blend_with_prior(var_key: str, x: float, p_learned: float) -> float:
    if var_key == "si_pct":
        prior = si_directional_prior(x)
        if prior is not None:
            p_learned = 0.40*prior + 0.60*p_learned
    return float(np.clip(p_learned, 1e-6, 1-1e-6))

ANCHORS = {
    "pm_pct_daily": (8.0, 25.0),
    "pm_pct_pred":  (8.0, 25.0),
}

def anchor_pm_percent(var_key: str, p: float, x_val: float, pb: float) -> float:
    band = ANCHORS.get(var_key)
    if not band or not np.isfinite(x_val): 
        return p
    lo, hi = band
    if lo <= x_val <= hi:
        return float(max(p, min(0.95, pb + 0.10)))
    return p

def anchor_atr(p: float, x: float, pb: float) -> float:
    if not np.isfinite(x): return p
    if 0.15 <= x <= 0.40:
        return max(p, min(0.92, pb + 0.10))
    if x < 0.10:
        return min(p, 0.50)
    return p

# Rank helpers
def _rank_from_value(model: dict, x: float) -> Optional[float]:
    if not np.isfinite(x): return None
    pr, vals = model["quantiles"]["pr"], model["quantiles"]["vals"]
    if x <= vals.min(): return 0.0
    if x >= vals.max(): return 1.0
    idx = np.searchsorted(vals, x)
    i0 = max(1, min(idx, len(vals)-1))
    x0, x1 = vals[i0-1], vals[i0]
    p0, p1 = pr[i0-1], pr[i0]
    t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    return float(p0 + t*(p1 - p0))

def _baseline_at_value(model: dict, x: float) -> float:
    pb_curve = model.get("pb_curve", None)
    centers  = model.get("centers", None)
    if pb_curve is not None and centers is not None and np.isfinite(x):
        r = _rank_from_value(model, x)
        j = int(np.clip(np.searchsorted(centers, r), 0, len(centers)-1))
        pb = float(pb_curve[j])
        if np.isfinite(pb):
            return pb
    return float(model.get("p_base_var", 0.5))

# ---------- Exponential percentile warping (for calibration) ----------
def _warp_p(s: float, alpha: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return float(1.0 - (1.0 - s) ** alpha)

def _inv_warp_p(s_warped: float, alpha: float) -> float:
    s_warped = float(np.clip(s_warped, 0.0, 1.0))
    return float(1.0 - (1.0 - s_warped) ** (1.0 / max(1e-9, alpha)))

# ---------- Hard floors to prevent absurd labels ----------
ODDS_FLOORS  = {"very_high": 0.85, "high": 0.70, "moderate": 0.55, "low": 0.40}
GRADE_FLOORS = {"App": 0.92, "Ap": 0.85, "A": 0.75, "B": 0.65, "C": 0.50}

# ---------- Logistic stacking (ridge) ----------
def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def _fit_logistic_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    IRLS (Newton) for logistic regression with L2 penalty on weights (not intercept).
    X: [n, k] features (e.g., logits of per-variable probabilities)
    y: [n] binary {0,1}
    Returns (coef_[k], intercept_)
    """
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    w = np.zeros(k+1)

    R = np.eye(k+1)
    R[0,0] = 0.0
    R *= l2

    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p * (1 - p)
        if np.all(W < 1e-8):
            break
        WX = Xb * W[:, None]
        H  = Xb.T @ WX + R
        g  = Xb.T @ (y - p)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w + delta
        if np.linalg.norm(delta) < tol:
            w = w_new
            break
        w = w_new

    intercept_ = float(w[0])
    coef_ = w[1:].astype(float)
    return coef_, intercept_

def _predict_logistic(X: np.ndarray, coef_: np.ndarray, intercept_: float) -> np.ndarray:
    z = intercept_ + X @ coef_
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

# ---------- Per-row per-variable probabilities (features for stacking) ----------
def _per_var_probs_for_row(models: Dict[str,dict], row: dict) -> Tuple[List[str], np.ndarray]:
    """
    Produce the AFTER-adjustment per-variable probabilities used for logistic stacking.
    Adjustments: SI prior, ATR anchor, PM% anchors.
    """
    use_pm_daily = "pm_pct_daily" in models
    var_order = [
        "gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m","fr_x","pmmc_pct",
        ("pm_pct_daily" if use_pm_daily else "pm_pct_pred"), "catalyst"
    ]

    keys: List[str] = []
    vals: List[float] = []
    for k in var_order:
        mdl = models.get(k)
        if mdl is None:
            continue
        x = row.get(k, np.nan)
        p = value_to_prob(k, mdl, x)
        pb_local = _baseline_at_value(mdl, x)
        p = blend_with_prior(k, x, p)
        if k == "atr_usd":
            p = anchor_atr(p, x, pb_local)
        p = anchor_pm_percent(k, p, x, pb_local)
        keys.append(k)
        vals.append(float(np.clip(p, 1e-6, 1-1e-6)))
    return keys, np.array(vals, dtype=float)

# ---------- Calibration helpers ----------
def _prob_to_odds(prob: float, cuts: Dict[str, float]) -> str:
    if prob >= cuts["very_high"]: return "Very High Odds"
    if prob >= cuts["high"]:      return "High Odds"
    if prob >= cuts["moderate"]:  return "Moderate Odds"
    if prob >= cuts["low"]:       return "Low Odds"
    return "Very Low Odds"

def _prob_to_grade(prob: float, cuts: Dict[str, float]) -> str:
    if prob >= cuts["App"]: return "A++"
    if prob >= cuts["Ap"]:  return "A+"
    if prob >= cuts["A"]:   return "A"
    if prob >= cuts["B"]:   return "B"
    if prob >= cuts["C"]:   return "C"
    return "D"

# ============================== Upload & Learn (Main Pane) ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn rules from merged", use_container_width=True)

if learn_btn:
    if not uploaded:
        st.error("Upload a workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if merged_sheet not in xls.sheet_names:
                st.error(f"Sheet '{merged_sheet}' not found. Available: {xls.sheet_names}")
            else:
                raw = pd.read_excel(xls, merged_sheet)

                # column mapping
                col_ft    = _pick(raw, ["ft","FT"])
                col_gap   = _pick(raw, ["gap %","gap%","premarket gap","gap"])
                col_atr   = _pick(raw, ["atr","atr $","atr$","atr (usd)"])
                col_rvol  = _pick(raw, ["rvol @ bo","rvol","relative volume"])
                col_pmvol = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)"])
                col_pmdol = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
                col_float = _pick(raw, ["float m shares","public float (m)","float (m)","float"])
                col_mcap  = _pick(raw, ["marketcap m","market cap (m)","mcap m","mcap"])
                col_si    = _pick(raw, ["si","short interest %","short float %","short interest (float) %"])
                col_cat   = _pick(raw, ["catalyst","news","pr"])
                col_daily = _pick(raw, ["daily vol (m)","day volume (m)","volume (m)"])

                if col_ft is None:
                    st.error("No 'FT' column found in merged sheet.")
                else:
                    df = pd.DataFrame()
                    df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce")

                    if col_gap:   df["gap_pct"]  = pd.to_numeric(raw[col_gap],   errors="coerce")
                    if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                    if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                    if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                    if col_pmdol: df["pm_dol_m"] = pd.to_numeric(raw[col_pmdol], errors="coerce")
                    if col_float: df["float_m"]  = pd.to_numeric(raw[col_float],  errors="coerce")
                    if col_mcap:  df["mcap_m"]   = pd.to_numeric(raw[col_mcap],  errors="coerce")
                    if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")
                    if col_cat:   df["catalyst"] = pd.to_numeric(raw[col_cat],   errors="coerce").clip(0,1)
                    if col_daily: df["daily_vol_m"] = pd.to_numeric(raw[col_daily], errors="coerce")

                    # derived
                    if {"pm_vol_m","float_m"}.issubset(df.columns):
                        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
                    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
                        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"]
                    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                        def _pred_row(r):
                            try:
                                return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                            except Exception:
                                return np.nan
                        pred = df.apply(_pred_row, axis=1)
                        df["pm_pct_pred"] = 100.0 * df["pm_vol_m"] / pred

                    df = df[df["FT"].notna()]
                    y = df["FT"].astype(float)

                    # learn models per variable (no sweet-band anchors, no guardrails)
                    candidates = [
                        "gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m",
                        "fr_x","pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"
                    ]
                    models: Dict[str, dict] = {}
                    for v in candidates:
                        if v in df.columns:
                            m = rank_hist_model(df[v], y, bins=BINS)
                            if m is not None:
                                centers = m["centers"]
                                p_base_var = m["p_base_var"]

                                # stretch raw curve (no penalty/guardrails)
                                p_use = stretch_curve_to_unit(m["p_raw"], base_p=p_base_var)
                                m["p"] = p_use

                                # local baseline curve
                                pb_curve = _smooth_local_baseline(centers, p_use, m["support"], bandwidth=0.22)
                                m["pb_curve"] = pb_curve

                                models[v] = m

                    # ---------- Calibration on the merged sheet (logistic stacking) ----------
                    stack_keys: List[str] = []
                    X_list: List[np.ndarray] = []
                    y_list: List[int] = []

                    if models:
                        # ensure derived fields exist (in case)
                        if {"pm_vol_m","float_m"}.issubset(df.columns) and "fr_x" not in df.columns:
                            df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                        if {"pm_dol_m","mcap_m"}.issubset(df.columns) and "pmmc_pct" not in df.columns:
                            df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

                        if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                            def _pred_row_cal2(r):
                                try:
                                    return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                                except Exception:
                                    return np.nan
                            pred_cal2 = df.apply(_pred_row_cal2, axis=1)
                            df["pm_pct_pred"] = np.where(
                                (pred_cal2 > 0) & np.isfinite(pred_cal2),
                                100.0 * df.get("pm_vol_m", np.nan) / pred_cal2,
                                df.get("pm_pct_pred", np.nan)
                            )

                        for _, rr in df.iterrows():
                            rowd = {k: float(rr[k]) if k in df.columns and np.isfinite(rr[k]) else np.nan for k in df.columns}
                            if "pm_pct_daily" in models:
                                rowd.setdefault("pm_pct_daily", float(rowd.get("pm_pct_pred", np.nan)))
                            keys, pvec = _per_var_probs_for_row(models, rowd)
                            if not stack_keys:
                                stack_keys = keys[:]
                            if keys != stack_keys:
                                key_to_p = {k: v for k, v in zip(keys, pvec)}
                                pvec = np.array([key_to_p.get(k, 0.5) for k in stack_keys], dtype=float)
                            X_list.append(_safe_logit(pvec))
                            y_list.append(int(rr["FT"]))

                    X = np.array(X_list, dtype=float)
                    y_arr = np.array(y_list, dtype=float)

                    logit_coef = None
                    logit_intercept = 0.0
                    cal_probs = np.array([], dtype=float)

                    if X.size > 0 and np.unique(y_arr).size == 2:
                        logit_coef, logit_intercept = _fit_logistic_ridge(X, y_arr, l2=1.0, max_iter=60, tol=1e-6)
                        cal_probs = _predict_logistic(X, logit_coef, logit_intercept)
                    else:
                        if X.size > 0:
                            coef_eq = np.ones(X.shape[1]) / max(1, X.shape[1])
                            logit_coef, logit_intercept = coef_eq, 0.0
                            cal_probs = _predict_logistic(X, logit_coef, logit_intercept)

                    # ---------- Exponential distribution for cuts ----------
                    ALPHA_ODDS  = 2.2
                    ALPHA_GRADE = 2.8

                    odds_targets_warped = {
                        "very_high": 0.98,
                        "high":      0.90,
                        "moderate":  0.65,
                        "low":       0.35,
                    }
                    grade_targets_warped = {
                        "App": 0.995,
                        "Ap":  0.97,
                        "A":   0.90,
                        "B":   0.65,
                        "C":   0.35,
                    }

                    def _cut_from_probs(probs: np.ndarray, warped_p: float, alpha: float) -> float:
                        if probs.size == 0:
                            return float(warped_p)
                        raw_p = _inv_warp_p(warped_p, alpha)
                        return float(np.quantile(probs, raw_p))

                    odds_cuts = {k: _cut_from_probs(cal_probs, v, ALPHA_ODDS) for k, v in odds_targets_warped.items()}
                    grade_cuts = {k: _cut_from_probs(cal_probs, v, ALPHA_GRADE) for k, v in grade_targets_warped.items()}

                    # Raise to hard floors
                    for k, floor in ODDS_FLOORS.items():
                        if k in odds_cuts:
                            odds_cuts[k] = max(odds_cuts[k], floor)
                    for k, floor in GRADE_FLOORS.items():
                        if k in grade_cuts:
                            grade_cuts[k] = max(grade_cuts[k], floor)

                    # Save models + stacking
                    st.session_state.MODELS       = models
                    st.session_state.ODDS_CUTS    = odds_cuts
                    st.session_state.GRADE_CUTS   = grade_cuts
                    st.session_state.STACK_KEYS   = stack_keys
                    st.session_state.STACK_COEF   = logit_coef
                    st.session_state.STACK_BIAS   = logit_intercept

                    st.success(f"Learned {len(models)} variables with {BINS} bins and trained logistic stacking.")
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank, tab_curves = st.tabs(["➕ Add Stock", "📊 Ranking", "📈 Curves"])

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
            dilution_flag = st.select_slider("Dilution present?", options=[0,1], value=0,
                                             help="0 = none/negligible, 1 = present/overhang (penalizes log-odds)")
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"])

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived live
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted daily (fallback for PM%)
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        # Models
        models  = st.session_state.MODELS or {}

        # Prefer pm_pct_daily curve if learned; else use pm_pct_pred
        use_pm_pct_daily = "pm_pct_daily" in models
        var_vals: Dict[str, float] = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "float_m": float_m, "mcap_m": mc_m, "fr_x": fr_x, "pmmc_pct": pmmc_pct,
            ("pm_pct_daily" if use_pm_pct_daily else "pm_pct_pred"): (np.nan if use_pm_pct_daily else pm_pct_pred),
            "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }
        if use_pm_pct_daily:
            var_vals["pm_pct_daily"] = pm_pct_pred  # proxy during premarket

        # ----- Live scoring via learned logistic stacker -----
        keys_live, pvec_live = _per_var_probs_for_row(models, var_vals)
        stack_keys = st.session_state.get("STACK_KEYS", keys_live)
        coef_ = st.session_state.get("STACK_COEF", None)
        bias_ = st.session_state.get("STACK_BIAS", 0.0)

        # Align features to training order
        if keys_live != stack_keys:
            k2p = {k:v for k,v in zip(keys_live, pvec_live)}
            pvec_live = np.array([k2p.get(k, 0.5) for k in stack_keys], dtype=float)
        X_live = _safe_logit(np.array(pvec_live, dtype=float))[None, :]

        if coef_ is not None and len(coef_) == X_live.shape[1]:
            z_sum = float(bias_ + (X_live @ coef_)[0])
        else:
            z_sum = float(np.mean(X_live))

        # dilution penalty in logit space
        z_sum += -0.90 * float(dilution_flag)
        z_sum = float(np.clip(z_sum, -12, 12))
        numeric_prob = 1.0 / (1.0 + math.exp(-z_sum))

        # Use calibrated thresholds
        odds_cuts = st.session_state.get("ODDS_CUTS", {"very_high":0.85,"high":0.70,"moderate":0.55,"low":0.40})
        grade_cuts = st.session_state.get("GRADE_CUTS", {"App":0.92,"Ap":0.85,"A":0.75,"B":0.65,"C":0.50})

        odds_name = _prob_to_odds(numeric_prob, odds_cuts)
        level = _prob_to_grade(numeric_prob, grade_cuts)

        final_score = float(np.clip(numeric_prob*100.0, 0.0, 100.0))
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if level in ("A++","A+","A") else
            '<span class="pill pill-warn">Constructive</span>' if level in ("B","C") else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )

        # Checklist with Good/Caution/Risk relative to local baseline (display only)
name_map = {
    "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
    "float_m":"Float (M)","mcap_m":"MarketCap (M)","fr_x":"PM Float Rotation ×",
    "pmmc_pct":"PM $Vol / MC %","pm_pct_daily":"PM Vol % of Daily",
    "pm_pct_pred":"PM Vol % of Pred","catalyst":"Catalyst"
}

# grab learned stacking to compute contributions
stack_keys = st.session_state.get("STACK_KEYS", [])
coef_ = st.session_state.get("STACK_COEF", None)

def _logit(x): 
    x = float(np.clip(x, 1e-6, 1-1e-6))
    return math.log(x/(1-x))

good, warn, risk = [], [], []
detail_rows = []  # optional: for a debugging table

# thresholds for contribution buckets (logit-space * weight units)
# tune: 0.25 ~= noticeable, 0.10 ~= small tilt
TH_GOOD = 0.25
TH_RISK = -0.25

for i, k in enumerate(stack_keys):
    mdl = models.get(k)
    x  = var_vals.get(k, np.nan)
    if mdl is None:
        continue

    # recompute adjusted per-var prob the same way features were built
    p = value_to_prob(k, mdl, x)
    pb = _baseline_at_value(mdl, x)
    p  = blend_with_prior(k, x, p)
    if k == "atr_usd":
        p = anchor_atr(p, x, pb)
    p  = anchor_pm_percent(k, p, x, pb)
    p  = float(np.clip(p, 1e-6, 1-1e-6))
    pb = float(np.clip(pb,1e-6,1-1e-6))

    # contribution uses learned coef; if none (rare), default to 1.0
    beta = float(coef_[i]) if (coef_ is not None and i < len(coef_)) else 1.0
    contrib = beta * (_logit(p) - _logit(pb))

    nm = name_map.get(k, k)
    p_pct = int(round(p*100))
    # bucket by contribution
    if contrib >= TH_GOOD:
        good.append(f"{nm}: {_fmt_value(x)} — good (p≈{p_pct}%, Δlogit×β=+{contrib:.2f})")
    elif contrib <= TH_RISK:
        risk.append(f"{nm}: {_fmt_value(x)} — risk (p≈{p_pct}%, Δlogit×β={contrib:.2f})")
    else:
        warn.append(f"{nm}: {_fmt_value(x)} — caution (p≈{p_pct}%, Δlogit×β={contrib:.2f})")

    # keep an internal row if you want to show a table later
    detail_rows.append({
        "Variable": nm,
        "Value": _fmt_value(x),
        "p": round(p,4),
        "pb": round(pb,4),
        "beta": round(beta,3),
        "Δlogit": round(_logit(p) - _logit(pb),3),
        "Contribution": round(contrib,3),
    })

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PredVol_M"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level") else 0.0
        st.dataframe(
            df[cols_to_show], use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
            }
        )
        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )
    else:
        st.info("Add at least one stock.")

# ============================== Curves ==============================
with tab_curves:
    st.markdown('<div class="section-title">Learned Curves (rank-space FT rate, local baselines dashed)</div>', unsafe_allow_html=True)
    models = st.session_state.MODELS or {}
    if not models:
        st.info("Upload + Learn first.")
    else:
        if plot_all_curves:
            learned_vars = list(models.keys())
            n = len(learned_vars)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.6, nrows*3.2))
            axes = np.atleast_2d(axes)
            for i, var in enumerate(learned_vars):
                ax = axes[i//ncols, i % ncols]
                m = models[var]
                centers = m["centers"]; p = m["p"]
                ax.plot(centers, p, lw=2)
                pb_curve = m.get("pb_curve")
                if show_baseline:
                    if pb_curve is not None:
                        ax.plot(centers, pb_curve, ls="--", lw=1)
                    else:
                        ax.axhline(m.get("p_base_var", 0.5), ls="--", lw=1)
                ax.set_title(var, fontsize=11)
                ax.set_xlabel("Rank (percentile)", fontsize=10)
                ax.set_ylabel("P(FT)", fontsize=10)
                ax.tick_params(labelsize=9)
            # remove any empty subplots
            total_axes = nrows * ncols
            for j in range(n, total_axes):
                fig.delaxes(axes[j//ncols, j % ncols])
            st.pyplot(fig, clear_figure=True)
        else:
            var = sel_curve_var
            m = models.get(var)
            if m is None:
                st.warning(f"No curve learned for '{var}'.")
            else:
                centers = m["centers"]; p = m["p"]
                fig, ax = plt.subplots(figsize=(6.4, 3.4))
                ax.plot(centers, p, lw=2)
                pb_curve = m.get("pb_curve")
                if show_baseline:
                    if pb_curve is not None:
                        ax.plot(centers, pb_curve, ls="--", lw=1)
                    else:
                        ax.axhline(m.get("p_base_var", 0.5), ls="--", lw=1)
                ax.set_xlabel("Rank (percentile of variable)", fontsize=10)
                ax.set_ylabel("P(FT | rank)", fontsize=10)
                ax.set_title(f"{var} — FT curve (baseline dashed)", fontsize=11)
                ax.tick_params(labelsize=9)
                st.pyplot(fig, clear_figure=True)

        # Show calibrated cuts for transparency
        odds_cuts = st.session_state.get("ODDS_CUTS")
        grade_cuts = st.session_state.get("GRADE_CUTS")
        if odds_cuts and grade_cuts:
            st.caption(
                f"Odds cuts (prob): Very High ≥ {odds_cuts['very_high']:.3f}, "
                f"High ≥ {odds_cuts['high']:.3f}, Moderate ≥ {odds_cuts['moderate']:.3f}, "
                f"Low ≥ {odds_cuts['low']:.3f}."
            )
            st.caption(
                f"Grade cuts (prob): A++ ≥ {grade_cuts['App']:.3f}, A+ ≥ {grade_cuts['Ap']:.3f}, "
                f"A ≥ {grade_cuts['A']:.3f}, B ≥ {grade_cuts['B']:.3f}, C ≥ {grade_cuts['C']:.3f}."
            )
