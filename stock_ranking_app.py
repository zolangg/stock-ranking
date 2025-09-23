# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

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
BINS = st.sidebar.slider("Curve bins (histogram)", min_value=5, max_value=50, value=5, step=1)
show_baseline = st.sidebar.checkbox("Curves: show baseline", True)
plot_all_curves = st.sidebar.checkbox("Curves: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or plot one variable",
    ["gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m","fr_x",
     "pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"]
)

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}   # var -> model dict
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {} # var -> weight
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}
if "ODDS_CUTS" not in st.session_state: st.session_state.ODDS_CUTS = {}
if "GRADE_CUTS" not in st.session_state: st.session_state.GRADE_CUTS = {}

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

# (legacy helpers not used for labels anymore; kept just in case)
def odds_label(score: float) -> str:
    if score >= 85: return "Very High Odds"
    elif score >= 70: return "High Odds"
    elif score >= 55: return "Moderate Odds"
    elif score >= 40: return "Low Odds"
    else: return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 90 else
            "A+"  if score_pct >= 85 else
            "A"   if score_pct >= 75 else
            "B"   if score_pct >= 65 else
            "C"   if score_pct >= 50 else "D")

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
    return np.clip(p_stretched, 1e-6, 1.0 - 1e-6)

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

    ranks = x.rank(pct=True)
    edges = np.linspace(0, 1, int(bins) + 1)
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, int(bins)-1)

    total = np.bincount(idx, minlength=int(bins))
    ft    = np.bincount(idx[y==1], minlength=int(bins))
    with np.errstate(divide='ignore', invalid='ignore'):
        p_bin = np.where(total>0, ft/total, np.nan)

    p_series = pd.Series(p_bin).interpolate(limit_direction="both")
    p_fill   = p_series.fillna(p_series.mean()).to_numpy()
    # window 3 still ok for larger bins; it just smooths lightly
    p_smooth = moving_average(p_fill, w=3)

    centers = (edges[:-1] + edges[1:]) / 2.0
    p0_global = float(y.mean())
    p_base_var = float(np.average(p_smooth, weights=(total + 1e-9)))

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    return {
        "edges": edges,
        "centers": centers,
        "support": total,
        "p_raw": p_smooth,
        "p0_global": p0_global,
        "p_base_var": p_base_var,
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

def auc_weight(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x,y = x[mask], y[mask]
    if len(x) < 40 or y.nunique()!=2: return 0.0
    r = x.rank(method="average")
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return 0.0
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    sep = abs(auc-0.5)*2.0
    return float(max(0.05, sep))

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

# ---------- Calibration helpers ----------
def _stack_prob_for_row(models: Dict[str,dict], weights: Dict[str,float], row: dict) -> float:
    """Recompute model-based prob for one historical row (for calibration)."""
    z_sum = 0.0
    use_pm_daily = "pm_pct_daily" in models
    var_vals = {
        "gap_pct": row.get("gap_pct", np.nan),
        "atr_usd": row.get("atr_usd", np.nan),
        "rvol":    row.get("rvol", np.nan),
        "si_pct":  row.get("si_pct", np.nan),
        "float_m": row.get("float_m", np.nan),
        "mcap_m":  row.get("mcap_m", np.nan),
        "fr_x":    row.get("fr_x", np.nan),
        "pmmc_pct":row.get("pmmc_pct", np.nan),
        ("pm_pct_daily" if use_pm_daily else "pm_pct_pred"): row.get("pm_pct_pred", np.nan),
        "catalyst": row.get("catalyst", 0.0),
    }
    if use_pm_daily:
        var_vals["pm_pct_daily"] = row.get("pm_pct_pred", np.nan)

    for k, x in var_vals.items():
        mdl = models.get(k)
        if mdl is None: 
            continue
        p = value_to_prob(k, mdl, x)
        pb_local = _baseline_at_value(mdl, x)

        # SI prior
        p = blend_with_prior(k, x, p)

        # domain anchors
        if k == "atr_usd":
            p = anchor_atr(p, x, pb_local)
        p = anchor_pm_percent(k, p, x, pb_local)

        p = float(np.clip(p, 1e-6, 1-1e-6))
        w = float(weights.get(k, 0.0))
        z_sum += w * math.log(p/(1-p))

    z_sum = float(np.clip(z_sum, -12, 12))
    prob = 1.0 / (1.0 + math.exp(-z_sum))
    return float(prob)

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
                    models, weights = {}, {}
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

                                # weights from separation & amplitude
                                w_sep = auc_weight(df[v], y)
                                amp   = float(np.nanstd(p_use))
                                weights[v] = max(0.05, min(1.0, 0.7*w_sep + 0.3*(amp*4)))

                    # normalize weights
                    if weights:
                        s = sum(weights.values()) or 1.0
                        weights = {k: v/s for k, v in weights.items()}

                    # ---------- Calibration on the merged sheet ----------
                    cal_probs = []
                    if models and weights:
                        # ensure derived fields exist
                        if {"pm_vol_m","float_m"}.issubset(df.columns) and "fr_x" not in df.columns:
                            df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                        if {"pm_dol_m","mcap_m"}.issubset(df.columns) and "pmmc_pct" not in df.columns:
                            df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

                        if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                            def _pred_row_cal(r):
                                try:
                                    return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                                except Exception:
                                    return np.nan
                            pred_cal = df.apply(_pred_row_cal, axis=1)
                            df["pm_pct_pred"] = np.where(
                                (pred_cal > 0) & np.isfinite(pred_cal),
                                100.0 * df.get("pm_vol_m", np.nan) / pred_cal,
                                df.get("pm_pct_pred", np.nan)
                            )

                        needed = {"gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","catalyst"}
                        for _, rr in df.iterrows():
                            rowd = {k: (float(rr[k]) if k in df.columns else np.nan) for k in needed}
                            cal_probs.append(_stack_prob_for_row(models, weights, rowd))

                    cal_probs = np.array(cal_probs, dtype=float)
                    cal_probs = cal_probs[np.isfinite(cal_probs)]

                    # ---------- Exponential distribution for cuts ----------
                    ALPHA_ODDS  = 2.2   # more aggressive = rarer top odds
                    ALPHA_GRADE = 2.8   # grades even steeper

                    # target warped percentiles (UI distribution intent)
                    odds_targets_warped = {
                        "very_high": 0.98,  # ~top 2%
                        "high":      0.90,  # ~top 10%
                        "moderate":  0.65,  # ~top 35%
                        "low":       0.35,  # ~top 65%
                    }
                    grade_targets_warped = {
                        "App": 0.995,  # A++ ~ top 0.5%
                        "Ap":  0.97,   # A+  ~ top 3%
                        "A":   0.90,   # A   ~ top 10%
                        "B":   0.65,   # B   ~ top 35%
                        "C":   0.35,   # C   ~ top 65%
                    }

                    def _cut_from_probs(probs: np.ndarray, warped_p: float, alpha: float) -> float:
                        if probs.size == 0:
                            return float(warped_p)  # fallback
                        raw_p = _inv_warp_p(warped_p, alpha)
                        return float(np.quantile(probs, raw_p))

                    odds_cuts = {k: _cut_from_probs(cal_probs, v, ALPHA_ODDS) for k, v in odds_targets_warped.items()}
                    grade_cuts = {k: _cut_from_probs(cal_probs, v, ALPHA_GRADE) for k, v in grade_targets_warped.items()}

                    # Raise calibrated cuts up to hard floors (prevents e.g. 0.51 -> A+/High)
                    for k, floor in ODDS_FLOORS.items():
                        if k in odds_cuts:
                            odds_cuts[k] = max(odds_cuts[k], floor)
                    for k, floor in GRADE_FLOORS.items():
                        if k in grade_cuts:
                            grade_cuts[k] = max(grade_cuts[k], floor)

                    # Save
                    st.session_state.MODELS  = models
                    st.session_state.WEIGHTS = weights
                    st.session_state.ODDS_CUTS = odds_cuts
                    st.session_state.GRADE_CUTS = grade_cuts

                    st.success(f"Learned {len(models)} variables with {BINS} bins and set exponential-calibrated thresholds.")
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

        # Models & weights
        models  = st.session_state.MODELS or {}
        weights = st.session_state.WEIGHTS or {}

        # Prefer pm_pct_daily curve if learned; else use pm_pct_pred
        use_pm_pct_daily = "pm_pct_daily" in models
        var_vals: Dict[str, float] = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "float_m": float_m, "mcap_m": mc_m, "fr_x": fr_x, "pmmc_pct": pmmc_pct,
            ("pm_pct_daily" if use_pm_pct_daily else "pm_pct_pred"): (np.nan if use_pm_pct_daily else pm_pct_pred),
            "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }
        if use_pm_pct_daily:
            # During premarket, use pm_pct_pred as proxy value to position inside the curve
            var_vals["pm_pct_daily"] = pm_pct_pred

        # Odds stacking with local baselines, SI prior, ATR/PM% anchors
        parts: Dict[str, Dict[str, float]] = {}
        z_sum = 0.0
        for k, x in var_vals.items():
            mdl = models.get(k)
            if mdl is None:
                continue
            p = value_to_prob(k, mdl, x)        # learned curve at x
            pb_local = _baseline_at_value(mdl, x)

            # directional prior
            p = blend_with_prior(k, x, p)

            # explicit domain anchors (no sweet-band, no guardrails)
            if k == "atr_usd":
                p = anchor_atr(p, x, pb_local)
            p = anchor_pm_percent(k, p, x, pb_local)

            w = float(weights.get(k, 0.0))
            parts[k] = {"x": x, "p": p, "w": w, "pb": pb_local}
            p_clip = float(np.clip(p, 1e-6, 1-1e-6))
            z_sum += w * math.log(p_clip/(1-p_clip))

        # dilution penalty (binary)
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

        # Checklist with Good/Caution/Risk relative to local baseline
        name_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
            "float_m":"Float (M)","mcap_m":"MarketCap (M)","fr_x":"PM Float Rotation ×",
            "pmmc_pct":"PM $Vol / MC %","pm_pct_daily":"PM Vol % of Daily",
            "pm_pct_pred":"PM Vol % of Pred","catalyst":"Catalyst"
        }
        good, warn, risk = [], [], []
        for k, d in parts.items():
            nm = name_map.get(k, k)
            p = d["p"]; x = d["x"]; pb = d["pb"]
            p_pct = int(round(p*100))
            if p >= pb + CLASS_LIFT:   good.append(f"{nm}: {_fmt_value(x)} — good (p≈{p_pct}%)")
            elif p <= pb - CLASS_LIFT: risk.append(f"{nm}: {_fmt_value(x)} — risk (p≈{p_pct}%)")
            else:                      warn.append(f"{nm}: {_fmt_value(x)} — caution (p≈{p_pct}%)")

        # Save row
        row = {
            "Ticker": ticker,
            "Odds": odds_name,
            "Level": level,
            "FinalScore": round(final_score, 2),
            "PredVol_M": round(pred_vol_m, 2),
            "VerdictPill": verdict_pill,
            "GoodList": good, "WarnList": warn, "RiskList": risk,
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})")

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

        with st.expander("Premarket Checklist", expanded=True):
            st.markdown(f"**Verdict:** {l.get('VerdictPill','—')}", unsafe_allow_html=True)
            g,w,r = st.columns(3)
            def ul(items): return "<ul>"+"".join([f"<li>{x}</li>" for x in items])+"</ul>" if items else "<ul><li>—</li></ul>"
            with g: st.markdown("**Good**");    st.markdown(ul(l.get("GoodList",[])), unsafe_allow_html=True)
            with w: st.markdown("**Caution**"); st.markdown(ul(l.get("WarnList",[])), unsafe_allow_html=True)
            with r: st.markdown("**Risk**");    st.markdown(ul(l.get("RiskList",[])), unsafe_allow_html=True)

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
