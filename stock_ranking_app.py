# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking — PMH BO Merged", layout="wide")
st.title("Premarket Stock Ranking — Data-derived (PMH BO Merged)")

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

# ============================== Sidebar (minimal) ==============================
st.sidebar.header("Curves & Penalty")
tail_strength = st.sidebar.slider("High-tail penalty (exhaustion vars)", 0.0, 1.0, 0.55, 0.05)
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
EXHAUSTION_VARS = {"gap_pct","rvol","pmmc_pct","fr_x"}  # high tails can exhaust

BINS = 5                # fixed
STRETCH_EPS = 0.10      # fixed (min→0.10, max→0.90)
STRETCH_BLEND = 0.0     # fixed (no blend to baseline)
CLASS_LIFT = 0.08       # fixed threshold for Good/Risk

def moving_average(y: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1: return y
    pad = w//2
    ypad = np.pad(y, (pad,pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')

def stretch_curve_to_unit(p: np.ndarray, base_p: float) -> np.ndarray:
    """Stretch curve min→ε and max→1−ε; (γ=0) no blend to baseline."""
    eps = STRETCH_EPS
    p = np.asarray(p, dtype=float)
    pmin, pmax = float(np.nanmin(p)), float(np.nanmax(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return np.full_like(p, base_p)
    scale = max(1e-9, (pmax - pmin))
    p_stretched = eps + (1.0 - 2.0*eps) * (p - pmin) / scale
    return np.clip(p_stretched, 1e-6, 1.0 - 1e-6)

def tail_penalize(var_key: str, centers: np.ndarray, p: np.ndarray, p_base: float, strength: float) -> np.ndarray:
    """Penalize extreme high ranks for exhaustion variables only, around the per-variable base."""
    if var_key not in EXHAUSTION_VARS or strength <= 0: 
        return p
    # ramp down from rank 0.90 → 1.00
    tail = np.clip((centers - 0.90) / 0.10, 0, 1)
    penalty = 1.0 - strength * tail
    return p_base + (p - p_base) * penalty

def rank_hist_model(x: pd.Series, y: pd.Series) -> Optional[Dict[str,Any]]:
    """Learn FT curve in rank space (bins=5), smooth, tail-penalize later, stretch to ε=0.1..0.9"""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 40 or y.nunique() != 2:
        return None

    ranks = x.rank(pct=True)   # 0..1
    edges = np.linspace(0,1,BINS+1)
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, BINS-1)

    total = np.bincount(idx, minlength=BINS)
    ft    = np.bincount(idx[y==1], minlength=BINS)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_bin = np.where(total>0, ft/total, np.nan)

    # Light smooth (bins already small)
    p_series = pd.Series(p_bin).interpolate(limit_direction="both")
    p_fill   = p_series.fillna(p_series.mean()).to_numpy()
    p_smooth = moving_average(p_fill, w=3)

    centers = (edges[:-1] + edges[1:]) / 2.0
    p0_global = float(y.mean())
    p_base_var = float(np.average(p_smooth, weights=(total + 1e-9)))  # per-variable baseline

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    return {
        "edges": edges,
        "centers": centers,
        "support": total,
        "p_raw": p_smooth,        # smoothed but not tail-penalized
        "p0_global": p0_global,   # dataset-wide baseline
        "p_base_var": p_base_var, # per-variable baseline
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

def auc_weight(x: pd.Series, y: pd.Series) -> float:
    """AUC-like separation → 0..1 (with floor)."""
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x,y = x[mask], y[mask]
    if len(x) < 40 or y.nunique()!=2: return 0.0
    r = x.rank(method="average")
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return 0.0
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    sep = abs(auc-0.5)*2.0  # 0..1
    return float(max(0.05, sep))

def value_to_prob(var_key: str, model: Dict[str,Any], x_val: float) -> float:
    """Map raw value to FT probability via rank lookup on the finalized curve (after tail & stretch)."""
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

# ---------- Priors, anchors & guardrails ----------
def si_directional_prior(x: float) -> Optional[float]:
    if not np.isfinite(x): return None
    # 0%→~0.25, 10%→~0.57, 20%→~0.73, 30%→~0.82
    k = 0.25
    x0 = 10.0
    base = 0.25 + 0.65 / (1 + math.exp(-k * (x - x0)))
    return float(np.clip(base, 0.20, 0.90))

def blend_with_prior(var_key: str, x: float, p_learned: float) -> float:
    if var_key == "si_pct":
        prior = si_directional_prior(x)
        if prior is not None:
            p_learned = 0.40*prior + 0.60*p_learned
    return float(np.clip(p_learned, 1e-6, 1-1e-6))

# PM% & ATR special-case anchors (optional; generic anchor applies to all variables too)
ANCHORS = {
    "pm_pct_daily": (8.0, 25.0),   # %
    "pm_pct_pred":  (8.0, 25.0),   # %
}

def anchor_pm_percent(var_key: str, p: float, x_val: float, pb: float) -> float:
    band = ANCHORS.get(var_key)
    if not band or not np.isfinite(x_val): 
        return p
    lo, hi = band
    if lo <= x_val <= hi:
        return float(max(p, min(0.95, pb + 0.10)))  # +10pp lift target cap
    return p

def anchor_atr(p: float, x: float, pb: float) -> float:
    if not np.isfinite(x): return p
    if 0.15 <= x <= 0.40:
        return max(p, min(0.92, pb + 0.10))
    if x < 0.10:
        return min(p, 0.50)
    return p

# Learned sweet band per variable (generic)
def _learn_anchor_band(centers: np.ndarray, p_curve: np.ndarray, pb: float, margin: float = 0.05):
    above = p_curve >= (pb + margin)
    if not np.any(above):
        return None
    idx = np.where(above)[0]
    return float(centers[idx.min()]), float(centers[idx.max()])

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

def apply_anchor_generic(var_key: str, model: dict, x: float, p: float, lift: float = 0.10) -> float:
    band = model.get("anchor_band", None)
    if band is None: 
        return p
    r = _rank_from_value(model, x)
    if r is None: 
        return p
    lo, hi = band
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return p
    if r < lo or r > hi:
        weight = 0.0
    else:
        t = (r - lo) / max(1e-9, (hi - lo))
        weight = 0.5 - 0.5 * math.cos(math.pi * t)
    if weight <= 0.0:
        return p
    pb = float(model.get("p_base_var", 0.5))
    target = min(0.95, pb + lift)
    p_new = p + weight * (target - p)
    return float(np.clip(p_new, 1e-6, 1-1e-6))

def apply_rule_overrides(var_key: str, x: float, p: float) -> float:
    # --- GAP ---
    if var_key == "gap_pct" and np.isfinite(x):
        if x > 350:
            p = min(p, 0.35)   # hard cap → risk zone
        elif x > 240:
            p = min(p, 0.50)   # caution cap
    # --- RVOL ---
    if var_key == "rvol" and np.isfinite(x):
        if x > 4000:
            p = min(p, 0.38)
        elif x > 3000:
            p = min(p, 0.52)
    # --- PM $Vol / MC ---
    if var_key == "pmmc_pct" and np.isfinite(x):
        if x > 200:
            p = min(p, 0.40)
        elif x > 100:
            p = min(p, 0.55)
    # --- PM Float Rotation ---
    if var_key == "fr_x" and np.isfinite(x):
        if x > 10:
            p = min(p, 0.45)
        elif x > 5:
            p = min(p, 0.58)
    return float(np.clip(p, 1e-6, 1-1e-6))

# ============================== Upload & Learn (Main Pane) ==============================
st.markdown('<div class="section-title">Upload workbook (sheet: PMH BO Merged)</div>', unsafe_allow_html=True)
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

                # column mapping (simple robust)
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

                    # derived from DB
                    if {"pm_vol_m","float_m"}.issubset(df.columns):
                        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
                    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
                        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"]
                    # predicted daily fallback curve
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

                    # learn models per variable (bins and stretch fixed; apply tail penalty here)
                    candidates = [
                        "gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m",
                        "fr_x","pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"
                    ]
                    models, weights = {}, {}
                    for v in candidates:
                        if v in df.columns:
                            m = rank_hist_model(df[v], y)
                            if m is not None:
                                centers = m["centers"]
                                p_base_var = m["p_base_var"]
                                # tail-penalize var-specifically
                                p_tp = tail_penalize(v, centers, m["p_raw"], p_base_var, tail_strength)
                                # stretch to ε=0.10..0.90 (no baseline blend)
                                p_use = stretch_curve_to_unit(p_tp, base_p=p_base_var)
                                m["p"] = p_use
                                m["p_tp"] = p_tp

                                # learn & store sweet-band anchor (rank space) for this variable
                                band = _learn_anchor_band(centers, p_use, p_base_var, margin=0.05)
                                m["anchor_band"] = band

                                models[v] = m

                                # weights from separation & amplitude
                                w_sep = auc_weight(df[v], y)
                                amp   = float(np.nanstd(p_use))
                                weights[v] = max(0.05, min(1.0, 0.7*w_sep + 0.3*(amp*4)))

                    # normalize weights
                    if weights:
                        s = sum(weights.values()) or 1.0
                        weights = {k: v/s for k, v in weights.items()}

                    st.session_state.MODELS  = models
                    st.session_state.WEIGHTS = weights
                    st.success(f"Learned {len(models)} variables.")
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
            rvol     = input_float("RVOL @ BO", 0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
            # Dilution slider under PM $ volume
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

        # Odds stacking with per-variable baselines, priors, anchors, guardrails
        parts: Dict[str, Dict[str, float]] = {}
        z_sum = 0.0
        for k, x in var_vals.items():
            mdl = models.get(k)
            if mdl is None:
                continue
            p = value_to_prob(k, mdl, x)        # learned curve
            pb = float(mdl.get("p_base_var", 0.5))

            # directional prior(s)
            p = blend_with_prior(k, x, p)

            # generic learned anchor (sweet band per variable)
            p = apply_anchor_generic(k, mdl, x, p, lift=0.10)

            # special-case anchors (keep if you want extra domain nudge)
            if k == "atr_usd":
                p = anchor_atr(p, x, pb)
            p = anchor_pm_percent(k, p, x, pb)

            # hard guardrails for exhaustion
            p = apply_rule_overrides(k, x, p)

            w = float(weights.get(k, 0.0))
            parts[k] = {"x": x, "p": p, "w": w, "pb": pb}
            p_clip = float(np.clip(p, 1e-6, 1-1e-6))
            z_sum += w * math.log(p_clip/(1-p_clip))

        # dilution penalty (binary)
        z_sum += -0.90 * float(dilution_flag)
        z_sum = float(np.clip(z_sum, -12, 12))
        numeric_prob = 1.0 / (1.0 + math.exp(-z_sum))

        final_score = float(np.clip(numeric_prob*100.0, 0.0, 100.0))
        odds_name = odds_label(final_score)
        level = grade(final_score)
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if final_score >= 70 else
            '<span class="pill pill-warn">Constructive</span>' if final_score >= 55 else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )

        # Checklist with Good/Caution/Risk relative to per-variable baseline (fixed lift)
        name_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL @ BO","si_pct":"Short Interest %",
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
        b.metric("Final Score", f"{l.get("FinalScore",0):.2f}")
        c.metric("Grade", l.get("Level","—"))
        d.metric("Odds", l.get("Odds","—"))
        e.metric("PredVol (M)", f"{l.get("PredVol_M",0):.2f}")

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
    st.markdown('<div class="section-title">Learned Curves (rank-space FT rate, per-variable baselines)</div>', unsafe_allow_html=True)
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
                centers = m["centers"]; p = m["p"]; pb = m["p_base_var"]
                ax.plot(centers, p, lw=2)
                if show_baseline: ax.axhline(pb, ls="--", lw=1)
                ax.set_title(var, fontsize=11)
                ax.set_xlabel("Rank (percentile)", fontsize=10)
                ax.set_ylabel("P(FT)", fontsize=10)
                ax.tick_params(labelsize=9)
            for j in range(i+1, nrows*ncols):
                fig.delaxes(axes[j//ncols, j % ncols])
            st.pyplot(fig, clear_figure=True)
        else:
            var = sel_curve_var
            m = models.get(var)
            if m is None:
                st.warning(f"No curve learned for '{var}'.")
            else:
                centers = m["centers"]; p = m["p"]; pb = m["p_base_var"]
                fig, ax = plt.subplots(figsize=(6.4, 3.4))
                ax.plot(centers, p, lw=2)
                if show_baseline: ax.axhline(pb, ls="--", lw=1)
                ax.set_xlabel("Rank (percentile of variable)", fontsize=10)
                ax.set_ylabel("P(FT | rank)", fontsize=10)
                ax.set_title(f"{var} — FT curve (baseline dashed)", fontsize=11)
                ax.tick_params(labelsize=9)
                st.pyplot(fig, clear_figure=True)
