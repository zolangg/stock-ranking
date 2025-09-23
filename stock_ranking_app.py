# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking — PMH BO Merged", layout="wide")
st.title("Premarket Stock Ranking — Data-derived (PMH BO Merged)")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color:#374151; }
  ul { margin: 6px 0 0 0; padding-left: 18px; }
  li { margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# ============================== Sidebar (settings) ==============================
st.sidebar.header("Learning / Scoring")

sb_bins     = st.sidebar.slider("Histogram bins (rank space)", 0, 120, 60, 5)
sb_lift     = st.sidebar.slider("Lift over baseline for sweet/risk", 0.02, 0.25, 0.08, 0.01)
sb_support  = st.sidebar.slider("Min samples per bin", 2, 50, 6, 1)
sb_gapmerge = st.sidebar.slider("Merge gaps ≤ (rank width)", 0.00, 0.10, 0.02, 0.005)
sb_minspan  = st.sidebar.slider("Min interval span (rank)", 0.005, 0.10, 0.03, 0.005)

st.sidebar.markdown("---")
tail_strength = st.sidebar.slider("High-tail penalty (exhaustion vars)", 0.0, 1.0, 0.55, 0.05)
weight_dampen = st.sidebar.slider("Weight dampening", 0.0, 1.0, 0.20, 0.05)

st.sidebar.markdown("---")
show_baseline = st.sidebar.checkbox("Curves: show baseline", True)
plot_all_curves = st.sidebar.checkbox("Curves: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or plot one variable",
    ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","catalyst"]
)

st.sidebar.markdown("---")
stretch_on   = st.sidebar.checkbox("Stretch curves to full 0–100%", True)
stretch_eps  = st.sidebar.slider("Stretch floor/ceiling ε", 0.0, 0.10, 0.01, 0.005,
                                 help="Min becomes ε, max becomes 1−ε")
stretch_blend= st.sidebar.slider("Blend with baseline γ", 0.0, 1.0, 0.20, 0.05,
                                 help="0 = pure stretched curve; 1 = baseline only")

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}   # var -> model
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {} # var -> weight
if "BASEP"  not in st.session_state: st.session_state.BASEP = 0.5
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
    if abs(v) >= 1_000_000: return f"{v/1_000_000:,.1f}M" # e.g. 1.2M
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 1:    return f"{v:.2f}"
    if abs(v) > 0.001: return f"{v:.3f}" # for very small numbers
    return f"{v:.0e}" # scientific for extremely small

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
    for cand in candidates: # second pass for partial matches
        n = _norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None

# ================= Predicted Day Volume (for checklist) =================
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """ ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$) """
    e = 1e-6 # small epsilon to prevent log(0)
    mc  = max(float(mcap_m or 0.0), e)
    gp  = max(float(gap_pct or 0.0), e) / 100.0
    atr = max(float(atr_usd or 0.0), e)
    ln_y = 3.1435 + 0.1608*math.log(mc) + 0.6704*math.log(gp) - 0.3878*math.log(atr)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0 or not np.isfinite(pred_m): return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# ================= Rank-hist learning (DB-derived) =================
EXHAUSTION_VARS = {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}

def moving_average(y: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1: return y
    pad = w//2
    # Use 'reflect' mode for padding to prevent edge effects
    ypad = np.pad(y, (pad,pad), mode='reflect')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')

def stretch_curve_to_unit(p: np.ndarray, base_p: float,
                          eps: float = 0.01, gamma: float = 0.20) -> np.ndarray:
    """
    Linearly stretch a probability curve so its min→eps and max→1-eps,
    then softly blend back toward the baseline by gamma.
    """
    p = np.asarray(p, dtype=float)
    pmin, pmax = float(np.nanmin(p)), float(np.nanmax(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        # degenerate -> just baseline
        return np.full_like(p, base_p)

    # 1) linear stretch to [eps, 1-eps]
    scale = max(1e-9, (pmax - pmin)) # Ensure scale is not zero
    p_stretched = eps + (1.0 - 2.0*eps) * (p - pmin) / scale

    # 2) optional soft blend back to baseline
    if gamma > 0:
        p_final = (1.0 - gamma) * p_stretched + gamma * base_p
    else:
        p_final = p_stretched

    # keep numeric sanity
    return np.clip(p_final, 1e-6, 1.0 - 1e-6)

def rank_hist_model(x: pd.Series, y: pd.Series, bins: int) -> Optional[Dict[str,Any]]:
    """Learn FT-rate curve in rank space with smoothing and interpolation."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 40 or y.nunique() != 2: return None

    ranks = x.rank(pct=True)   # 0..1
    edges = np.linspace(0,1,bins+1)
    # Ensure idx stays within bounds for bincount
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, bins-1)

    total = np.bincount(idx, minlength=bins)
    ft    = np.bincount(idx[y==1], minlength=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(total>0, ft/total, np.nan)

    # Interpolate gaps then smooth
    p_series = pd.Series(p).interpolate(method='linear', limit_direction="both")
    p_fill = p_series.fillna(p_series.mean()).to_numpy() # Fill remaining NaNs with mean
    p_smooth = moving_average(p_fill, w=5)
    
    # Define quantiles before returning
    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    # Store the raw smoothed curve. Stretching/tail-penalizing will happen later.
    return {
        "edges": edges,
        "centers": (edges[:-1]+edges[1:])/2,
        "p_raw": p_smooth.copy(),            # <— store raw smoothed curve
        "support": total,
        "p0": float(y.mean()),     # baseline
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

def auc_weight(x: pd.Series, y: pd.Series) -> float:
    """AUC-like separation → 0..1 (with floor), then dampened later."""
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x,y = x[mask], y[mask]
    if len(x) < 40 or y.nunique()!=2: return 0.0
    
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return 0.0

    r = x.rank(method="average") # Use average rank for AUC calculation
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    
    # Ensure AUC is between 0 and 1, then calculate separation
    auc = np.clip(auc, 0, 1)
    sep = abs(auc-0.5)*2.0  # 0..1

    # also scale by curve amplitude so flat curves don’t get weight (this part will be done outside)
    # The amplitude consideration will be integrated into the main learning loop
    return float(max(0.05, sep)) # floor at 0.05 to give some base weight

def tail_penalize(var_key: str, centers: np.ndarray, p: np.ndarray, p0: float, strength: float) -> np.ndarray:
    """Penalize extreme high ranks for exhaustion variables."""
    if var_key not in EXHAUSTION_VARS or strength <= 0: return p
    # ramp penalty in top tail (rank > 0.90 → up to strength)
    tail = np.clip((centers - 0.90) / 0.08, 0, 1)  # 0→1 over 0.90..0.98
    penalty_factor = 1.0 - strength * tail
    return p0 + (p - p0) * penalty_factor

def find_intervals_adaptive(var_key: str, model: Dict[str,Any], lift: float, min_support: int,
                            gap_merge: float, min_span: float, strength: float) -> Tuple[List[Tuple[float,float]],List[Tuple[float,float]],float]:
    """
    Extract sweet/risk intervals in rank. If none found, relax lift down to 30% of initial.
    Returns (sweet_val_ranges, risk_val_ranges, used_lift).
    """
    centers = model["centers"]; p_curve = model["p"].copy(); p0 = model["p0"]; sup = model["support"]
    
    # Apply tail penalty to the curve used for interval detection
    p_tail_adjusted = tail_penalize(var_key, centers, p_curve, p0, strength)

    def _merge(mask, gl=gap_merge, ms=min_span):
        idx = np.where(mask)[0]
        if idx.size == 0: return []
        intervals = []
        start_rank = centers[idx[0]]; last_rank = centers[idx[0]]
        for k in idx[1:]:
            if (centers[k] - last_rank) <= gl:
                last_rank = centers[k]
            else:
                if (last_rank - start_rank) >= ms: intervals.append((float(start_rank), float(last_rank)))
                start_rank = last_rank = centers[k]
        if (last_rank - start_rank) >= ms: intervals.append((float(start_rank), float(last_rank)))
        return intervals

    used_lift = lift
    sweet_r, risk_r = [], []
    pr = model["quantiles"]["pr"]; vals = model["quantiles"]["vals"]

    for step in range(6):  # relax up to ~lift*0.8^5 ~ lift*0.32
        s_mask = (p_tail_adjusted >= p0 + used_lift) & (sup >= min_support)
        r_mask = (p_tail_adjusted <= p0 - used_lift) & (sup >= min_support)
        sweet_r = _merge(s_mask)
        risk_r  = _merge(r_mask)
        if sweet_r or risk_r:
            break
        used_lift *= 0.8  # relax lift threshold

    def r2v(lo_rank, hi_rank):
        # Interpolate ranks to values. Ensure bounds are handled.
        lo_v = float(np.interp(np.clip(lo_rank, 0, 1), pr, vals))
        hi_v = float(np.interp(np.clip(hi_rank, 0, 1), pr, vals))
        return lo_v, hi_v

    sweet_v = [r2v(a,b) for a,b in sweet_r]
    risk_v  = [r2v(a,b) for a,b in risk_r]
    return sweet_v, risk_v, used_lift

def value_to_prob(var_key: str, model: Dict[str,Any], x_val: float, tail_strength: float) -> float:
    """Map raw value to FT probability via rank lookup + tail adjust."""
    if model is None or not np.isfinite(x_val): return 0.5
    
    pr, vals = model["quantiles"]["pr"], model["quantiles"]["vals"]
    
    # Clamp x_val to the range of learned values to avoid interpolation errors
    x_val_clamped = np.clip(x_val, vals.min(), vals.max())
    
    # Interpolate x_val to its corresponding rank percentile
    r = float(np.interp(x_val_clamped, vals, pr))

    centers = model["centers"]; p_curve = model["p"]; base = model["p0"]
    
    # Find the corresponding bin in the learned curve
    j = int(np.clip(np.searchsorted(centers, r), 0, len(centers)-1))
    p_local = float(p_curve[j])
    
    # Apply tail penalty to this specific point
    p_adj = tail_penalize(var_key, np.array([centers[j]]), np.array([p_local]), base, tail_strength)[0]
    
    return float(np.clip(p_adj, 1e-6, 1-1e-6)) # Ensure probability is not exactly 0 or 1

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

                # column mapping (robust-ish)
                col_ft    = _pick(raw, ["ft","FT","f/t","follow through"])
                col_gap   = _pick(raw, ["gap %","gap%","premarket gap","gap"])
                col_atr   = _pick(raw, ["atr","atr $","atr$","atr (usd)"])
                col_rvol  = _pick(raw, ["rvol @ bo","rvol","relative volume","relative vol"])
                col_pmvol = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)"])
                col_pmdol = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
                col_float = _pick(raw, ["float m shares","public float (m)","float (m)","float"])
                col_mcap  = _pick(raw, ["marketcap m","market cap (m)","mcap m","mcap"])
                col_si    = _pick(raw, ["si","short interest %","short float %","short interest (float) %"])
                col_cat   = _pick(raw, ["catalyst","news","pr","catalyst present"])
                col_daily = _pick(raw, ["daily vol (m)","day volume (m)","volume (m)","volume at close (m)"])

                if col_ft is None:
                    st.error("No 'FT' (Follow-Through) column found in merged sheet. Please ensure it's labeled clearly like 'FT' or 'ft'.")
                else:
                    df = pd.DataFrame()
                    # Ensure FT is boolean-like (0 or 1) for probability calculation
                    df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce").fillna(0).astype(int).clip(0,1)

                    if col_gap:   df["gap_pct"]  = pd.to_numeric(raw[col_gap],   errors="coerce")
                    if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                    if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                    if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                    if col_pmdol: df["pm_dol_m"] = pd.to_numeric(raw[col_pmdol], errors="coerce")
                    if col_float: df["float_m"]  = pd.to_numeric(raw[col_float], errors="coerce")
                    if col_mcap:  df["mcap_m"]   = pd.to_numeric(raw[col_mcap],  errors="coerce")
                    if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")
                    # Catalyst should be 0 or 1
                    if col_cat:   df["catalyst"] = pd.to_numeric(raw[col_cat],   errors="coerce").fillna(0).clip(0,1)
                    if col_daily: df["daily_vol_m"] = pd.to_numeric(raw[col_daily], errors="coerce")

                    # derived — from DB
                    # Adding checks for division by zero or NaN
                    df["fr_x"] = df.apply(lambda r: r["pm_vol_m"] / r["float_m"] if pd.notna(r["pm_vol_m"]) and r["float_m"] > 0 else np.nan, axis=1)
                    df["pmmc_pct"] = df.apply(lambda r: 100.0 * r["pm_dol_m"] / r["mcap_m"] if pd.notna(r["pm_dol_m"]) and r["mcap_m"] > 0 else np.nan, axis=1)
                    
                    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                        def _pred_row(r):
                            try:
                                return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                            except Exception: # Catch potential errors from log(0)
                                return np.nan
                        pred = df.apply(_pred_row, axis=1)
                        df["pm_pct_pred"] = df.apply(lambda r: 100.0 * r["pm_vol_m"] / r["predicted_vol"] if pd.notna(r["pm_vol_m"]) and r["predicted_vol"] > 0 else np.nan, predicted_vol=pred, axis=1)
                        df["predicted_day_vol_m"] = pred # Keep for potential use/display

                    df = df[df["FT"].notna()] # Filter out rows where FT is NaN
                    y = df["FT"].astype(float)
                    base_p = float(y.mean())

                    # learn models
                    candidates = [
                        "gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m",
                        "float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","catalyst"
                    ]
                    models, weights = {}, {}
                    for v in candidates:
                        if v in df.columns and df[v].notna().sum() > 40: # Only try to learn if enough non-null data
                           m = rank_hist_model(df[v], y, bins=sb_bins)
                           if m is not None:
                                  # 1) tail-penalize on the raw smoothed curve (p_raw)
                                  centers = m["centers"]
                                  p_base  = m["p0"]
                                  p_raw_smoothed = m["p_raw"] # This is the curve coming out of rank_hist_model
                                  p_tp    = tail_penalize(v, centers, p_raw_smoothed, p_base, tail_strength)
                              
                                  # 2) optional stretch to 0..1 (with eps) and blend to baseline
                                  if stretch_on:
                                      p_use = stretch_curve_to_unit(p_tp, base_p=p_base,
                                                                    eps=stretch_eps, gamma=stretch_blend)
                                  else:
                                      p_use = p_tp
                              
                                  # save the working curve (p) and optionally the raw tail-penalized curve
                                  m["p"] = p_use       # <— this is what the rest of the app uses now
                                  m["p_raw_tp"] = p_tp # (optional) keep for debugging/plots if needed
                              
                                  # 3) find intervals using the working curve (p)
                                  # The `find_intervals_adaptive` function needs the 'p' key in the model.
                                  # Since we just assigned `p_use` to `m["p"]`, we can directly pass `m`.
                                  sweet_v, risk_v, used_lift = find_intervals_adaptive(
                                      v, m, lift=sb_lift, min_support=sb_support,
                                      gap_merge=sb_gapmerge, min_span=sb_minspan, strength=tail_strength
                                  )
                                  m["sweet_vals"] = sweet_v
                                  m["risk_vals"]  = risk_v
                                  m["used_lift"]  = used_lift
                              
                                  models[v] = m
                                  # weights calculation
                                  w_sep = auc_weight(df[v], y)
                                  # Use the actual amplitude of the *final working curve* for weighting
                                  amp = float(np.nanstd(p_use))
                                  # Combine separation and amplitude for a more robust weight
                                  weights[v] = max(0.05, min(1.0, 0.7*w_sep + 0.3*(amp*4))) # 'amp*4' is a heuristic to scale amplitude to a similar range as AUC separation

                    # dampen & normalize weights
                    if weights:
                        damp = {k: (1.0 - weight_dampen) * w for k, w in weights.items()}
                        s = sum(damp.values())
                        if s > 0: # Avoid division by zero if all weights are zero
                           weights = {k: v/s for k, v in damp.items()}
                        else:
                           weights = {k: 0.0 for k in weights} # Set all to 0 if sum is 0

                    st.session_state.MODELS  = models
                    st.session_state.WEIGHTS = weights
                    st.session_state.BASEP   = base_p
                    st.success(f"Learned {len(models)} variables · Baseline P(FT) ≈ {base_p:.2f}")
        except Exception as e:
            st.error(f"Learning failed: {e}")
            st.exception(e) # Show full traceback for debugging

# ============================== Tabs ==============================
tab_add, tab_rank, tab_curves, tab_spots = st.tabs(["➕ Add Stock", "📊 Ranking", "📈 Curves", "🎯 Sweet Spots"])

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
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"])
            dilution_flag = st.select_slider("Dilution present?", options=[0,1], value=0,
                                             help="0 = none/negligible, 1 = present/
