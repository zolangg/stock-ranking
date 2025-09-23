import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List, Tuple

# ================= Page & Styles =================
st.set_page_config(page_title="Premarket Stock Ranking — Smooth + Data-Weighted", layout="wide")
st.title("Premarket Stock Ranking — Smooth Curves + Data-Weighted Scoring")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color:#374151; }
  .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
</style>
""", unsafe_allow_html=True)

# ================= Small helpers =================
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try: return float(s)
    except Exception: return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"; default_str = fmt.format(float(value))
    s = st.text_input(label, default_str, key=key, help=help)
    v = _parse_local_float(s)
    if v is None:
        st.caption('<span class="hint">Enter a number, e.g. 5,05</span>', unsafe_allow_html=True)
        return float(value)
    if v < min_value:
        st.caption(f'<span class="hint">Clamped to minimum: {fmt.format(min_value)}</span>', unsafe_allow_html=True)
        v = min_value
    if max_value is not None and v > max_value:
        st.caption(f'<span class="hint">Clamped to maximum: {fmt.format(max_value)}</span>', unsafe_allow_html=True)
        v = max_value
    if ("," in s) or (" " in s) or ("'" in s):
        st.caption(f'<span class="hint">= {fmt.format(v)}</span>', unsafe_allow_html=True)
    return float(v)

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
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def odds_label(score: float) -> str:
    if score >= 85: return "Very High Odds"
    elif score >= 70: return "High Odds"
    elif score >= 55: return "Moderate Odds"
    elif score >= 40: return "Low Odds"
    else: return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 85 else
            "A+"  if score_pct >= 80 else
            "A"   if score_pct >= 70 else
            "B"   if score_pct >= 60 else
            "C"   if score_pct >= 45 else "D")

# ================= Stable math helpers =================
def stable_sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def stable_logit(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p) - np.log(1.0 - p)

def prob_from_logodds(z):
    z = float(np.clip(z, -50.0, 50.0))
    return 1.0 / (1.0 + math.exp(-z))

# ================= Prediction model (Predicted Day Vol) =================
def predict_day_volume_m(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Y in millions of shares.
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    if mc <= 0 or gp <= 0 or atr <= 0: return float("nan")
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float) -> Tuple[float,float]:
    if not np.isfinite(pred_m) or pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# ================= State =================
if "CURVES" not in st.session_state: st.session_state.CURVES = {}
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash); st.session_state.flash = None

# ================= Upload (MAIN PANE, full width) =================
st.markdown('<div class="section-title">Upload your database</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Excel (.xlsx) — prefers 'PMH BO Merged' (with FT label). Fallback: 'PMH BO FT' + 'PMH BO Fail'.", type=["xlsx"])

c1, c2, c3, c4 = st.columns([0.28, 0.24, 0.24, 0.24])
with c1:
    merged_sheet = st.text_input("Merged sheet (preferred)", "PMH BO Merged")
with c2:
    ft_sheet = st.text_input("FT sheet (fallback)", "PMH BO FT")
with c3:
    fail_sheet = st.text_input("Fail sheet (fallback)", "PMH BO Fail")
with c4:
    learn_btn = st.button("📚 Learn curves + weights", use_container_width=True)

import numpy as np
import pandas as pd

# --- Rank-smoothing (Gaussian kernel on rank space) ---
def _rank_smooth(x, y, bandwidth=0.06, grid_points=201, min_rows=20):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(float), y[mask].to_numpy(float)
    if x.size < min_rows:
        return None
    ranks = pd.Series(x).rank(pct=True).to_numpy()
    grid = np.linspace(0, 1, grid_points)
    h = float(bandwidth)
    p_grid = np.empty_like(grid)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((ranks - g)/h)**2)
        sw = w.sum()
        p_grid[i] = (w*y).sum()/sw if sw > 0 else np.nan
    # fill ends
    p_grid = pd.Series(p_grid).interpolate(limit_direction="both").fillna(y.mean()).to_numpy()

    # local support via rank histogram
    bins = np.linspace(0,1,51)
    counts, _ = np.histogram(ranks, bins=bins)
    bin_idx = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(counts)-1)
    support = counts[bin_idx]
    return {"grid":grid, "p_grid":p_grid, "p0":float(y.mean()), "support":support}

# --- Merge mask runs allowing tiny gaps ---
def _intervals_with_gap(mask, x, max_gap=0.02, min_span=0.03):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    ivals, start, last = [], idx[0], idx[0]
    for k in idx[1:]:
        if (x[k] - x[last]) <= max_gap:
            last = k
        else:
            lo, hi = x[start], x[last]
            if (hi - lo) >= min_span:
                ivals.append((float(lo), float(hi)))
            start = last = k
    lo, hi = x[start], x[last]
    if (hi - lo) >= min_span:
        ivals.append((float(lo), float(hi)))
    return ivals

# --- Tail penalty for exhaustion-prone variables (damp lift beyond ~95th rank) ---
_EXHAUSTION_VARS = {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}

def _apply_tail_penalty(var_key, grid, p_grid, p0):
    if var_key not in _EXHAUSTION_VARS:
        return p_grid
    # linear ramp from rank 0.90->0.98; keep at most 30% of the lift at the very top
    tail = np.clip((grid - 0.90) / 0.08, 0, 1)
    penalty = 1.0 - 0.7 * tail
    return p0 + (p_grid - p0) * penalty

# --- Sweet/Risk finder (lift-over-baseline + support + gap-merge + tail penalty) ---
def find_sweet_risk_ranges(x, y, var_key,
                           lift_thr=0.06,     # how much over baseline to call "sweet"
                           min_support=3,     # minimum local obs
                           max_gap=0.02,      # allow gaps in rank
                           min_span=0.03):    # minimum span in rank
    res = _rank_smooth(x, y, bandwidth=0.06, grid_points=201, min_rows=20)
    if res is None:
        return {"sweet_rank":[], "sweet_vals":[], "risk_rank":[], "risk_vals":[], "p0":np.nan}

    grid, p, p0, sup = res["grid"], res["p_grid"], res["p0"], res["support"]

    # penalize exhaustion tails for certain vars
    p = _apply_tail_penalty(var_key, grid, p, p0)

    # define sweet/risk by lift vs baseline and support
    sweet_mask = (p >= p0 + lift_thr) & (sup >= min_support)
    risk_mask  = (p <= p0 - lift_thr) & (sup >= min_support)

    sweet_rank = _intervals_with_gap(sweet_mask, grid, max_gap=max_gap, min_span=min_span)
    risk_rank  = _intervals_with_gap(risk_mask,  grid, max_gap=max_gap, min_span=min_span)

    # map rank intervals -> original value intervals
    xv = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    def r2v(ivals):
        out=[]
        for lo,hi in ivals:
            lo_v = float(np.nanquantile(xv, lo))
            hi_v = float(np.nanquantile(xv, hi))
            out.append((lo_v, hi_v))
        return out

    sweet_vals = r2v(sweet_rank)
    risk_vals  = r2v(risk_rank)
    return {"sweet_rank":sweet_rank, "sweet_vals":sweet_vals,
            "risk_rank":risk_rank,   "risk_vals":risk_vals, "p0":p0}
          
# ================= Sidebar (utility controls only) =================
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Predicted Day Vol log-σ", 0.10, 1.50, 0.60, 0.01)

# ---- Model tuning (put in sidebar or details expander) ----
st.sidebar.header("Model tuning")
curve_sharpness = st.sidebar.slider(
    "Curve sharpness (bandwidth scale)", 0.30, 1.50, 0.70, 0.05,
    help="Lower = sharper curves (less smoothing)."
)
curve_temperature = st.sidebar.slider(
    "Curve temperature (probability contrast)", 0.50, 1.20, 0.70, 0.05,
    help="Lower (<1) increases contrast; higher (>1) flattens."
)
anchor_target = st.sidebar.slider(
    "Sweet-spot anchor p*", 0.60, 0.90, 0.75, 0.01,
    help="Target probability at the prior 'center' for hump-shaped variables."
)

# ================= Mapping & cleaning =================
_SYNONYMS = {
    "gap_pct":       ["gap %","gap%","gap pct","premarket gap","pm gap"],
    "pm_vol_m":      ["pm vol (m)","pm volume (m)","premarket vol (m)","pm shares","pm vol m"],
    "pm_dol_m":      ["pm $vol (m)","pm $ vol (m)","pm dollar vol (m)","premarket $vol (m)","pm $volume"],
    "daily_vol_m":   ["daily vol (m)","day vol (m)","volume (m)","vol (m)"],
    "float_m":       ["float m shares","public float (m)","float (m)","float(m)","float m"],
    "mcap_m":        ["marketcap m","market cap (m)","market cap m","mcap (m)","mcap"],
    "rvol":          ["rvol @ bo","rvol","relative volume","rvol bo"],
    "atr_usd":       ["atr","atr $","atr (usd)","atr $/day","atr$"],
    "si_pct":        ["short interest %","si %","short %","short float %","short interest (float) %","shortinterest %"],
    "catalyst":      ["catalyst","news","pr","has news","with news","catalyst flag","news flag","pr flag"],
    "label":         ["ft","label","outcome","y","success","followthrough","follow through","ft flag"],
}

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.replace("$","").replace("%","").replace("’","").replace("'","")
    return s

def _pick(df: pd.DataFrame, logical_key: str, *prefer_exact: str) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {c: _norm(c) for c in cols}
    # prefer exact
    for n in prefer_exact:
        for c in cols:
            if norm_map[c] == _norm(n): return c
    # synonyms
    for cand in _SYNONYMS.get(logical_key, []):
        n_cand = _norm(cand)
        for c in cols:
            if norm_map[c] == n_cand: return c
        for c in cols:
            if n_cand in norm_map[c]: return c
    # tokens
    tokens = re.split(r"[^\w]+", logical_key); tokens = [t for t in tokens if t]
    for c in cols:
        nm = norm_map[c]
        if all(t in nm for t in tokens): return c
    return None

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip() for c in df.columns]; return df

def _numify(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(str).str.replace(",","", regex=False).str.replace("'","", regex=False).str.strip()
    s = s.str.replace("%","", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _ensure_gap_pct(series: pd.Series) -> pd.Series:
    s = _numify(series)
    if s.dropna().lt(5).mean() > 0.6:  # values like 0.8 instead of 80
        s = s * 100.0
    return s

def build_numeric_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_cols(df)
    col_gap   = _pick(df, "gap_pct")
    col_pmvol = _pick(df, "pm_vol_m")
    col_pmdol = _pick(df, "pm_dol_m")
    col_daily = _pick(df, "daily_vol_m")
    col_float = _pick(df, "float_m")
    col_mcap  = _pick(df, "mcap_m")
    col_rvol  = _pick(df, "rvol")
    col_atr   = _pick(df, "atr_usd")
    col_si    = _pick(df, "si_pct")
    col_cat   = _pick(df, "catalyst")
    col_lab   = _pick(df, "label")

    out = pd.DataFrame()
    if col_gap:   out["gap_pct"] = _ensure_gap_pct(df[col_gap])
    if col_atr:   out["atr_usd"] = _numify(df[col_atr])
    if col_rvol:  out["rvol"]    = _numify(df[col_rvol])
    if col_si:    out["si_pct"]  = _numify(df[col_si])
    if col_pmvol: out["pm_vol_m"]= _numify(df[col_pmvol])
    if col_pmdol: out["pm_dol_m"]= _numify(df[col_pmdol])
    if col_float: out["float_m"] = _numify(df[col_float])
    if col_mcap:  out["mcap_m"]  = _numify(df[col_mcap])
    if col_daily: out["daily_vol_m"] = _numify(df[col_daily])
    if col_cat:
        cat_raw = df[col_cat].astype(str).str.strip().str.lower()
        out["catalyst"] = cat_raw.isin(["1","true","yes","y","t"]).astype(float)
    if col_lab:
        lab_raw = df[col_lab].astype(str).str.strip().str.lower()
        out["_y"] = lab_raw.isin(["1","true","yes","y","t","ft"]).astype(int)

    # Derived
    if "pm_vol_m" in out and "float_m" in out:
        out["fr_x"] = out["pm_vol_m"] / out["float_m"]                    # PM Float Rotation ×
    if "pm_dol_m" in out and "mcap_m" in out:
        out["pmmc_pct"] = 100.0 * out["pm_dol_m"] / out["mcap_m"]         # PM $Vol / MC %
    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(out.columns):
        pred = []
        for mc, gp, atr in zip(out["mcap_m"], out["gap_pct"], out["atr_usd"]):
            pred.append(predict_day_volume_m(mc, gp, atr))
        out["pred_vol_m"] = pred
        out["pm_pct_pred"] = 100.0 * out["pm_vol_m"] / out["pred_vol_m"]  # PM Vol % of Predicted
    if "pm_vol_m" in out and "daily_vol_m" in out:
        out["pm_pct_daily"] = 100.0 * out["pm_vol_m"] / out["daily_vol_m"]

    return out

def reveal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    dfc = _clean_cols(df); rows = []
    for key in ["label","catalyst","gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","daily_vol_m"]:
        col = _pick(dfc, key); rows.append({"Logical": key, "Matched Column": col if col else "(not found)"})
    return pd.DataFrame(rows)

# ================= Smooth Curves (Kernel) =================
MIN_ROWS = 30

def _kernel_prob(x_train, y_train, x_eval, bandwidth):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval  = np.asarray(x_eval, dtype=float)
    h = max(1e-6, float(bandwidth))
    out = np.zeros_like(x_eval, dtype=float)
    for i, xe in enumerate(x_eval):
        w = np.exp(-0.5 * ((xe - x_train)/h)**2)
        sw = w.sum()
        out[i] = (w * y_train).sum() / sw if sw > 0 else np.nan
    return out

def fit_curve(series: pd.Series, labels: pd.Series, force_log: bool=False,
              sharpness: float = 0.70, temperature: float = 0.70,
              anchor_target: float = 0.75, var_name: str = ""):
    x_raw = pd.to_numeric(series, errors="coerce").to_numpy()
    y_raw = pd.to_numeric(labels, errors="coerce").to_numpy()
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]; y_raw = y_raw[mask]
    if len(x_raw) < MIN_ROWS: return None

    # force log for SI, auto-log for skewed positive
    use_log = False
    if force_log and (x_raw > 0).all(): use_log = True
    elif (x_raw > 0).all() and abs(pd.Series(x_raw).skew()) > 0.75: use_log = True

    if use_log:
        pos = x_raw > 0; x_raw = x_raw[pos]; y_raw = y_raw[pos]
        if len(x_raw) < MIN_ROWS: return None
        x_t = np.log(x_raw)
    else:
        x_t = x_raw.copy()

    # Base bandwidth (Silverman-ish), scaled by sharpness
    std = np.std(x_t); n = len(x_t)
    h0 = 0.6 * std * (n ** (-1/5)) if std > 0 else 0.3
    h = max(1e-6, h0 * float(sharpness))

    lo = np.nanpercentile(x_t, 2); hi = np.nanpercentile(x_t, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi: return None
    grid = np.linspace(lo, hi, 201)

    # kernel regression on transformed axis
    def _kernel_prob(x_train, y_train, x_eval, bandwidth):
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_eval  = np.asarray(x_eval, dtype=float)
        h = max(1e-6, float(bandwidth))
        out = np.zeros_like(x_eval, dtype=float)
        for i, xe in enumerate(x_eval):
            w = np.exp(-0.5 * ((xe - x_train)/h)**2)
            sw = w.sum()
            out[i] = (w * y_train).sum() / sw if sw > 0 else np.nan
        return out

    p_grid_raw = _kernel_prob(x_t, y_raw, grid, h)
    p_grid_raw = np.where(np.isfinite(p_grid_raw), p_grid_raw, 0.5)
    p_grid_raw = np.clip(p_grid_raw, 1e-3, 1-1e-3)

    # ---- calibration in logit space: temperature + anchor to prior center ----
    z_grid = stable_logit(p_grid_raw)
    T = float(max(0.5, min(1.2, temperature)))  # safety
    zT = z_grid / T

    # anchor: pick a center 'c' from PRIORS (for hump vars); otherwise no shift
    bias = 0.0
    if var_name in PRIORS and PRIORS[var_name].get("kind") == "hump":
        c = PRIORS[var_name].get("c", None)
        if c is not None:
            # find grid x (in original scale) closest to c
            grid_x = np.exp(grid) if use_log else grid
            idx = int(np.argmin(np.abs(grid_x - float(c))))
            target = float(np.clip(anchor_target, 0.55, 0.9))
            b = stable_logit(target) - zT[idx]
            bias = float(b)

    z_cal = zT + bias
    p_grid_cal = 1.0 / (1.0 + np.exp(-np.clip(z_cal, -50, 50)))
    p_grid_cal = np.clip(p_grid_cal, 1e-3, 1-1e-3)

    def predict(x_new):
        x_new = np.array(x_new, dtype=float)
        if use_log:
            bad = ~np.isfinite(x_new) | (x_new <= 0)
            z = np.full_like(x_new, np.nan, dtype=float); ok = ~bad
            z[ok] = np.log(x_new[ok])
        else:
            z = x_new
        p = _kernel_prob(x_t, y_raw, z, h)
        p = np.where(np.isfinite(p), p, 0.5)
        p = np.clip(p, 1e-3, 1-1e-3)
        # apply the same calibration: temperature + bias
        z_ = stable_logit(p) / T + bias
        p_ = 1.0 / (1.0 + np.exp(-np.clip(z_, -50, 50)))
        return np.clip(p_, 1e-3, 1-1e-3)

    q10, q25, q50, q75, q90 = [float(pd.Series(x_raw).quantile(p)) for p in [0.10,0.25,0.50,0.75,0.90]]
    return {
        "use_log": use_log,
        "grid_t": grid,
        "p_grid": p_grid_cal,     # calibrated curve for plotting
        "predict": predict,       # calibrated predictor
        "n": len(x_t),
        "bandwidth_t": h,
        "temperature": T,
        "bias": bias,
        "q": {"p10": q10, "p25": q25, "p50": q50, "p75": q75, "p90": q90}
    }

# ================= Priors & Guardrails =================
PRIORS = {
    "gap_pct":     {"kind":"hump", "c":110.0, "w":45.0},    # ~70–180 sweet; workable until ~250
    "rvol":        {"kind":"hump", "c":9.0,   "w":6.0},     # this operates in raw scale; still acts as soft prior
    "pm_dol_m":    {"kind":"hump", "c":11.0,  "w":8.0},     # $7–15M sweet
    "pm_pct_pred": {"kind":"hump", "c":18.0,  "w":10.0},    # 10–20% sweet
    "pmmc_pct":    {"kind":"hump", "c":12.0,  "w":10.0},    # ~8–30% okay; very high = dilution/exhaustion risk
    "atr_usd":     {"kind":"hump", "c":0.30,  "w":0.18},    # ≥0.15; 0.2–0.4 typical
    "float_m":     {"kind":"low_better"},
    "mcap_m":      {"kind":"low_better"},
    "si_pct":      {"kind":"high_better", "pivot": math.log(15.0), "scale": 0.35},  # **log SI**, pivot ~15%
    "fr_x":        {"kind":"hump", "c":0.25,  "w":0.15},
    "catalyst":    {"kind":"high_better", "pivot":0.8, "scale":0.25},
}
DEFAULT_PRIOR_BLEND = 0.30
PRIOR_BLEND_MAP = {"si_pct":0.65, "catalyst":0.60}

DOMAIN_OVERRIDES = {
    "gap_pct":     {"low": 70.0,   "high": 300.0},
    "rvol":        {"low": 3.0,    "high": 4000.0},
    "pm_dol_m":    {"low": 7.0,    "high": 150.0},
    "pm_vol_m":    {"low": None,   "high": None},
    "pm_pct_pred": {"low": 8.0,    "high": 35.0},
    "pmmc_pct":    {"low": 8.0,    "high": 100.0},
    "atr_usd":     {"low": 0.15,   "high": 1.50},
    "float_m":     {"low": 1.0,    "high": 25.0},
    "mcap_m":      {"low": None,   "high": 300.0},
    "si_pct":      {"low": 0.5,    "high": 35.0},
    "fr_x":        {"low": 1e-3,   "high": 1.00},
    "catalyst":    {"low": None,   "high": None},
}

# softened tails; still penalize extreme exhaustion
BASE_PEN_STRENGTH = {
    "gap_pct":     (0.30, 0.55),
    "rvol":        (0.25, 0.65),
    "pm_dol_m":    (0.20, 0.60),
    "pm_vol_m":    (0.12, 0.30),
    "pm_pct_pred": (0.20, 0.45),
    "pmmc_pct":    (0.18, 0.55),
    "atr_usd":     (0.50, 0.30),
    "float_m":     (0.30, 0.40),
    "mcap_m":      (0.10, 0.30),
    "si_pct":      (0.10, 0.25),
    "fr_x":        (0.12, 0.40),
    "catalyst":    (0.00, 0.00),
}

def _prior_p(var: str, x: float) -> float:
    if x is None or not np.isfinite(x): 
        return 0.5
    spec = PRIORS.get(var, {"kind":"flat"})
    kind = spec.get("kind","flat")

    if kind == "hump":
        c = float(spec.get("c", 0.0)); w = max(1e-6, float(spec.get("w", 1.0)))
        t = (x - c) / w
        t = np.clip(t, -1e3, 1e3)
        p = np.exp(-0.5 * (t**2))
        return float(np.clip(0.2 + 0.6 * p, 1e-3, 1-1e-3))

    if kind == "low_better":
        z = -(x - 12.0) / 4.0
        s = stable_sigmoid(z)
        return float(np.clip(0.2 + 0.6 * s, 1e-3, 1-1e-3))

    if kind == "high_better":
        pivot = float(spec.get("pivot", 0.0))
        scale = max(1e-6, float(spec.get("scale", 1.0)))
        z = (x - pivot) / scale
        s = stable_sigmoid(z)
        return float(np.clip(0.2 + 0.6 * s, 1e-3, 1-1e-3))

    return 0.5

def blend_with_prior(var: str, p_learned: float, x: float) -> float:
    alpha = PRIOR_BLEND_MAP.get(var, DEFAULT_PRIOR_BLEND)
    pr = _prior_p(var, x)
    p = (1 - alpha) * p_learned + alpha * pr
    return float(np.clip(p, 1e-3, 1-1e-3))

def tail_penalty_bidirectional_soft(x, low, high, k_low, k_high, reliability, deadzone_frac=0.05, alpha=0.60):
    if x is None or not np.isfinite(x): return 0.0
    if low is None and high is None: return 0.0
    if low is not None and high is not None:
        span = (high - low); dz = deadzone_frac * span
        if x < low - dz:
            d = (low - x) / max(1e-6, span/2); return -k_low * reliability * (1 - math.exp(-alpha*d))
        elif x > high + dz:
            d = (x - high) / max(1e-6, span/2); return -k_high * reliability * (1 - math.exp(-alpha*d))
        else:
            return 0.0
    if low is not None and x < low:
        d = (low - x) / max(1e-6, abs(low));  return -k_low * reliability * (1 - math.exp(-alpha*d))
    if high is not None and x > high:
        d = (x - high) / max(1e-6, abs(high)); return -k_high * reliability * (1 - math.exp(-alpha*d))
    return 0.0

# ================= Learn Curves + Weights =================
def learn_all_curves_and_weights(file, merged_sheet: str, ft_sheet: str, fail_sheet: str):
    xls = pd.ExcelFile(file)
    if merged_sheet in xls.sheet_names:
        raw = pd.read_excel(xls, merged_sheet)
        mapping = reveal_mapping(raw)
        num = build_numeric_table(raw)
        if "_y" not in num.columns or num["_y"].nunique() < 2:
            raise ValueError("Merged sheet found but no valid FT label column matched.")
    else:
        if ft_sheet not in xls.sheet_names or fail_sheet not in xls.sheet_names:
            raise ValueError(f"Sheets not found. Available: {xls.sheet_names}")
        ft_raw   = pd.read_excel(xls, ft_sheet)
        fail_raw = pd.read_excel(xls, fail_sheet)
        mapping = pd.concat([reveal_mapping(ft_raw).assign(Sheet="FT"),
                             reveal_mapping(fail_raw).assign(Sheet="Fail")], ignore_index=True)
        ft_num   = build_numeric_table(ft_raw); ft_num["_y"] = 1
        fail_num = build_numeric_table(fail_raw); fail_num["_y"] = 0
        num = pd.concat([ft_num, fail_num], axis=0, ignore_index=True)

    # Learn per-variable curves
    var_list = [
        "gap_pct","atr_usd","rvol","si_pct",
        "pm_vol_m","pm_dol_m","float_m","mcap_m",
        "fr_x","pmmc_pct","pm_pct_pred","catalyst"
    ]
    curves, summary_rows = {}, []
    for v in var_list:
        if v in num.columns:
            model = fit_curve(
                num[v], num["_y"],
                force_log=(v=="si_pct"),
                sharpness=curve_sharpness,
                temperature=curve_temperature,
                anchor_target=anchor_target,
                var_name=v
            )
            if model:
                curves[v] = model
                q = model["q"]
                summary_rows.append({
                    "Variable": v, "n": model["n"], "Used Log": model["use_log"],
                    "BW (t)": round(model["bandwidth_t"],3),
                    "P10": round(q["p10"],3), "P50": round(q["p50"],3), "P90": round(q["p90"],3)
                })
            else:
                summary_rows.append({"Variable": v, "n": 0, "Used Log": "", "BW (t)": "", "P10": "", "P50": "", "P90": ""})
        else:
            summary_rows.append({"Variable": v, "n": 0, "Used Log": "", "BW (t)": "", "P10": "", "P50": "", "P90": ""})
    summary_df = pd.DataFrame(summary_rows)

    # Build per-variable probability matrix for meta-weight learning
    def predict_var(vkey: str, xcol: pd.Series) -> np.ndarray:
        if vkey not in curves: return np.full(len(xcol), 0.5)
        m = curves[vkey]; x = xcol.to_numpy(dtype=float)
        if m["use_log"]: x = np.where(x<=0, np.nan, x)
        p = m["predict"](x)
        # blend with prior for stability / direction
        out = []
        for xi, pi in zip(x, p):
            xi_f = float(xi) if (xi is not None and np.isfinite(xi)) else None
            out.append(blend_with_prior(vkey, float(pi), xi_f))
        return np.array(out, dtype=float)

    X = []; feat_names=[]
    for v in var_list:
        if v in num.columns:
            feat_names.append(v); X.append(predict_var(v, num[v]))
    if not feat_names: raise ValueError("No features available to fit weights.")
    X = np.vstack(X).T  # N x K
    y = num["_y"].to_numpy(dtype=float)

    # Ridge logistic regression (IRLS) — stable
    def fit_logreg_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1.0, maxit: int = 50) -> Tuple[np.ndarray, float]:
        N, K = X.shape
        X1 = np.concatenate([np.ones((N,1)), X], axis=1)
        beta = np.zeros(K+1)
        for _ in range(maxit):
            z   = np.clip(X1 @ beta, -50.0, 50.0)
            p   = 1.0 / (1.0 + np.exp(-z))
            W   = np.clip(p*(1-p), 1e-6, None)
            R   = y - p
            WX  = X1 * W[:,None]
            H   = X1.T @ WX
            H[1:,1:] += l2 * np.eye(K)
            g   = X1.T @ (W*(z + R/np.clip(W,1e-9,None)))
            try:
                beta_new = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                beta_new = beta + 0.1 * (g - H @ beta)
            if np.max(np.abs(beta_new - beta)) < 1e-6:
                beta = beta_new; break
            beta = beta_new
        return beta, float(beta[0])

    try:
        beta, intercept = fit_logreg_ridge(X, y, l2=1.0)
        coefs = beta[1:]
        weights = np.abs(coefs); weights = weights / (weights.sum() if weights.sum()>0 else 1.0)
        weights_table = pd.DataFrame({"Variable": feat_names, "Coef": coefs, "Weight": weights})
    except Exception:
        weights = np.ones(len(feat_names)); weights = weights / weights.sum()
        weights_table = pd.DataFrame({"Variable": feat_names, "Coef": np.zeros(len(feat_names)), "Weight": weights})

    return {
        "curves": curves,
        "summary": summary_df,
        "mapping": mapping,
        "feat_names": feat_names,
        "weights": dict(zip(feat_names, weights_table["Weight"].to_numpy(float))),
        "weights_table": weights_table,
    }

# Learn button
if learn_btn:
    if not uploaded:
        st.error("Upload an Excel first.")
    else:
        try:
            learned = learn_all_curves_and_weights(uploaded, merged_sheet, ft_sheet, fail_sheet)
            st.session_state.CURVES = learned
            st.session_state.WEIGHTS = learned.get("weights", {})
            st.success(f"Learned {len(learned.get('curves',{}))} variable curves; weights fitted for {len(learned.get('feat_names',[]))} vars.")
        except Exception as e:
            st.error(f"Learning failed: {e}")

# Mapping & training summary (collapsible)
if st.session_state.CURVES:
    with st.expander("Detected Column Mapping, Curves & Weights (details)", expanded=False):
        st.markdown("**Detected column mapping**")
        st.dataframe(st.session_state.CURVES.get("mapping", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Learned curve training summary**")
        st.dataframe(st.session_state.CURVES.get("summary", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Logistic meta-weights (normalized)**")
        st.dataframe(st.session_state.CURVES.get("weights_table", pd.DataFrame()), use_container_width=True, hide_index=True)

# --- Curve plots ---
if st.session_state.CURVES:
    curves = st.session_state.CURVES.get("curves", {})
    if curves:
        st.markdown('<div class="section-title">📈 Learned probability curves</div>', unsafe_allow_html=True)
        for v, m in curves.items():
            if "grid_t" in m and "p_grid" in m:
                df_curve = pd.DataFrame({
                    "x": np.exp(m["grid_t"]) if m["use_log"] else m["grid_t"],
                    "p_ft": m["p_grid"]
                })
                st.line_chart(df_curve.set_index("x"), height=200)
                st.caption(f"{v} — learned P(FT|x) curve (log scale: {m['use_log']})")
              
# ================= Per-variable prob helpers =================
def per_var_prob(curves: Dict[str, Any], var_key: str, xval: float) -> float:
    m = curves.get(var_key)
    if m is None: return 0.5
    if m["use_log"] and (xval is None or not np.isfinite(xval) or xval <= 0): return 0.5
    try:
        p = float(m["predict"]([xval])[0]); p = float(np.clip(p, 1e-3, 1-1e-3))
        return blend_with_prior(var_key, p, xval)
    except Exception:
        return 0.5

def tail_penalty_for(k: str, v: float, reliability: float) -> float:
    band = DOMAIN_OVERRIDES.get(k, {})
    low, high = band.get("low", None), band.get("high", None)
    base_low, base_high = BASE_PEN_STRENGTH.get(k, (0.25, 0.25))
    return tail_penalty_bidirectional_soft(v, low, high, base_low, base_high, reliability, deadzone_frac=0.05, alpha=0.60)

def bucket_line(var_label: str, value: Optional[float], p: float, reason: str) -> str:
    if value is None or not np.isfinite(value): v_str = "—"
    else:
        if abs(value) >= 1000: v_str = f"{value:,.0f}"
        elif abs(value) >= 100: v_str = f"{value:.0f}"
        elif abs(value) >= 10:  v_str = f"{value:.1f}"
        else: v_str = f"{value:.2f}"
    return f"- **{var_label}**: {v_str} — *{reason}* (p≈{p*100:.0f}%)"

# ================= Tabs =================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1.2, 1.2, 1.0])

        with col1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)

        with col2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
            # Dilution slider directly below PM $ Volume
            dilution_flag = st.select_slider("Dilution (manual)", options=[0,1], value=0,
                                             help="0 = none/negligible; 1 = present/meaningful")

        with col3:
            catalyst_override = st.selectbox("Catalyst", ["Auto (from DB)", "No", "Yes"])

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        curves = (st.session_state.CURVES or {}).get("curves", {}) if isinstance(st.session_state.CURVES, dict) else {}
        learned_w = st.session_state.WEIGHTS or {}

        # derived live vars
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")
        pred_vol_m = predict_day_volume_m(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if (pred_vol_m and pred_vol_m>0) else float("nan")

        # catalyst override
        if catalyst_override.startswith("Auto"):
            catalyst_val = np.nan
        else:
            catalyst_val = 1.0 if catalyst_override == "Yes" else 0.0

        val_map = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "pm_vol_m": pm_vol_m, "pm_dol_m": pm_dol_m, "float_m": float_m, "mcap_m": mc_m,
            "fr_x": fr_x, "pmmc_pct": pmmc_pct, "pm_pct_pred": pm_pct_pred,
            "catalyst": catalyst_val,
        }

        # per-variable probs
        probs = {k: per_var_prob(curves, k, v) for k, v in val_map.items()}

        # reliability: we trust learned curves more
        reliab = {k: (1.0 if k in curves else 0.3) for k in val_map.keys()}

        # penalties (Δlogodds), including soft guardrails
        pen_vec = np.array([tail_penalty_for(k, val_map[k], reliab.get(k,0.5)) for k in val_map.keys()], dtype=float)

        # learned weights for variables (weighted average, not stacking)
        feat_keys = list(val_map.keys())
        weights_vec = np.array([learned_w.get(k, 0.0) for k in feat_keys], dtype=float)
        if weights_vec.sum() <= 1e-9:
            weights_vec = np.ones(len(feat_keys), dtype=float)
        weights_vec = weights_vec / weights_vec.sum()

        p_vec = np.array([probs[k] for k in feat_keys], dtype=float)

        # weighted mean probability
        p_final_mean = float(np.clip(np.sum(weights_vec * p_vec), 1e-6, 1.0 - 1e-6))

        # convert to log-odds, add penalties + dilution
        z = stable_logit(p_final_mean)
        z_pen = float(np.clip(pen_vec.sum(), -1.5, 0.0)) + (-0.35 if dilution_flag==1 else 0.0)
        p_numeric = prob_from_logodds(z + z_pen)
        final_score = float(np.clip(100.0 * p_numeric, 0.0, 100.0))
        odds = odds_label(final_score); level = grade(final_score)

        # verdict pill
        if final_score >= 75.0: verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
        elif final_score >= 55.0: verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
        else: verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

        # Checklist buckets
        names_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","pm_pct_pred":"PM Vol % of Pred",
            "catalyst":"Catalyst"
        }
        good_items, caution_items, risk_items = [], [], []

        def bucket_for(k: str, p: float, val: Optional[float]) -> Tuple[str,str]:
            band = DOMAIN_OVERRIDES.get(k, {})
            low, high = band.get("low", None), band.get("high", None)
            base_low, base_high = BASE_PEN_STRENGTH.get(k, (0.25, 0.25))
            pen = tail_penalty_bidirectional_soft(val, low, high, base_low, base_high, 1.0, 0.05, 0.60)
            if pen <= -0.15: return "risk","risk"
            if p >= 0.60:    return "good","supports"
            if p <= 0.40:    return "risk","headwind"
            return "caution","caution"

        for k in names_map.keys():
            label = names_map[k]; val = val_map.get(k, None); p = probs.get(k, 0.5)
            bucket, reason = bucket_for(k, p, val if isinstance(val,(int,float)) else np.nan)
            # show input value and probability
            if val is None or not np.isfinite(val): v_str = "—"
            else:
                if abs(val) >= 1000: v_str = f"{val:,.0f}"
                elif abs(val) >= 100: v_str = f"{val:.0f}"
                elif abs(val) >= 10:  v_str = f"{val:.1f}"
                else: v_str = f"{val:.2f}"
            line = f"- **{label}**: {v_str} — *{reason}* (p≈{p*100:.0f}%)"
            if bucket == "good": good_items.append(line)
            elif bucket == "risk": risk_items.append(line)
            else: caution_items.append(line)

        # save row
        row = {
            "Ticker": ticker, "Odds": odds, "Level": level, "FinalScore": round(final_score, 2),
            "PremarketVerdict": verdict, "Dilution": "Yes" if dilution_flag==1 else "No",
            "PredVol_M": round(pred_vol_m, 2) if np.isfinite(pred_vol_m) else "",
            "PredVol_CI68_L": round(ci68_l, 2) if np.isfinite(ci68_l) else "",
            "PredVol_CI68_U": round(ci68_u, 2) if np.isfinite(ci68_u) else "",
            "PremarketGood": good_items, "PremarketCaution": caution_items, "PremarketRisk": risk_items,
        }
        st.session_state.rows.append(row); st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---- Preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC = st.columns(3)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Odds", l.get("Odds","—"))

        d1, d2 = st.columns(2)
        d1.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M','')}")
        d1.caption(f"CI68: {l.get('PredVol_CI68_L','')}–{l.get('PredVol_CI68_U','')} M")
        d2.metric("Dilution (manual)", l.get("Dilution","No"))

        with st.expander("Premarket Checklist (data-weighted, smooth)", expanded=True):
            pill_html = (
                '<span class="pill pill-good">Strong Setup</span>' if l.get("PremarketVerdict")=="Strong Setup" else
                '<span class="pill pill-warn">Constructive</span>' if l.get("PremarketVerdict")=="Constructive" else
                '<span class="pill pill-bad">Weak / Avoid</span>'
            )
            st.markdown(f"**Verdict:** {pill_html}", unsafe_allow_html=True)
            g_col, w_col, r_col = st.columns(3)
            def _ul(items): return "<ul>" + "".join([f"<li>{x}</li>" for x in (items or [])]) + "</ul>" if items else "<ul><li><span class='hint'>None</span></li></ul>"
            with g_col: st.markdown("**Good**"); st.markdown(_ul(l.get("PremarketGood", [])), unsafe_allow_html=True)
            with w_col: st.markdown("**Caution**"); st.markdown(_ul(l.get("PremarketCaution", [])), unsafe_allow_html=True)
            with r_col: st.markdown("**Risk**"); st.markdown(_ul(l.get("PremarketRisk", [])), unsafe_allow_html=True)

with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns: df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PremarketVerdict","PredVol_M","PredVol_CI68_L","PredVol_CI68_U","Dilution"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level","PremarketVerdict","Dilution") else 0.0
        df = df[cols_to_show]
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Premarket Verdict"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "Dilution": st.column_config.TextColumn("Dilution"),
            }
        )
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "ranking.csv", "text/csv", use_container_width=True)
        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []; st.session_state.last = {}; do_rerun()
    else:
        st.info("No rows yet. Upload your DB and click **Learn curves + weights**, then add a stock in the **Add Stock** tab.")
