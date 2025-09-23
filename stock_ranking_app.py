import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List, Tuple

# =============== Page & Styles ===============
st.set_page_config(page_title="Premarket Stock Ranking (Smooth + Data-Weighted)", layout="wide")
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
  .debug-table th, .debug-table td { font-size:12px; }
</style>
""", unsafe_allow_html=True)

# =============== Helpers ===============
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    s = s.replace(" ", "").replace("’", "").replace("'", "")
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
    default_str = fmt.format(float(value))
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

# =============== Prediction Model (for Predicted Day Vol) ===============
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Returns Y in **millions of shares**
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    if mc <= 0 or gp <= 0 or atr <= 0:
        return float("nan")
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if not np.isfinite(pred_m) or pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# =============== Upload / Learn (MAIN PANE) ===============
st.markdown('<div class="section-title">Upload your database</div>', unsafe_allow_html=True)
upcol = st.columns([1.0])[0]
uploaded = upcol.file_uploader("Excel (.xlsx) with PMH sheets", type=["xlsx"], accept_multiple_files=False)

sc1, sc2, sc3 = st.columns([0.33, 0.33, 0.34])
with sc1:
    merged_sheet = st.text_input("Merged sheet name (preferred)", "PMH BO Merged")
with sc2:
    ft_sheet = st.text_input("FT sheet name (fallback)", "PMH BO FT")
with sc3:
    fail_sheet = st.text_input("Fail sheet name (fallback)", "PMH BO Fail")

learn_btn = st.button("📚 Learn curves + weights from uploaded file", use_container_width=True)

# =============== Session State ===============
if "CURVES" not in st.session_state: st.session_state.CURVES = {}
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =============== Column detection, cleaning, mapping ===============
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
    "catalyst":      ["catalyst","news","pr","has news","with news","with catalyst","catalyst flag","news flag","pr flag"],
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
            if norm_map[c] == _norm(n):
                return c
    # synonyms
    for cand in _SYNONYMS.get(logical_key, []):
        n_cand = _norm(cand)
        for c in cols:
            if norm_map[c] == n_cand:
                return c
        for c in cols:
            if n_cand in norm_map[c]:
                return c
    # loose contains
    tokens = re.split(r"[^\w]+", logical_key)
    tokens = [t for t in tokens if t]
    for c in cols:
        nm = norm_map[c]
        if all(t in nm for t in tokens):
            return c
    return None

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def _numify(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(str).str.replace(",","", regex=False).str.replace("'","", regex=False).str.strip()
    s = s.str.replace("%","", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _ensure_gap_pct(series: pd.Series) -> pd.Series:
    s = _numify(series)
    if s.dropna().lt(5).mean() > 0.6:  # many values like 0.8 instead of 80
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
            pred.append(predict_day_volume_m_premarket(mc, gp, atr))
        out["pred_vol_m"] = pred
        out["pm_pct_pred"] = 100.0 * out["pm_vol_m"] / out["pred_vol_m"]  # PM Vol % of Predicted
    if "pm_vol_m" in out and "daily_vol_m" in out:
        out["pm_pct_daily"] = 100.0 * out["pm_vol_m"] / out["daily_vol_m"]

    return out

def reveal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    dfc = _clean_cols(df)
    rows = []
    for key in ["label","catalyst","gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","daily_vol_m"]:
        col = _pick(dfc, key)
        rows.append({"Logical": key, "Matched Column": col if col else "(not found)"})
    return pd.DataFrame(rows)

# =============== Smooth Curves (Kernel) ===============
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

def fit_curve(series: pd.Series, labels: pd.Series, force_log: bool=False):
    x_raw = pd.to_numeric(series, errors="coerce").to_numpy()
    y_raw = pd.to_numeric(labels, errors="coerce").to_numpy()
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]; y_raw = y_raw[mask]
    if len(x_raw) < MIN_ROWS: return None

    use_log = False
    if force_log and (x_raw > 0).all(): use_log = True
    elif (x_raw > 0).all() and abs(pd.Series(x_raw).skew()) > 0.75:
        use_log = True

    if use_log:
        pos = x_raw > 0
        x_raw = x_raw[pos]; y_raw = y_raw[pos]
        if len(x_raw) < MIN_ROWS: return None
        x_t = np.log(x_raw)
    else:
        x_t = x_raw.copy()

    std = np.std(x_t); n = len(x_t)
    h = 0.6 * std * (n ** (-1/5)) if std > 0 else 0.3
    lo = np.nanpercentile(x_t, 2); hi = np.nanpercentile(x_t, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi: return None
    grid = np.linspace(lo, hi, 161)
    p_grid = _kernel_prob(x_t, y_raw, grid, h)

    def predict(x_new):
        x_new = np.array(x_new, dtype=float)
        if use_log:
            bad = ~np.isfinite(x_new) | (x_new <= 0)
            z = np.full_like(x_new, np.nan, dtype=float)
            ok = ~bad; z[ok] = np.log(x_new[ok])
        else:
            z = x_new
        res = _kernel_prob(x_t, y_raw, z, h)
        res = np.where(np.isfinite(res), res, 0.5)
        return np.clip(res, 1e-3, 1-1e-3)

    q10, q25, q50, q75, q90 = [float(pd.Series(x_raw).quantile(p)) for p in [0.10,0.25,0.50,0.75,0.90]]
    return {"use_log": use_log, "grid_t": grid, "p_grid": p_grid, "predict": predict,
            "n": len(x_t), "bandwidth_t": h,
            "q": {"p10": q10, "p25": q25, "p50": q50, "p75": q75, "p90": q90}}

# =============== Priors, Guardrails, Penalties (soft) ===============
PRIORS = {
    "gap_pct":     {"kind":"hump", "c":110.0, "w":45.0},
    "rvol":        {"kind":"hump", "c":9.0,   "w":6.0},
    "pm_dol_m":    {"kind":"hump", "c":11.0,  "w":8.0},
    "pm_pct_pred": {"kind":"hump", "c":18.0,  "w":10.0},
    "pmmc_pct":    {"kind":"hump", "c":12.0,  "w":10.0},
    "atr_usd":     {"kind":"hump", "c":0.30,  "w":0.18},
    "float_m":     {"kind":"low_better"},
    "mcap_m":      {"kind":"low_better"},
    # SI: high is better; pivot ~ ln(15)
    "si_pct":      {"kind":"high_better", "pivot": math.log(15.0), "scale": 0.35},
    "fr_x":        {"kind":"hump", "c":0.25,  "w":0.15},
    # catalyst learned from DB as a feature (0/1), but we also define a prior fallback
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

# High-side penalties softened (still real for exhaustion)
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

# Global tail scaling
tail_scale = 1.00
PER_VAR_CLIP = 1.20  # clip per-var logit effect before weighting
TAU = 1.00           # effect scaling

# =============== Prior curves helpers ===============
def _prior_p(var: str, x: float) -> float:
    """Simple prior curve p in [0.05,0.95]."""
    if x is None or not np.isfinite(x): return 0.5
    spec = PRIORS.get(var, {"kind":"flat"})
    kind = spec.get("kind","flat")
    if kind == "hump":
        c = spec.get("c", 0.0); w = spec.get("w", 1.0)
        p = math.exp(-0.5 * ((x - c) / max(1e-6,w))**2)
        # rescale to ~[0.2, 0.8]
        return 0.2 + 0.6 * p
    if kind == "low_better":
        s = 1.0 / (1.0 + math.exp( (x - 12.0) / 4.0 ))  # soft step around 12M by default
        return 0.2 + 0.6 * s
    if kind == "high_better":
        pivot = spec.get("pivot", 0.0)
        scale = spec.get("scale", 1.0)
        z = (x - pivot) / max(1e-6, scale)
        s = 1.0 / (1.0 + math.exp(-z))
        return 0.2 + 0.6 * s
    return 0.5

def blend_with_prior(var: str, p_learned: float, x: float) -> float:
    alpha = PRIOR_BLEND_MAP.get(var, DEFAULT_PRIOR_BLEND)
    pr = _prior_p(var, x)
    p = (1 - alpha) * p_learned + alpha * pr
    return float(np.clip(p, 1e-3, 1-1e-3))

def tail_penalty_bidirectional_soft(x, low, high, k_low, k_high, reliability, deadzone_frac=0.05, alpha=0.60):
    """Return log-odds delta (negative) for tails."""
    if x is None or not np.isfinite(x): return 0.0
    if low is None and high is None: return 0.0
    # Decide a center from guardrails if both present
    if low is not None and high is not None:
        center = (low + high) / 2.0
        span = (high - low)
        dz = deadzone_frac * span
        if x < low - dz:
            d = (low - x) / max(1e-6, span/2)
            pen = -k_low * reliability * (1 - math.exp(-alpha * d))
            return pen
        elif x > high + dz:
            d = (x - high) / max(1e-6, span/2)
            pen = -k_high * reliability * (1 - math.exp(-alpha * d))
            return pen
        else:
            return 0.0
    # Only one side present
    if low is not None and x < low:
        d = (low - x) / max(1e-6, abs(low))
        return -k_low * reliability * (1 - math.exp(-alpha * d))
    if high is not None and x > high:
        d = (x - high) / max(1e-6, abs(high))
        return -k_high * reliability * (1 - math.exp(-alpha * d))
    return 0.0

# =============== Learn Curves + Weights ===============
def learn_all_curves_and_weights(file, merged_sheet: str, ft_sheet: str, fail_sheet: str):
    xls = pd.ExcelFile(file)
    used_merged = False
    if merged_sheet in xls.sheet_names:
        raw = pd.read_excel(xls, merged_sheet)
        used_merged = True
        merged_map = reveal_mapping(raw)
        num = build_numeric_table(raw)
        if "_y" not in num.columns or num["_y"].nunique() < 2:
            raise ValueError("Merged sheet found but no valid label column matched.")
    else:
        if ft_sheet not in xls.sheet_names or fail_sheet not in xls.sheet_names:
            raise ValueError(f"Sheets not found. Available: {xls.sheet_names}")
        ft_raw   = pd.read_excel(xls, ft_sheet)
        fail_raw = pd.read_excel(xls, fail_sheet)
        merged_map = pd.concat([reveal_mapping(ft_raw).assign(Sheet="FT"),
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
    curves = {}; learned = 0; summary_rows = []
    for v in var_list:
        if v in num.columns:
            model = fit_curve(num[v], num["_y"], force_log=(v=="si_pct"))
            if model:
                curves[v] = model; learned += 1
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

    # Build training matrix of per-variable p_i(x) for logistic meta-model
    def predict_var(vkey: str, xcol: pd.Series) -> np.ndarray:
        if vkey not in curves: return np.full(len(xcol), 0.5)
        m = curves[vkey]
        x = xcol.to_numpy(dtype=float)
        if m["use_log"]:
            x = np.where(x<=0, np.nan, x)
        p = m["predict"](x)
        # Blend with prior for stability (variable-specific alpha)
        out = []
        for xi, pi in zip(x, p):
            out.append(blend_with_prior(vkey, float(pi), float(xi) if np.isfinite(xi) else None))
        return np.array(out, dtype=float)

    X = []; feat_names=[]
    for v in var_list:
        if v in num.columns:
            feat_names.append(v)
            X.append(predict_var(v, num[v]))
    if not feat_names:
        raise ValueError("No features available to fit weights.")
    X = np.vstack(X).T  # N x K
    y = num["_y"].to_numpy(dtype=float)

    # Regularized logistic regression (IRLS with L2)
    def fit_logreg_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1.0, maxit: int = 50) -> Tuple[np.ndarray, float]:
        N, K = X.shape
        X1 = np.concatenate([np.ones((N,1)), X], axis=1)
        beta = np.zeros(K+1)  # intercept + K
        for _ in range(maxit):
            z = X1 @ beta
            p = 1.0 / (1.0 + np.exp(-z))
            W = p * (1 - p)
            # Guard: avoid zero weights
            W = np.clip(W, 1e-6, None)
            R = y - p
            # (X^T W X + l2*I) beta = X^T W (X beta + R/W)
            WX = X1 * W[:,None]
            H = X1.T @ WX
            H[1:,1:] += l2 * np.eye(K)  # ridge on coefficients, not intercept
            g = X1.T @ (W*(z + R/W))
            try:
                beta_new = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < 1e-6:
                beta = beta_new; break
            beta = beta_new
        # Return coefficients (exclude intercept here; we use weights only)
        return beta, float(beta[0])

    try:
        beta, intercept = fit_logreg_ridge(X, y, l2=1.0)
        coefs = beta[1:]  # drop intercept
        # turn to positive weights via abs, then normalize
        weights = np.abs(coefs)
        if weights.sum() <= 1e-9:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        weights_table = pd.DataFrame({"Variable": feat_names, "Coef": coefs, "Weight": weights})
    except Exception as e:
        # fallback to equal weights
        weights = np.ones(len(feat_names)); weights = weights / weights.sum()
        weights_table = pd.DataFrame({"Variable": feat_names, "Coef": np.zeros(len(feat_names)), "Weight": weights})

    return {
        "curves": curves,
        "summary": summary_df,
        "mapping": merged_map,
        "feat_names": feat_names,
        "weights": dict(zip(feat_names, weights_table["Weight"].to_numpy(dtype=float))),
        "weights_table": weights_table,
    }

# =============== Learn Button ===============
if learn_btn:
    if not uploaded:
        st.error("Upload an Excel first.")
    else:
        try:
            learned = learn_all_curves_and_weights(uploaded, merged_sheet, ft_sheet, fail_sheet)
            st.session_state.CURVES = learned
            st.session_state.WEIGHTS = learned.get("weights", {})
            st.success(f"Learned curves for {learned['summary']['n'].astype(int).sum()} samples across variables; weights fitted for {len(learned.get('feat_names',[]))} variables.")
        except Exception as e:
            st.error(f"Learning failed: {e}")

# =============== Mapping & Training Summary (collapsible) ===============
if st.session_state.CURVES:
    with st.expander("Detected Column Mapping & Training Summary", expanded=False):
        st.markdown("**Detected column mapping**")
        st.dataframe(st.session_state.CURVES.get("mapping", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Learned curve training summary**")
        st.dataframe(st.session_state.CURVES.get("summary", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Logistic meta-weights (normalized)**")
        st.dataframe(st.session_state.CURVES.get("weights_table", pd.DataFrame()), use_container_width=True, hide_index=True)

# =============== Controls (Prediction Uncertainty, Dilution) ===============
cc1, cc2 = st.columns([0.5, 0.5])
with cc1:
    sigma_ln = st.slider("Predicted Day Vol log-σ (for CI)", 0.10, 1.50, 0.60, 0.01,
                         help="Residual std dev of ln(day volume) used for confidence bands.")
with cc2:
    dilution_flag = st.select_slider("Dilution (manual)", options=[0,1],
                                     value=0, help="0 = none/negligible; 1 = present/meaningful")

# =============== Add Stock / Ranking Tabs ===============
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

# ---------- Per-variable probability helper ----------
def per_var_prob(curves: Dict[str, Any], var_key: str, xval: float) -> float:
    m = curves.get(var_key)
    if m is None:
        return 0.5
    if m["use_log"] and (xval is None or not np.isfinite(xval) or xval <= 0):
        return 0.5
    try:
        p = float(m["predict"]([xval])[0])
        p = float(np.clip(p, 1e-3, 1-1e-3))
        # blend with prior
        return blend_with_prior(var_key, p, xval)
    except Exception:
        return 0.5

def bucket_line(var_label: str, value: Optional[float], p: float, reason: str) -> str:
    # pretty number
    if value is None or not np.isfinite(value):
        v_str = "—"
    else:
        if abs(value) >= 1000: v_str = f"{value:,.0f}"
        elif abs(value) >= 100: v_str = f"{value:.0f}"
        elif abs(value) >= 10:  v_str = f"{value:.1f}"
        else:                   v_str = f"{value:.2f}"
    return f"- **{var_label}**: {v_str} — *{reason}* (p≈{p*100:.0f}%)"

with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)

    # Form
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

        with col3:
            catalyst_input = st.selectbox("Catalyst", ["Auto (from DB priors)", "No", "Yes"])
            st.caption("If ‘Auto’, the learned curve on your DB drives Catalyst’s p(x).")
            st.caption("If ‘Yes/No’, we hard-set value for this ticker.")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        curves = (st.session_state.CURVES or {}).get("curves", {}) if isinstance(st.session_state.CURVES, dict) else {}

        # Derived real-time
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted Day Vol & PM% of Predicted
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if (pred_vol_m and pred_vol_m>0) else float("nan")

        # Gather inputs for per-variable probabilities
        val_map = {
            "gap_pct": gap_pct,
            "atr_usd": atr_usd,
            "rvol": rvol,
            "si_pct": si_pct,
            "pm_vol_m": pm_vol_m,
            "pm_dol_m": pm_dol_m,
            "float_m": float_m,
            "mcap_m": mc_m,
            "fr_x": fr_x,
            "pmmc_pct": pmmc_pct,
            "pm_pct_pred": pm_pct_pred,
        }
        # Catalyst value for probability
        if catalyst_input.startswith("Auto"):
            catalyst_val = np.nan  # use curve on NA as 0.5 (then prior blend elevates by DB)
        else:
            catalyst_val = 1.0 if catalyst_input == "Yes" else 0.0
        val_map["catalyst"] = catalyst_val

        # Per-variable probabilities
        probs = {}
        for k, v in val_map.items():
            probs[k] = per_var_prob(curves, k, v)

        # Soft tail penalties (exhaustion / weak zones)
        reliab = {k: (1.0 if k in curves else 0.3) for k in val_map.keys()}
        guardrails = DOMAIN_OVERRIDES

        def tail_penalty_for(k: str, v: float) -> float:
            band = guardrails.get(k, {})
            low, high = band.get("low", None), band.get("high", None)
            base_low, base_high = BASE_PEN_STRENGTH.get(k, (0.25, 0.25))
            kL = tail_scale * base_low
            kH = tail_scale * base_high
            rel_w = float(np.clip(reliab.get(k, 0.0), 0.0, 1.0))
            return tail_penalty_bidirectional_soft(v, low, high, kL, kH, rel_w, deadzone_frac=0.05, alpha=0.60)

        # Weighted averaging (learned weights); map any missing to equal share
        learned_w = st.session_state.WEIGHTS or {}
        feat_keys = list(val_map.keys())
        weights_vec = np.array([learned_w.get(k, 0.0) for k in feat_keys], dtype=float)
        if weights_vec.sum() <= 1e-9:
            weights_vec = np.ones(len(feat_keys), dtype=float)
        weights_vec = weights_vec / weights_vec.sum()

        p_vec = np.array([probs[k] for k in feat_keys], dtype=float)
        # Convert penalties (Δlogodds) to equivalent prob deltas by local linearization
        # For UI verdicts we’ll use buckets; for final score, apply soft penalty as weight dampener
        pen_vec = np.array([tail_penalty_for(k, val_map[k]) for k in feat_keys], dtype=float)

        # Final numeric probability = weighted mean of probs, then apply a mild global tail impact
        p_final_mean = float(np.clip(np.sum(weights_vec * p_vec), 0.0, 1.0))
        # Convert combined penalty (sum of Δlogodds) to prob shift
        z = math.log(max(p_final_mean,1e-6)/max(1-p_final_mean,1e-6))
        z_pen = float(np.clip(pen_vec.sum(), -1.5, 0.0))  # cap negative push
        # manual dilution: only 0/1, subtract modest edge when present
        z_pen += (-0.35 if dilution_flag == 1 else 0.0)
        p_numeric = 1.0 / (1.0 + math.exp(-(z + z_pen)))
        numeric_pct = 100.0 * p_numeric

        # Final score/grade from numeric only (per your request)
        final_score = float(np.clip(numeric_pct, 0.0, 100.0))
        odds = odds_label(final_score)
        level = grade(final_score)

        # Verdict pill
        if final_score >= 75.0:
            verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
        elif final_score >= 55.0:
            verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
        else:
            verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

        # Checklist categories: good / caution / risk
        good_items, caution_items, risk_items = [], [], []

        names_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","pm_pct_pred":"PM Vol % of Pred",
            "catalyst":"Catalyst"
        }

        # Decide line bucket using p and tail penalty
        def bucket_for(k: str, p: float, val: Optional[float]) -> Tuple[str,str]:
            band = guardrails.get(k, {})
            low, high = band.get("low", None), band.get("high", None)
            base_low, base_high = BASE_PEN_STRENGTH.get(k, (0.25, 0.25))
            kL = tail_scale * base_low; kH = tail_scale * base_high
            rel_w = float(np.clip(reliab.get(k,0.0),0.0,1.0))
            pen = tail_penalty_bidirectional_soft(val, low, high, kL, kH, rel_w, deadzone_frac=0.05, alpha=0.60)
            if pen <= -0.15:
                return "risk","risk"
            if p >= 0.60:
                return "good","supports"
            if p <= 0.40:
                return "risk","headwind"
            return "caution","caution"

        for k in names_map.keys():
            label = names_map[k]
            val = val_map.get(k, None)
            p = probs.get(k, 0.5)
            bucket, reason = bucket_for(k, p, val if isinstance(val,(int,float)) else np.nan)
            line = bucket_line(label, val, p, reason)
            if bucket == "good": good_items.append(line)
            elif bucket == "risk": risk_items.append(line)
            else: caution_items.append(line)

        # Row for ranking
        row = {
            "Ticker": ticker,
            "Odds": odds,
            "Level": level,
            "FinalScore": round(final_score, 2),
            "PremarketVerdict": verdict,
            "PredVol_M": round(pred_vol_m, 2) if np.isfinite(pred_vol_m) else "",
            "PredVol_CI68_L": round(ci68_l, 2) if np.isfinite(ci68_l) else "",
            "PredVol_CI68_U": round(ci68_u, 2) if np.isfinite(ci68_u) else "",
            "Dilution": "Yes" if dilution_flag==1 else "No",
            # Checklist sets
            "PremarketGood": good_items,
            "PremarketCaution": caution_items,
            "PremarketRisk": risk_items,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC = st.columns(3)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score",   f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Odds", l.get("Odds","—"))

        d1, d2 = st.columns(2)
        d1.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M','')}")
        d1.caption(
            f"CI68: {l.get('PredVol_CI68_L','')}–{l.get('PredVol_CI68_U','')} M"
        )
        d2.metric("Dilution (manual)", l.get("Dilution","No"))

        with st.expander("Premarket Checklist (data-weighted, smooth)", expanded=True):
            pill = l.get("PremarketVerdict","—")
            pill_html = (
                '<span class="pill pill-good">Strong Setup</span>' if pill=="Strong Setup" else
                '<span class="pill pill-warn">Constructive</span>' if pill=="Constructive" else
                '<span class="pill pill-bad">Weak / Avoid</span>'
            )
            st.markdown(f"**Verdict:** {pill_html}", unsafe_allow_html=True)

            g_col, w_col, r_col = st.columns(3)
            def _ul(items):
                if not items: return "<ul><li><span class='hint'>None</span></li></ul>"
                return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

            with g_col:
                st.markdown("**Good**"); st.markdown(_ul(l.get("PremarketGood", [])), unsafe_allow_html=True)
            with w_col:
                st.markdown("**Caution**"); st.markdown(_ul(l.get("PremarketCaution", [])), unsafe_allow_html=True)
            with r_col:
                st.markdown("**Risk**"); st.markdown(_ul(l.get("PremarketRisk", [])), unsafe_allow_html=True)

# =============== Ranking Tab ===============
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level","FinalScore","PremarketVerdict",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","Dilution"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","PremarketVerdict","Dilution") else 0.0
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
                "Dilution": st.column_config.TextColumn("Dilution (manual)"),
            }
        )

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []; st.session_state.last = {}; do_rerun()
    else:
        st.info("No rows yet. Upload your DB and click **Learn curves + weights**, then add a stock in the **Add Stock** tab.")
