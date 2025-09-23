import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List, Tuple

# ====== Optional: ensure Excel engine is available ======
try:
    import openpyxl  # noqa: F401
    _EXCEL_ENGINE = "openpyxl"
except Exception:
    _EXCEL_ENGINE = None  # pandas will try; if missing, we'll warn

# =============== Page & Styles ===============
st.set_page_config(page_title="Premarket Stock Ranking (Smooth, Data-Driven)", layout="wide")
st.title("Premarket Stock Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  ul { margin-top: 4px; margin-bottom: 4px; padding-left: 18px; }
  li { margin-bottom: 2px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color:#374151; }
  .debug-table th, .debug-table td { font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ======= Calibrated stacking / correlation settings =======
PER_VAR_CLIP = 0.9      # per-variable log-odds delta clip (±)
TAU = 0.70              # global temperature for curve contributions
PEN_TAU = 0.60          # penalties scaled by this (softer than TAU)
MAX_PER_VAR_PEN = 0.50  # max |Δlogodds| per variable from tail penalty
MAX_TOTAL_PEN   = 1.20  # max |sum of penalties| in log-odds (~ -20–25 ppt)

# Modifiers (shown in Checklist as manual)
CATALYST_LOGODDS_COEF = 0.35
DILUTION_LOGODDS_COEF = -0.35

# Correlated PM activity cluster
CORR_GROUPS = [["pm_vol_m", "fr_x", "pm_pct_pred", "pmmc_pct"]]
GROUP_MAX_WEIGHT = 1.2

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
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float) -> Tuple[float, float]:
    if pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# =============== Column Detection (Robust) ===============
_SYNONYMS = {
    "gap_pct":       ["gap %","gap%","gap pct","premarket gap","pm gap"],
    "pm_vol_m":      ["pm vol (m)","pm volume (m)","premarket vol (m)","pm vol m","pm shares"],
    "pm_dol_m":      ["pm $vol (m)","pm $ vol (m)","pm dollar vol (m)","premarket $vol (m)","pm $volume"],
    "daily_vol_m":   ["daily vol (m)","day vol (m)","volume (m)","vol (m)"],
    "float_m":       ["float m shares","public float (m)","float (m)","float(m)","float m"],
    "mcap_m":        ["marketcap m","market cap (m)","market cap m","mcap (m)","mcap"],
    "rvol":          ["rvol @ bo","rvol","relative volume","rvol bo"],
    "atr_usd":       ["atr","atr $","atr (usd)","atr $/day"],
    "si_pct": [
        "short interest %","SI","si %","shortinterest %",
        "short float %","short %","short float","shortinterest float %",
        "short interest (float) %","short interest percent","short ratio %"
    ],
}

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.replace("$","").replace("%","")
    s = s.replace("’","").replace("'","")
    return s

def _pick(df: pd.DataFrame, logical_key: str, *prefer_exact: str) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {c: _norm(c) for c in cols}
    for n in prefer_exact:
        for c in cols:
            if norm_map[c] == _norm(n):
                return c
    for cand in _SYNONYMS.get(logical_key, []):
        n_cand = _norm(cand)
        for c in cols:
            if norm_map[c] == n_cand:
                return c
        for c in cols:
            if n_cand in norm_map[c]:
                return c
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

    out = pd.DataFrame()
    if col_gap:   out["gap_pct"]     = _ensure_gap_pct(df[col_gap])
    if col_atr:   out["atr_usd"]     = _numify(df[col_atr])
    if col_rvol:  out["rvol"]        = _numify(df[col_rvol])
    if col_si:    out["si_pct"]      = _numify(df[col_si])
    if col_pmvol: out["pm_vol_m"]    = _numify(df[col_pmvol])
    if col_pmdol: out["pm_dol_m"]    = _numify(df[col_pmdol])
    if col_float: out["float_m"]     = _numify(df[col_float])
    if col_mcap:  out["mcap_m"]      = _numify(df[col_mcap])
    if col_daily: out["daily_vol_m"] = _numify(df[col_daily])

    # Derived
    if "pm_vol_m" in out and "float_m" in out:
        out["fr_x"] = out["pm_vol_m"] / out["float_m"]                    # PM Float Rotation ×
    if "pm_dol_m" in out and "mcap_m" in out:
        out["pmmc_pct"] = 100.0 * out["pm_dol_m"] / out["mcap_m"]         # PM $Vol / MC %
    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(out.columns):
        pred = []
        for mc, gp, atr in zip(out["mcap_m"], out["gap_pct"], out["atr_usd"]):
            if pd.isna(mc) or pd.isna(gp) or pd.isna(atr) or mc<=0 or gp<=0 or atr<=0:
                pred.append(np.nan)
            else:
                pred.append(predict_day_volume_m_premarket(mc, gp, atr))
        out["pred_vol_m"] = pred
        out["pm_pct_pred"] = 100.0 * out["pm_vol_m"] / out["pred_vol_m"]  # PM Vol % of Predicted
    if "pm_vol_m" in out and "daily_vol_m" in out:
        out["pm_pct_daily"] = 100.0 * out["pm_vol_m"] / out["daily_vol_m"]

    return out

def reveal_mapping(df: pd.DataFrame) -> pd.DataFrame:
    dfc = _clean_cols(df)
    rows = []
    for key in ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","daily_vol_m"]:
        col = _pick(dfc, key)
        rows.append({"Logical": key, "Matched Column": col if col else "(not found)"})
    return pd.DataFrame(rows)

# =============== Smooth Curves (Kernel) ===============
MIN_ROWS = 30  # minimum usable rows for a curve (FT+Fail combined)

def fit_curve(series: pd.Series, labels: pd.Series):
    x_raw = pd.to_numeric(series, errors="coerce").to_numpy()
    y_raw = pd.to_numeric(labels, errors="coerce").to_numpy()

    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]
    if len(x_raw) < MIN_ROWS:
        return None

    use_log = False
    if (x_raw > 0).all():
        if abs(pd.Series(x_raw).skew()) > 0.75:
            use_log = True

    if use_log:
        pos = x_raw > 0
        x_raw = x_raw[pos]; y_raw = y_raw[pos]
        if len(x_raw) < MIN_ROWS:
            return None
        x_t = np.log(x_raw)
    else:
        x_t = x_raw.copy()

    std = np.std(x_t); n = len(x_t)
    h = 0.6 * std * (n ** (-1/5)) if std > 0 else 0.3

    lo = np.nanpercentile(x_t, 2); hi = np.nanpercentile(x_t, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    grid = np.linspace(lo, hi, 161)

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

    return {
        "use_log": use_log,
        "grid_t": grid,
        "p_grid": p_grid,
        "predict": predict,
        "n": len(x_t),
        "bandwidth_t": h,
        "q": {"p10": q10, "p25": q25, "p50": q50, "p75": q75, "p90": q90}
    }

# ====== AUC & calibration utils ======
def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size < 3 or len(np.unique(y)) < 2:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_ranks = np.bincount(inv, weights=ranks) / np.bincount(inv)
    ranks = avg_ranks[inv]
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0: return 0.5
    rank_sum_pos = ranks[y == 1].sum()
    u = rank_sum_pos - n1 * (n1 + 1) / 2.0
    auc = u / (n0 * n1)
    return float(np.clip(auc, 0.0, 1.0))

def _prob_from_curve(model, xvals: np.ndarray) -> np.ndarray:
    p = model["predict"](xvals)
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def _rebalance_group_weights(reliab: Dict[str, float]) -> Dict[str, float]:
    r = dict(reliab)
    for group in CORR_GROUPS:
        s = sum(r.get(k, 0.0) for k in group)
        if s > GROUP_MAX_WEIGHT and s > 0:
            scale = GROUP_MAX_WEIGHT / s
            for k in group:
                r[k] = r.get(k, 0.0) * scale
    return r

# =============== Domain priors (direction/shape hints) ===============
# Each variable gets a weak/medium prior curve (0..1) blended with learned curve.
# kind: "hump" (sweet-spot), "high_better", "low_better"
PRIORS = {
    "gap_pct":     {"kind":"hump", "c":100.0, "w":60.0},   # sweet ~100%, fades outside
    "rvol":        {"kind":"hump", "c":10.0,  "w":8.0},    # sweet ~10x, fades low/high (too high overheats)
    "pm_dol_m":    {"kind":"hump", "c":10.0,  "w":10.0},   # $7–15M sweet
    "pm_vol_m":    {"kind":"high_better"},                 # more PM shares (to a point)
    "pm_pct_pred": {"kind":"hump", "c":18.0,  "w":12.0},   # ~10–25% sweet
    "pmmc_pct":    {"kind":"hump", "c":15.0,  "w":15.0},   # ~5–30% ok; very high bloated
    "atr_usd":     {"kind":"hump", "c":0.30,  "w":0.20},   # 0.2–0.4 sweet
    "float_m":     {"kind":"low_better"},                  # smaller float better until extreme
    "mcap_m":      {"kind":"low_better"},                  # smaller cap better
    "si_pct":      {"kind":"high_better", "pivot":10.0, "scale":5.0},  # <<< skewed high-is-better
    "fr_x":        {"kind":"hump", "c":0.25,  "w":0.20},   # ~0.1–0.5x sweet; too high => front-loaded
}

# Prior blend weights (per-variable)
DEFAULT_PRIOR_BLEND = 0.30
PRIOR_BLEND_MAP = {
    "si_pct": 0.55,    # stronger blend for SI so low SI doesn't look great; high SI helps
    # others inherit DEFAULT_PRIOR_BLEND
}

def _prior_p(var: str, x: float) -> float:
    if not np.isfinite(x): return 0.5
    info = PRIORS.get(var, None)
    if info is None: return 0.5
    kind = info.get("kind","hump")
    if kind == "hump":
        c = float(info.get("c", 0.0)); w = float(info.get("w", 1.0))
        return float(np.clip(np.exp(-0.5 * ((x - c)/max(w,1e-6))**2), 1e-3, 1-1e-3))
    elif kind == "high_better":
        pivot = float(info.get("pivot", 3.0))
        scale = float(info.get("scale", 2.0))
        z = (x - pivot) / max(scale, 1e-6)
        return float(1.0 / (1.0 + math.exp(-z)))
    elif kind == "low_better":
        pivot = (15.0 if var=="float_m" else 150.0 if var=="mcap_m" else 10.0)
        scale = (6.0  if var=="float_m" else 80.0  if var=="mcap_m" else 5.0)
        z = -(x - pivot) / max(scale, 1e-6)
        return float(1.0 / (1.0 + math.exp(-z)))
    return 0.5

def _prior_blend_weight(var: str) -> float:
    return float(PRIOR_BLEND_MAP.get(var, DEFAULT_PRIOR_BLEND))

# =============== Learn from Excel (with reliability & priors) ===============
def learn_all_curves_from_excel(file, ft_sheet: str, fail_sheet: str) -> Dict[str, Any]:
    if _EXCEL_ENGINE is None:
        try:
            xls = pd.ExcelFile(file)
        except Exception as e:
            st.error("Reading .xlsx requires **openpyxl**. Install it (`pip install openpyxl`).")
            raise e
    else:
        xls = pd.ExcelFile(file, engine=_EXCEL_ENGINE)

    if ft_sheet not in xls.sheet_names or fail_sheet not in xls.sheet_names:
        raise ValueError(f"Sheets not found. Available: {xls.sheet_names}")

    ft_raw   = pd.read_excel(xls, ft_sheet, engine=_EXCEL_ENGINE) if _EXCEL_ENGINE else pd.read_excel(xls, ft_sheet)
    fail_raw = pd.read_excel(xls, fail_sheet, engine=_EXCEL_ENGINE) if _EXCEL_ENGINE else pd.read_excel(xls, fail_sheet)

    ft_map = reveal_mapping(ft_raw)
    fail_map = reveal_mapping(fail_raw)

    ft_num   = build_numeric_table(ft_raw)
    fail_num = build_numeric_table(fail_raw)

    ft_num["_y"] = 1
    fail_num["_y"] = 0
    all_num = pd.concat([ft_num, fail_num], axis=0, ignore_index=True)

    var_list = [
        "gap_pct","atr_usd","rvol","si_pct",
        "pm_vol_m","pm_dol_m","float_m","mcap_m",
        "fr_x","pmmc_pct","pm_pct_pred"
    ]
    curves = {}
    for v in var_list:
        if v in all_num.columns:
            model = fit_curve(all_num[v], all_num["_y"])
            if model:
                curves[v] = model

    # Training summary
    summary_rows = []
    for v in var_list:
        m = curves.get(v)
        if not m:
            summary_rows.append({"Variable": v, "n": 0, "Used Log": "", "BW (t)": "", "P10": "", "P50": "", "P90": ""})
        else:
            q = m["q"]
            summary_rows.append({
                "Variable": v, "n": m["n"], "Used Log": m["use_log"], "BW (t)": round(m["bandwidth_t"],3),
                "P10": round(q["p10"],3), "P50": round(q["p50"],3), "P90": round(q["p90"],3)
            })
    summary_df = pd.DataFrame(summary_rows)

    # Base rate & variable reliabilities
    y_all = all_num["_y"].to_numpy()
    base_rate = float(np.mean(y_all)) if len(y_all) else 0.5
    prior_logodds = math.log(max(base_rate,1e-6) / max(1 - base_rate, 1e-6))

    reliab = {}; ref_logit = {}
    for v in var_list:
        m = curves.get(v)
        if (m is None) or (v not in all_num.columns):
            reliab[v] = 0.0; ref_logit[v] = 0.0
            continue
        x = pd.to_numeric(all_num[v], errors="coerce").to_numpy()
        mask = np.isfinite(x)
        x = x[mask]; y = y_all[mask]
        if len(x) < 10 or len(np.unique(y)) < 2:
            reliab[v] = 0.0; ref_logit[v] = 0.0
            continue
        p_hat = _prob_from_curve(m, x)
        # Blend domain prior
        p_prior = np.array([_prior_p(v, xi) for xi in x], dtype=float)
        w_blend = _prior_blend_weight(v)
        p_hat = (1.0 - w_blend) * p_hat + w_blend * p_prior

        auc  = _safe_auc(y, p_hat)
        gamma = 0.8
        w = (2.0 * abs(auc - 0.5)) ** gamma
        reliab[v] = float(np.clip(w, 0.0, 1.0))
        ref_logit[v] = float(np.median(_logit(p_hat)))

    learned_meta = {
        "curves": curves,
        "summary": summary_df,
        "ft_map": ft_map,
        "fail_map": fail_map,
        "learned_count": len(curves),
        "base_rate": base_rate,
        "prior_logodds": prior_logodds,
        "reliability": reliab,
        "ref_logit": ref_logit,
        "train_num": all_num,
    }
    return learned_meta

# =============== Tail-Risk Guardrails (soft for all variables) ===============

DOMAIN_OVERRIDES = {
    "gap_pct":     {"low": 30.0,  "high": 300.0},
    "rvol":        {"low": 3.0,   "high": 3000.0},
    "pm_dol_m":    {"low": 3.0,   "high": 40.0},
    "pm_vol_m":    {"low": None,  "high": None},
    "pm_pct_pred": {"low": 5.0,   "high": 35.0},
    "pmmc_pct":    {"low": None,  "high": 35.0},
    "atr_usd":     {"low": 0.15,  "high": 1.50},
    "float_m":     {"low": 2.0,   "high": 20.0},
    "mcap_m":      {"low": None,  "high": 150.0},
    "si_pct":      {"low": 0.5,   "high": 35.0},
    "fr_x":        {"low": 1e-3,  "high": 1.00},
}

BASE_PEN_STRENGTH = {
    "gap_pct":     (0.35, 0.55),
    "rvol":        (0.30, 0.55),
    "pm_dol_m":    (0.25, 0.45),
    "pm_vol_m":    (0.15, 0.35),
    "pm_pct_pred": (0.25, 0.55),
    "pmmc_pct":    (0.15, 0.45),
    "atr_usd":     (0.30, 0.35),
    "float_m":     (0.25, 0.40),
    "mcap_m":      (0.10, 0.30),
    "si_pct":      (0.10, 0.25),
    "fr_x":        (0.15, 0.45),
}

def _qr_band(s: pd.Series, p_lo=5, p_hi=95) -> Tuple[float, float]:
    z = pd.to_numeric(s, errors="coerce")
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return (np.nan, np.nan)
    return (float(np.nanpercentile(z, p_lo)), float(np.nanpercentile(z, p_hi)))

def _blend(a_low, a_high, b_low, b_high, w=0.35):
    low  = None; high = None
    if a_low is not None or b_low is not None:
        aa = a_low if a_low is not None else b_low
        bb = b_low if b_low is not None else a_low
        low = (1-w)*aa + w*bb if (aa is not None and bb is not None) else (aa or bb)
    if a_high is not None or b_high is not None:
        aa = a_high if a_high is not None else b_high
        bb = b_high if b_high is not None else a_high
        high = (1-w)*aa + w*bb if (aa is not None and bb is not None) else (aa or bb)
    return low, high

def build_guardrails(curves_pack: Dict[str,Any]) -> Dict[str, Dict[str,float]]:
    out = {}
    train = curves_pack.get("train_num", pd.DataFrame())
    vars_all = ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred"]
    for v in vars_all:
        if v in train.columns:
            lo_d, hi_d = _qr_band(train[v], 5, 95)
        else:
            lo_d, hi_d = (np.nan, np.nan)
        dom = DOMAIN_OVERRIDES.get(v, {})
        lo_dom = dom.get("low", None); hi_dom = dom.get("high", None)
        lo_final, hi_final = _blend(
            lo_d if np.isfinite(lo_d) else None,
            hi_d if np.isfinite(hi_d) else None,
            lo_dom, hi_dom,
            w=0.35
        )
        out[v] = {"low": lo_final, "high": hi_final}
    return out

def tail_penalty_bidirectional_soft(x: float, low: Optional[float], high: Optional[float],
                                    k_low: float, k_high: float,
                                    rel_w: float,
                                    deadzone_frac: float = 0.05,
                                    alpha: float = 0.60,
                                    eps: float = 1e-9) -> float:
    if not np.isfinite(x):
        return 0.0
    pen = 0.0
    if (low is not None) and (x < low):
        margin = max(0.0, (low - x) - deadzone_frac * max(abs(low), 1.0))
        if margin > 0.0:
            dist = margin / (max(abs(low), 1.0) + eps)
            pen -= (k_low * (math.log1p(dist) ** alpha)) * rel_w
    if (high is not None) and (x > high):
        margin = max(0.0, (x - high) - deadzone_frac * max(abs(high), 1.0))
        if margin > 0.0:
            dist = margin / (max(abs(high), 1.0) + eps)
            pen -= (k_high * (math.log1p(dist) ** alpha)) * rel_w
    pen = float(np.clip(pen, -MAX_PER_VAR_PEN, MAX_PER_VAR_PEN))
    return pen

# =============== Sidebar: Modifiers & Uncertainty ===============
st.sidebar.header("Manual Modifiers")
catalyst_points = st.sidebar.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05,
                                    help="Manual catalyst strength. Positive moves odds up.")
dilution_points = st.sidebar.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05,
                                    help="Manual dilution/ATM/overhang. Negative moves odds down.")

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01)

st.sidebar.header("Tail-Risk (global)")
tail_scale = st.sidebar.slider("Global tail-risk strength", 0.25, 1.50, 1.00, 0.05,
                               help="Multiplies all variable penalty strengths.")

adv_tail = st.sidebar.checkbox("Advanced tail-risk tuning", value=False,
                               help="Show per-variable penalty multipliers.")
per_var_mult = {}
for v in BASE_PEN_STRENGTH.keys():
    if adv_tail:
        c1, c2 = st.sidebar.columns(2)
        with c1:
            per_var_mult[(v,"low")]  = st.slider(f"{v} low×", 0.25, 2.00, 1.00, 0.05, key=f"{v}_low_mult")
        with c2:
            per_var_mult[(v,"high")] = st.slider(f"{v} high×",0.25, 2.00, 1.00, 0.05, key=f"{v}_high_mult")
    else:
        per_var_mult[(v,"low")]  = 1.00
        per_var_mult[(v,"high")] = 1.00

# =============== Session State ===============
if "CURVES" not in st.session_state: st.session_state.CURVES = {}
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}
if "flash"  not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =============== Upload & Learn (MAIN AREA) ===============
st.markdown('<div class="section-title">1) Upload your database & learn curves</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False, label_visibility="visible")

# Sheet names row below, same row
c1, c2 = st.columns(2)
with c1:
    ft_sheet   = st.text_input("FT (winners) sheet name", "PMH BO FT")
with c2:
    fail_sheet = st.text_input("Fail sheet name", "PMH BO Fail")

learn_btn  = st.button("Learn curves from uploaded file", use_container_width=True)

if learn_btn:
    if not uploaded:
        st.error("Upload an Excel first.")
    else:
        try:
            learned = learn_all_curves_from_excel(uploaded, ft_sheet, fail_sheet)
            st.session_state.CURVES = learned
            if learned["learned_count"] == 0:
                st.warning("No curves learned. Check mapping below and ensure ≥30 valid rows per variable across FT+Fail.")
            else:
                st.success(f"Curves learned for {learned['learned_count']} variables. Base FT rate ≈ {learned['base_rate']*100:.1f}%.")
        except Exception as e:
            st.error(f"Learning failed: {e}")

if st.session_state.CURVES:
    with st.expander("Detected Column Mapping (click to open)"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**FT sheet mapping**")
            st.dataframe(st.session_state.CURVES.get("ft_map", pd.DataFrame()), use_container_width=True, hide_index=True)
        with cols[1]:
            st.markdown("**Fail sheet mapping**")
            st.dataframe(st.session_state.CURVES.get("fail_map", pd.DataFrame()), use_container_width=True, hide_index=True)

    with st.expander("Learned curve training summary (click to open)"):
        st.dataframe(st.session_state.CURVES.get("summary", pd.DataFrame()), use_container_width=True, hide_index=True)

st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)

# =============== Tabs ===============
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">2) Enter inputs</div>', unsafe_allow_html=True)

    with st.form("add_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1.2, 1.2, 0.9])

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
            st.caption("Catalyst / Dilution are set in the sidebar ➜")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived real-time
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted Day Vol & PM% of Predicted
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        # Curves pack
        curves_pack = st.session_state.CURVES if isinstance(st.session_state.CURVES, dict) else {}
        curves = curves_pack.get("curves", {}) if isinstance(curves_pack.get("curves", {}), dict) else {}
        prior_logodds = float(curves_pack.get("prior_logodds", 0.0))
        reliab = curves_pack.get("reliability", {}) or {}
        ref_logit = curves_pack.get("ref_logit", {}) or {}

        # Rebalance reliabilities for correlated groups
        reliab = _rebalance_group_weights(reliab)

        # Guardrails (data + domain, blended)
        guardrails = build_guardrails(curves_pack)

        # Per-variable probabilities with diagnostics + domain-prior blend (SI gets stronger blend)
        st.session_state["DEBUG_POF"] = {}
        def p_of(var_key, x) -> float:
            dbg = st.session_state.setdefault("DEBUG_POF", {})
            m = curves.get(var_key)
            if m is None:
                p_curve = 0.5
            else:
                if x is None or not np.isfinite(x):
                    dbg[var_key] = "neutral: input missing/non-finite"
                    return 0.5
                if m.get("use_log", False) and x <= 0:
                    dbg[var_key] = "neutral: input <=0 but curve is log-scale"
                    return 0.5
                try:
                    p_curve = float(m["predict"]([x])[0])
                    if not np.isfinite(p_curve):
                        p_curve = 0.5
                except Exception as e:
                    dbg[var_key] = f"neutral: curve error {e}"
                    p_curve = 0.5
            p_prior = _prior_p(var_key, float(x) if x is not None else np.nan)
            w_blend = _prior_blend_weight(var_key)
            p = (1.0 - w_blend) * p_curve + w_blend * p_prior
            p = float(np.clip(p, 1e-3, 1-1e-3))
            dbg[var_key] = f"ok: curve={p_curve:.3f}, prior={p_prior:.3f}, blend_w={w_blend:.2f}, blend={p:.3f}"
            return p

        # Evaluate all variables
        probs = {
            "gap_pct":     p_of("gap_pct", gap_pct),
            "atr_usd":     p_of("atr_usd", atr_usd),
            "rvol":        p_of("rvol", rvol),
            "si_pct":      p_of("si_pct", si_pct),
            "pm_vol_m":    p_of("pm_vol_m", pm_vol_m),
            "pm_dol_m":    p_of("pm_dol_m", pm_dol_m),
            "float_m":     p_of("float_m", float_m),
            "mcap_m":      p_of("mcap_m", mc_m),
            "fr_x":        p_of("fr_x", fr_x),
            "pmmc_pct":    p_of("pmmc_pct", pmmc_pct),
            "pm_pct_pred": p_of("pm_pct_pred", pm_pct_pred),
        }

        # --- Calibrated odds stacking (curves) ---
        z_sum = prior_logodds  # base-rate prior
        for k, p in probs.items():
            z_k = math.log(max(p,1e-6) / max(1 - p, 1e-6))
            z0  = float(ref_logit.get(k, 0.0))
            dz  = np.clip(z_k - z0, -PER_VAR_CLIP, PER_VAR_CLIP)
            w   = float(np.clip(reliab.get(k, 0.0), 0.0, 1.0))
            z_sum += TAU * w * dz

        # --- Soft tail-risk penalties for ALL variables ---
        penalty_debug = {}
        def apply_pen(var_key: str, value: float):
            band = guardrails.get(var_key, {})
            low, high = band.get("low", None), band.get("high", None)
            base_low, base_high = BASE_PEN_STRENGTH.get(var_key, (0.25, 0.25))
            kL = tail_scale * base_low  * per_var_mult.get((var_key,"low"), 1.0)
            kH = tail_scale * base_high * per_var_mult.get((var_key,"high"),1.0)
            rel_w = float(np.clip(reliab.get(var_key, 0.0), 0.0, 1.0))  # evidence-weighted
            pen = tail_penalty_bidirectional_soft(value, low, high, kL, kH, rel_w,
                                                  deadzone_frac=0.05, alpha=0.60)
            penalty_debug[var_key] = {"low":low,"high":high,"pen_raw":pen}
            return pen

        pen_list = []
        for vkey in ["gap_pct","rvol","pm_dol_m","pm_vol_m","pm_pct_pred","pmmc_pct","atr_usd","float_m","mcap_m","si_pct","fr_x"]:
            pen_list.append(apply_pen(vkey, locals().get(vkey, float("nan"))))
        pen_total = float(np.sum(pen_list))
        pen_total = float(np.clip(pen_total, -MAX_TOTAL_PEN, MAX_TOTAL_PEN))
        z_sum += PEN_TAU * pen_total
        penalty_debug["_total_penalty"] = round(PEN_TAU * pen_total, 3)

        # Modifiers (manual)
        z_sum += CATALYST_LOGODDS_COEF * float(catalyst_points)
        z_sum += DILUTION_LOGODDS_COEF * float(dilution_points)

        # Final numeric probability
        numeric_prob = 1.0 / (1.0 + math.exp(-z_sum))
        final_score  = float(np.clip(numeric_prob*100.0, 0.0, 100.0))

        # Verdict pill (based on final score only)
        if final_score >= 75.0:
            verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
        elif final_score >= 55.0:
            verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
        else:
            verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

        # Names mapping for Checklist
        names_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","pm_pct_pred":"PM Vol % of Pred"
        }

        def contribution_label(var_key: str, p: float) -> str:
            z_k = math.log(max(p,1e-6) / max(1 - p, 1e-6))
            z0  = float(ref_logit.get(var_key, 0.0))
            dz  = float(np.clip(z_k - z0, -PER_VAR_CLIP, PER_VAR_CLIP))
            w   = float(np.clip(reliab.get(var_key, 0.0), 0.0, 1.0))
            eff = TAU * w * dz
            if eff >= 0.25:        return "supports"
            elif eff <= -0.25:     return "headwind"
            else:                  return "neutral"

        # Build readable checklist rows with values + probs
        checklist_rows = []
        value_map = {
            "Gap %": gap_pct, "ATR $": atr_usd, "RVOL": rvol, "Short Interest %": si_pct,
            "PM Volume (M)": pm_vol_m, "PM $Vol (M)": pm_dol_m, "Float (M)": float_m,
            "MarketCap (M)": mc_m, "PM Float Rotation ×": fr_x, "PM $Vol / MC %": pmmc_pct,
            "PM Vol % of Pred": pm_pct_pred,
        }
        # Prediction info first
        if np.isfinite(pred_vol_m):
            pred_ci68 = f"{ci68_l:.2f}–{ci68_u:.2f}M"
            pred_ci95 = f"{ci95_l:.2f}–{ci95_u:.2f}M"
            checklist_rows.append(f"- **Predicted Day Volume**: {pred_vol_m:.2f}M  (CI68 {pred_ci68} · CI95 {pred_ci95})")
        if np.isfinite(pm_pct_pred):
            checklist_rows.append(f"- **PM Vol / Predicted Day Vol**: {pm_pct_pred:.1f}%")

        for key, p in probs.items():
            label = names_map.get(key, key)
            role = contribution_label(key, p)
            v = value_map.get(label, None)
            if v is None or not np.isfinite(v):
                val_str = "—"
            else:
                if abs(v) >= 1000: val_str = f"{v:,.0f}"
                elif abs(v) >= 100: val_str = f"{v:.0f}"
                elif abs(v) >= 10:  val_str = f"{v:.1f}"
                else:               val_str = f"{v:.2f}"
            checklist_rows.append(f"- **{label}**: {val_str} → *{role}* (p≈{p*100:.0f}%)")

        # Manual modifiers at the end (explicit)
        if abs(catalyst_points) > 1e-9:
            checklist_rows.append(f"- **Catalyst (manual)**: {catalyst_points:+.2f} → Δlogodds ≈ {CATALYST_LOGODDS_COEF*catalyst_points:+.2f}")
        if abs(dilution_points) > 1e-9:
            checklist_rows.append(f"- **Dilution (manual)**: {dilution_points:+.2f} → Δlogodds ≈ {DILUTION_LOGODDS_COEF*dilution_points:+.2f}")

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "FinalScore": round(final_score, 2),

            # Prediction fields (kept for table export)
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),

            # Inputs for export/debug
            "_MCap_M": mc_m, "_Gap_%": gap_pct, "_SI_%": si_pct, "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m, "_Float_M": float_m,
            "_Catalyst": float(catalyst_points), "_Dilution": float(dilution_points),

            # Display
            "PremarketVerdict": verdict,
            "PremarketPill": pill,
            "PremarketChecklist": checklist_rows,

            # Debug penalties
            "_TailPenalties": penalty_debug,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ========== Preview card ==========
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC = st.columns(3)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Odds", l.get("Odds","—"))

        with st.expander("Premarket Checklist", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            pill = l.get("PremarketPill","")
            st.markdown(f"**Verdict:** {pill if pill else verdict}", unsafe_allow_html=True)
            items = l.get("PremarketChecklist", [])
            if not items:
                st.caption("No checklist yet.")
            else:
                st.markdown("\n".join(items))

        # Diagnostics
        with st.expander("Diagnostics (curves & tail penalties)"):
            dbg = st.session_state.get("DEBUG_POF", {})
            if dbg:
                dd = pd.DataFrame([{"Variable": k, "Status": v} for k, v in dbg.items()])
                st.dataframe(dd, hide_index=True, use_container_width=True)

            pen_debug = l.get("_TailPenalties", {})
            if pen_debug:
                pen_rows = []
                for k, v in pen_debug.items():
                    if isinstance(v, dict):
                        pen_rows.append({
                            "Variable": k,
                            "Low": v.get("low", None),
                            "High": v.get("high", None),
                            "Penalty (Δlogodds)": round(v.get("pen_raw", 0.0), 3)
                        })
                    else:
                        pen_rows.append({
                            "Variable": k,
                            "Low": "",
                            "High": "",
                            "Penalty (Δlogodds)": round(float(v), 3)
                        })
                st.dataframe(pd.DataFrame(pen_rows), hide_index=True, use_container_width=True)

# =============== Ranking Tab ===============
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "FinalScore",
            "PremarketVerdict",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","PremarketVerdict") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Level"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Premarket Verdict"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
            }
        )

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Delete rows</div>', unsafe_allow_html=True)
        del_cols = st.columns(4)
        head12 = df.head(12).reset_index(drop=True)
        for i, r in head12.iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    do_rerun()

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
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Upload your DB and click **Learn curves**, then add a stock in the **Add Stock** tab.")
