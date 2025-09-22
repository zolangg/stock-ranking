import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List

# =============== Page & Styles ===============
st.set_page_config(page_title="Premarket Stock Ranking (Smooth Curves)", layout="wide")
st.title("Premarket Stock Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151; }
  .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
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

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# =============== Sidebar: Upload & Learn ===============
st.sidebar.header("Database (learn smooth curves)")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
ft_sheet   = st.sidebar.text_input("FT (winners) sheet name", "PMH BO FT")
fail_sheet = st.sidebar.text_input("Fail sheet name", "PMH BO Fail")
learn_btn  = st.sidebar.button("Learn curves from uploaded file", use_container_width=True)

# Qualitative block (unchanged)
st.sidebar.header("Qualitative Weights")
QUAL_CRITERIA = [
    {"name":"GapStruct","question":"Gap & Trend Development:","options":[
        "Gap fully reversed: price loses >80% of gap.",
        "Choppy reversal: price loses 50–80% of gap.",
        "Partial retracement: price loses 25–50% of gap.",
        "Sideways consolidation: gap holds, price within top 25% of gap.",
        "Uptrend with deep pullbacks (>30% retrace).",
        "Uptrend with moderate pullbacks (10–30% retrace).",
        "Clean uptrend, only minor pullbacks (<10%).",
    ],"weight":0.15},
    {"name":"LevelStruct","question":"Key Price Levels:","options":[
        "Fails at all major support/resistance; cannot hold any key level.",
        "Briefly holds/reclaims a level but loses it quickly; repeated failures.",
        "Holds one support but unable to break resistance; capped below a key level.",
        "Breaks above resistance but cannot stay; dips below reclaimed level.",
        "Breaks and holds one major level; most resistance remains above.",
        "Breaks and holds several major levels; clears most overhead resistance.",
        "Breaks and holds above all resistance; blue sky.",
    ],"weight":0.15},
    {"name":"Monthly","question":"Monthly/Weekly Chart Context:","options":[
        "Sharp, accelerating downtrend; new lows repeatedly.",
        "Persistent downtrend; still lower lows.",
        "Downtrend losing momentum; flattening.",
        "Clear base; sideways consolidation.",
        "Bottom confirmed; higher low after base.",
        "Uptrend begins; breaks out of base.",
        "Sustained uptrend; higher highs, blue sky.",
    ],"weight":0.10},
]
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(crit["name"], 0.0, 1.0, crit["weight"], 0.01)
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights: q_weights[k] = q_weights[k] / qual_sum

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01)

# =============== Session State ===============
if "CURVES" not in st.session_state: st.session_state.CURVES = {}
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}
if "flash"  not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

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
        "short interest %","short interest","si %","shortinterest %",
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
    """
    Learn smooth P(FT|x) via Gaussian-kernel Nadaraya–Watson in transformed space.
    Auto log-transform for positive skewed vars. Needs ~MIN_ROWS usable rows.
    """
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
        x_raw = x_raw[pos]
        y_raw = y_raw[pos]
        if len(x_raw) < MIN_ROWS:
            return None
        x_t = np.log(x_raw)
    else:
        x_t = x_raw.copy()

    std = np.std(x_t)
    n = len(x_t)
    h = 0.6 * std * (n ** (-1/5)) if std > 0 else 0.3

    lo = np.nanpercentile(x_t, 2)
    hi = np.nanpercentile(x_t, 98)
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
            ok = ~bad
            z[ok] = np.log(x_new[ok])
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

def learn_all_curves_from_excel(file, ft_sheet: str, fail_sheet: str) -> Dict[str, Any]:
    xls = pd.ExcelFile(file)
    if ft_sheet not in xls.sheet_names or fail_sheet not in xls.sheet_names:
        raise ValueError(f"Sheets not found. Available: {xls.sheet_names}")

    ft_raw   = pd.read_excel(xls, ft_sheet)
    fail_raw = pd.read_excel(xls, fail_sheet)

    ft_map = reveal_mapping(ft_raw)
    fail_map = reveal_mapping(fail_raw)

    ft_num   = build_numeric_table(ft_raw)
    fail_num = build_numeric_table(fail_raw)

    def usable_counts(df: pd.DataFrame) -> pd.DataFrame:
      keys = ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m",
              "float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","daily_vol_m"]
      rows = []
      for k in keys:
          if k in df.columns:
              s = pd.to_numeric(df[k], errors="coerce")
              ok = s[np.isfinite(s)]
              pos = ok[ok > 0]
              rows.append({
                  "var": k,
                  "non_null": int(ok.shape[0]),
                  "positive": int(pos.shape[0]),
                  "unique": int(ok.nunique())
              })
          else:
              rows.append({"var": k, "non_null": 0, "positive": 0, "unique": 0})
      return pd.DataFrame(rows)
  
  # inside learn_all_curves_from_excel(...)
  ft_num   = build_numeric_table(ft_raw)
  fail_num = build_numeric_table(fail_raw)
  all_num  = pd.concat([ft_num.assign(_y=1), fail_num.assign(_y=0)], ignore_index=True)
  
  st.markdown("**Usable rows (after cleaning)**")
  c1, c2, c3 = st.columns(3)
  with c1: st.caption("FT usable");  st.dataframe(usable_counts(ft_num), use_container_width=True, hide_index=True)
  with c2: st.caption("Fail usable"); st.dataframe(usable_counts(fail_num), use_container_width=True, hide_index=True)
  with c3: st.caption("Combined");    st.dataframe(usable_counts(all_num), use_container_width=True, hide_index=True)

    ft_num["_y"] = 1
    fail_num["_y"] = 0
    all_num = pd.concat([ft_num, fail_num], axis=0, ignore_index=True)

    var_list = [
        "gap_pct","atr_usd","rvol","si_pct",
        "pm_vol_m","pm_dol_m","float_m","mcap_m",
        "fr_x","pmmc_pct","pm_pct_pred"
    ]
    curves = {}
    learned = 0
    for v in var_list:
        if v in all_num.columns:
            model = fit_curve(all_num[v], all_num["_y"])
            if model:
                curves[v] = model
                learned += 1

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

    return {"curves": curves, "summary": summary_df, "ft_map": ft_map, "fail_map": fail_map, "learned_count": learned}

# =============== Qualitative % (unchanged) ===============
def qualitative_percent(q_weights: Dict[str, float]) -> float:
    qual_0_7 = 0.0
    for crit in QUAL_CRITERIA:
        key = f"qual_{crit['name']}"
        sel = st.session_state.get(key, (1,))[0] if isinstance(st.session_state.get(key, 1), tuple) else st.session_state.get(key, 1)
        qual_0_7 += q_weights[crit["name"]] * float(sel)
    return (qual_0_7/7.0)*100.0

# =============== Odds Stacking ===============
def prob_from_logodds(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def combine_probs_oddsstack(probs: List[float]) -> float:
    z_sum = 0.0
    for p in probs:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        z_sum += math.log(p / (1 - p))
    return prob_from_logodds(z_sum)

# Sliders → log-odds shifts per +1.0
CATALYST_LOGODDS_COEF = 0.5    # ~ +12 ppt around mid
DILUTION_LOGODDS_COEF = -0.5   # ~ -12 ppt around mid

# =============== Learn Button ===============
if learn_btn:
    if not uploaded:
        st.sidebar.error("Upload an Excel first.")
    else:
        try:
            learned = learn_all_curves_from_excel(uploaded, ft_sheet, fail_sheet)
            st.session_state.CURVES = learned
            if learned["learned_count"] == 0:
                st.sidebar.warning("No curves learned. Check the mapping tables below and ensure ≥30 valid rows per variable across FT+Fail.")
            else:
                st.sidebar.success(f"Curves learned for {learned['learned_count']} variables.")
        except Exception as e:
            st.sidebar.error(f"Learning failed: {e}")

# =============== Show mapping & training summary ===============
if st.session_state.CURVES:
    st.markdown('<div class="section-title">Detected Column Mapping</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**FT sheet mapping**")
        st.dataframe(st.session_state.CURVES.get("ft_map", pd.DataFrame()), use_container_width=True, hide_index=True)
    with cols[1]:
        st.markdown("**Fail sheet mapping**")
        st.dataframe(st.session_state.CURVES.get("fail_map", pd.DataFrame()), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Learned curve training summary</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.CURVES.get("summary", pd.DataFrame()), use_container_width=True, hide_index=True)

# =============== Tabs ===============
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Numeric & Modifiers</div>', unsafe_allow_html=True)

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
            st.markdown("**Modifiers**")
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Qualitative Context</div>', unsafe_allow_html=True)

        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                )

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

        # Curves
        curves = (st.session_state.CURVES or {}).get("curves", {}) if isinstance(st.session_state.CURVES, dict) else {}

        # Per-variable probabilities with diagnostics
        st.session_state["DEBUG_POF"] = {}
        def p_of(var_key, x) -> float:
            dbg = st.session_state.setdefault("DEBUG_POF", {})
            m = curves.get(var_key)
            if m is None:
                dbg[var_key] = "neutral: no curve learned"
                return 0.5
            if x is None or not np.isfinite(x):
                dbg[var_key] = "neutral: input missing/non-finite"
                return 0.5
            if m["use_log"] and x <= 0:
                dbg[var_key] = "neutral: input <=0 but curve is log-scale"
                return 0.5
            try:
                p = float(m["predict"]([x])[0])
                if not np.isfinite(p):
                    dbg[var_key] = "neutral: prediction NaN"
                    return 0.5
                dbg[var_key] = f"ok: {p:.3f}"
                return max(1e-3, min(1 - 1e-3, p))
            except Exception as e:
                dbg[var_key] = f"neutral: error {e}"
                return 0.5

        # Evaluate all PM variables you use
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

        # Combine via odds stacking + modifiers
        z_sum = 0.0
        for p in probs.values():
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            z_sum += math.log(p / (1 - p))
        z_sum += CATALYST_LOGODDS_COEF * float(catalyst_points)
        z_sum += DILUTION_LOGODDS_COEF * float(dilution_points)
        numeric_prob = prob_from_logodds(z_sum)
        numeric_pct = 100.0 * numeric_prob

        # Qualitative %
        qual_pct = qualitative_percent(q_weights)

        # Final score
        final_score = float(max(0.0, min(100.0, 0.5*numeric_pct + 0.5*qual_pct)))

        # Verdict pill
        if numeric_pct >= 75.0:
            verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
        elif numeric_pct >= 55.0:
            verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
        else:
            verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

        # Checklist buckets by per-variable probability
        def bucket(name, p):
            if p >= 0.60: return ("good", f"{name}: supports FT ({p*100:.0f}%)")
            if p >= 0.45: return ("warn", f"{name}: neutral ({p*100:.0f}%)")
            return ("risk", f"{name}: headwind ({p*100:.0f}%)")

        names_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","pm_pct_pred":"PM Vol % of Pred"
        }
        good, warn, risk = [], [], []
        for k,v in probs.items():
            cat, txt = bucket(names_map.get(k,k), v)
            if cat=="good": good.append(txt)
            elif cat=="warn": warn.append(txt)
            else: risk.append(txt)

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(numeric_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": round(final_score, 2),

            # Prediction fields (for info & PM% of Pred in checklist)
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),
            "PM_%_of_Pred": round(100.0 * pm_vol_m / pred_vol_m, 1) if pred_vol_m > 0 else "",

            # Inputs for export/debug
            "_MCap_M": mc_m, "_Gap_%": gap_pct, "_SI_%": si_pct, "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m, "_Float_M": float_m,
            "_Catalyst": float(catalyst_points), "_Dilution": float(dilution_points),

            # Checklist sets
            "PremarketVerdict": verdict,
            "PremarketPill": pill,
            "PremarketGood": good,
            "PremarketWarn": warn,
            "PremarketRisk": risk,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ========== Preview card ==========
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC, cD, cE = st.columns(5)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Numeric % (Smooth FT)", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qualitative %",         f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",           f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        d1, d2 = st.columns(2)
        d1.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d1.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        pmpred = l.get("PM_%_of_Pred","")
        if pmpred != "":
            d2.metric("PM Vol / Predicted Day Vol", f"{pmpred}%")

        with st.expander("Premarket Checklist (smooth, data-driven)", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            pill = l.get("PremarketPill","")
            st.markdown(f"**Verdict:** {pill if pill else verdict}", unsafe_allow_html=True)

            if pmpred != "":
                st.markdown(f"<div class='mono'>PM Vol / Pred. Day Vol: {pmpred}%</div>", unsafe_allow_html=True)

            g_col, w_col, r_col = st.columns(3)
            def _ul(items):
                if not items:
                    return "<ul><li><span class='hint'>None</span></li></ul>"
                return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

            with g_col:
                st.markdown("**Good**"); st.markdown(_ul(l.get("PremarketGood", [])), unsafe_allow_html=True)
            with w_col:
                st.markdown("**Caution**"); st.markdown(_ul(l.get("PremarketWarn", [])), unsafe_allow_html=True)
            with r_col:
                st.markdown("**Risk**"); st.markdown(_ul(l.get("PremarketRisk", [])), unsafe_allow_html=True)

        # Why-neutral diagnostics
        with st.expander("Why are some variables neutral?", expanded=False):
            dbg = st.session_state.get("DEBUG_POF", {})
            if dbg:
                dd = pd.DataFrame([{"Variable": k, "Status": v} for k,v in dbg.items()])
                st.dataframe(dd, hide_index=True, use_container_width=True)
            else:
                st.caption("No diagnostics yet. Add a stock after learning curves.")

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
            "Numeric_%","Qual_%","FinalScore",
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
                "Numeric_%": st.column_config.NumberColumn("Numeric_% (Smooth FT)", format="%.2f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
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
