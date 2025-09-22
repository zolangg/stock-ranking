import streamlit as st
import pandas as pd
import numpy as np
import math
from typing import Optional, Dict, Any

# =========================
# Page + styles
# =========================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
  .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
  .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0;}
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa;}
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
  ul { margin-top: 4px; margin-bottom: 4px; padding-left: 18px; }
  li { margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Small helpers
# =========================
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

def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
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
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

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

# =========================
# Prediction model (kept)
# =========================
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

# =========================
# Sidebar – Upload DB & settings
# =========================
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])
learn_sheet = st.sidebar.text_input(
    "Sheet to learn distributions from (e.g., 'PMH BO FT', 'PMH BO Merged')",
    value="PMH BO FT",
    help="Use your FT (follow-through) sheet for winners-only percentiles."
)

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

# =========================
# Data learning utilities
# =========================
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("\n"," ") for c in df.columns]
    return df

def _numify(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(str).str.replace(",","", regex=False).str.replace("'","", regex=False).str.strip()
    s = s.str.replace("%","", regex=False)
    return pd.to_numeric(s, errors="coerce")

def learn_distributions_from_excel(file, sheet_name: str):
    """
    Returns:
      MODELS: dict of per-feature models (for smooth scoring on transformed axis)
      DISPLAY_Q: dict of per-feature original-axis quantiles for checklist:
        {'p10','p25','p50','p75','p90'}
    """
    try:
        xls = pd.ExcelFile(file)
    except Exception:
        return None

    # choose sheet
    if sheet_name not in xls.sheet_names:
        candidates = [s for s in xls.sheet_names if sheet_name.lower() in s.lower()]
        sheet = candidates[0] if candidates else xls.sheet_names[0]
    else:
        sheet = sheet_name

    raw = pd.read_excel(xls, sheet_name=sheet)
    df = _clean_cols(raw)

    def pick(*names):
        for n in names:
            if n in df.columns: return n
        for c in df.columns:
            cc = c.replace(" ", "")
            if "float" in cc and ("mshares" in cc or "publicfloat" in cc): return c
        return None

    col_float = pick("float m shares","public float (m)","float (m)","float_m")
    col_mcap  = pick("marketcap m","market cap (m)","market cap m","mcap")
    col_gap   = pick("gap %","gap%","gap pct","premarket gap %")
    col_pm_d  = pick("pm $vol (m)","pm $ vol (m)","pm dollar vol (m)","premarket $vol (m)")
    col_rvol  = pick("rvol @ bo","rvol","relative volume","rv ol bo","rvol bo")
    col_atr   = pick("atr","atr $","atr (usd)","atr $/day")

    cols = {}
    if col_float: cols["float_m"]  = _numify(df[col_float])
    if col_mcap:  cols["mcap_m"]   = _numify(df[col_mcap])
    if col_gap:
        gap = _numify(df[col_gap])
        # if mostly ratio (<5), convert to percent
        if gap.dropna().lt(5).mean() > 0.6: gap = gap * 100.0
        cols["gap_pct"] = gap
    if col_pm_d:   cols["pm_dol_m"] = _numify(df[col_pm_d])
    if col_rvol:   cols["rvol"]     = _numify(df[col_rvol])
    if col_atr:    cols["atr_usd"]  = _numify(df[col_atr])

    num = pd.DataFrame(cols)

    # derived: PM $Vol / MC %
    if "pm_dol_m" in num.columns and "mcap_m" in num.columns:
        num["pm_mc_pct"] = 100.0 * num["pm_dol_m"] / num["mcap_m"]

    for k in num.columns:
        num.loc[num[k] == 0, k] = np.nan

    MODELS = {}
    DISPLAY_Q = {}

    features = ["float_m","mcap_m","atr_usd","gap_pct","pm_dol_m","pm_mc_pct","rvol"]
    for f in features:
        if f not in num.columns: continue
        s = num[f].dropna().astype(float)
        if len(s) < 20:
            continue

        # decide transform: log if positive & skewed
        use_log = False
        if (s > 0).all():
            skew = s.skew()
            if abs(skew) > 0.75:
                use_log = True

        x = np.log(s.values) if use_log else s.values
        x = x[np.isfinite(x)]
        if len(x) < 20: continue

        # original-axis percentiles for checklist
        qs = np.quantile(s, [0.10,0.25,0.50,0.75,0.90])
        DISPLAY_Q[f] = dict(p10=qs[0], p25=qs[1], p50=qs[2], p75=qs[3], p90=qs[4])

        # empirical CDF on transformed axis
        vals_t = np.sort(x)
        n = len(vals_t)
        perc = (np.arange(n) + 0.5) / n

        # quantiles on transformed axis
        q1_t, med_t, q3_t = np.quantile(x, [0.25,0.50,0.75])
        iqr_t = max(1e-9, q3_t - q1_t)
        sigma_l = max((med_t - q1_t) / 1.349, 1e-6)
        sigma_r = max((q3_t - med_t) / 1.349, 1e-6)

        MODELS[f] = dict(
            vals_t=vals_t, perc=perc,
            q=dict(q1_t=q1_t, med_t=med_t, q3_t=q3_t),
            iqr=iqr_t, med_t=med_t,
            sigma_l=sigma_l, sigma_r=sigma_r,
            use_log=use_log
        )

    return MODELS, DISPLAY_Q

# =========================
# Session state: models & rows
# =========================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}
if "DISPLAY_Q" not in st.session_state: st.session_state.DISPLAY_Q = {}
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# learn from uploaded file (if provided)
if uploaded is not None:
    learned = learn_distributions_from_excel(uploaded, learn_sheet)
    if learned is not None:
        st.session_state.MODELS, st.session_state.DISPLAY_Q = learned

MODELS = st.session_state.MODELS
DISPLAY_Q = st.session_state.DISPLAY_Q

# =========================
# Smooth scoring
# =========================
EPS_FLOOR = 0.02  # never zero

def _to_t(x, use_log: bool):
    if x is None: return None
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x): return None
    if use_log:
        if x <= 0: return None
        return math.log(x)
    return x

def score_feature(x, model):
    """
    Hybrid score:
      - percentile score s_p on empirical CDF (peak at 0.5; width ~0.25)
      - asymmetric Gaussian s_g around median in transformed space
      - final = sqrt(s_p * s_g), floored at EPS_FLOOR
    """
    if not model: 
        return 0.5  # neutral when no model
    xt = _to_t(x, model["use_log"])
    if xt is None: 
        return EPS_FLOOR

    vals_t = model["vals_t"]; perc = model["perc"]

    if xt <= vals_t[0]: p = 1e-6
    elif xt >= vals_t[-1]: p = 1 - 1e-6
    else:
        p = float(np.interp(xt, vals_t, perc))

    # Percentile bell around 0.5
    width = 0.25  # can make dynamic if desired
    z_p = (p - 0.5) / width
    s_p = math.exp(-0.5 * z_p * z_p)

    # Asymmetric Gaussian around median
    med_t = model["med_t"]; sigma_l = model["sigma_l"]; sigma_r = model["sigma_r"]
    if xt < med_t:
        z = (xt - med_t) / sigma_l
    else:
        z = (xt - med_t) / sigma_r
    s_g = math.exp(-0.5 * z * z)

    return max(EPS_FLOOR, math.sqrt(s_p * s_g))

# Weights for Numeric % (sum ≈ 1)
W = {
    "float_m":   0.18,
    "mcap_m":    0.15,
    "atr_usd":   0.12,
    "gap_pct":   0.18,
    "pm_dol_m":  0.12,
    "pm_mc_pct": 0.07,   # NEW feature
    "rvol":      0.16,
    "catalyst":  0.01,
    "dilution":  0.01,
}
Wsum = sum(W.values()) or 1.0
for k in W: W[k] = W[k]/Wsum

# =========================
# Qualitative weights (kept)
# =========================
def qualitative_percent(q_weights: Dict[str, float]) -> float:
    qual_0_7 = 0.0
    for crit in QUAL_CRITERIA:
        key = f"qual_{crit['name']}"
        sel = st.session_state.get(key, (1,))[0] if isinstance(st.session_state.get(key, 1), tuple) else st.session_state.get(key, 1)
        qual_0_7 += q_weights[crit["name"]] * float(sel)
    return (qual_0_7/7.0)*100.0

# =========================
# Checklist builder (percentile bands + info lines)
# =========================
def percentile_band_text(name: str, x: Optional[float], feature_key: str, fmt: str, good, warn, risk):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        warn.append(f"{name}: missing"); return
    q = DISPLAY_Q.get(feature_key)
    val = fmt.format(x) if isinstance(x, (int,float)) else str(x)
    if not q:
        warn.append(f"{name}: {val}")
        return
    p10, p25, p50, p75, p90 = q["p10"], q["p25"], q["p50"], q["p75"], q["p90"]
    if p25 <= x <= p75:
        good.append(f"{name} in 25–75 pct (you {val})")
    elif (p10 <= x < p25) or (p75 < x <= p90):
        warn.append(f"{name} in 10–25/75–90 pct (you {val})")
    else:
        risk.append(f"{name} in tails (<10 or >90 pct; you {val})")

def make_premarket_checklist(
    *, float_m: float, mcap_m: float, atr_usd: float,
    gap_pct: float, pm_vol_m: float, pm_dol_m: float,
    rvol: float, catalyst_points: float, dilution_points: float,
    pm_mc_pct: Optional[float], pm_pct_of_pred: Optional[float]
) -> Dict[str, Any]:
    good, warn, risk = [], [], []

    # Per-feature smooth scores
    s_float = score_feature(float_m,   MODELS.get("float_m"))
    s_mcap  = score_feature(mcap_m,    MODELS.get("mcap_m"))
    s_atr   = score_feature(atr_usd,   MODELS.get("atr_usd"))
    s_gap   = score_feature(gap_pct,   MODELS.get("gap_pct"))
    s_pm_d  = score_feature(pm_dol_m,  MODELS.get("pm_dol_m"))
    s_pmmc  = score_feature(pm_mc_pct, MODELS.get("pm_mc_pct")) if pm_mc_pct is not None else 0.5
    s_rvol  = score_feature(rvol,      MODELS.get("rvol"))
    s_cat   = score_feature(catalyst_points, MODELS.get("catalyst")) if "catalyst" in MODELS else 0.5
    s_dil   = score_feature(dilution_points, MODELS.get("dilution")) if "dilution" in MODELS else 0.5

    # Checklist text via percentile bands
    percentile_band_text("Float (M)",            float_m,   "float_m",   "{:.2f}", good, warn, risk)
    percentile_band_text("Market Cap ($M)",      mcap_m,    "mcap_m",    "{:.0f}", good, warn, risk)
    percentile_band_text("ATR ($)",              atr_usd,   "atr_usd",   "{:.2f}", good, warn, risk)
    percentile_band_text("Gap (%)",              gap_pct,   "gap_pct",   "{:.1f}", good, warn, risk)
    percentile_band_text("Premarket $Vol ($M)",  pm_dol_m,  "pm_dol_m",  "{:.1f}", good, warn, risk)
    percentile_band_text("PM $Vol / MC (%)",     pm_mc_pct, "pm_mc_pct", "{:.1f}", good, warn, risk)
    percentile_band_text("RVOL (×)",             rvol,      "rvol",      "{:.0f}", good, warn, risk)

    # Informational line requested: PM Vol / Predicted Vol %
    if pm_pct_of_pred is not None and pm_pct_of_pred == pm_pct_of_pred:
        warn.append(f"Info: PM Vol / Predicted Day Vol = {pm_pct_of_pred:.1f}%")

    # Weighted Numeric %
    soft = (
        W["float_m"]   * s_float +
        W["mcap_m"]    * s_mcap  +
        W["atr_usd"]   * s_atr   +
        W["gap_pct"]   * s_gap   +
        W["pm_dol_m"]  * s_pm_d  +
        W["pm_mc_pct"] * s_pmmc  +
        W["rvol"]      * s_rvol  +
        W["catalyst"]  * s_cat   +
        W["dilution"]  * s_dil
    )
    numeric_pct = float(max(0.0, min(100.0, 100.0 * soft)))

    # Verdict from same signal (so it aligns)
    if numeric_pct >= 75.0:
        verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
    elif numeric_pct >= 55.0:
        verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
    else:
        verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

    # Greens/Reds counters by percentile bands
    greens = 0; reds = 0
    for key, val in [("float_m", float_m), ("mcap_m", mcap_m), ("atr_usd", atr_usd),
                     ("gap_pct", gap_pct), ("pm_dol_m", pm_dol_m), ("pm_mc_pct", pm_mc_pct),
                     ("rvol", rvol)]:
        q = DISPLAY_Q.get(key)
        if q and val is not None and np.isfinite(val):
            if q["p25"] <= val <= q["p75"]:
                greens += 1
            elif val < q["p10"] or val > q["p90"]:
                reds += 1

    return {
        "greens": greens, "reds": reds,
        "good": good, "warn": warn, "risk": risk,
        "verdict": verdict, "pill": pill,
        "phb_numeric": numeric_pct
    }

# =========================
# Tabs
# =========================
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
        # Predicted day volume (M) + CIs
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        # PM diagnostics used in checklist
        pm_mc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")
        pm_pct_of_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        # Qualitative %
        qual_pct = qualitative_percent(q_weights)

        # Checklist + Numeric %
        ck = make_premarket_checklist(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_vol_m=pm_vol_m, pm_dol_m=pm_dol_m,
            rvol=rvol, catalyst_points=catalyst_points, dilution_points=dilution_points,
            pm_mc_pct=pm_mc_pct, pm_pct_of_pred=pm_pct_of_pred
        )
        numeric_pct = ck["phb_numeric"]

        # Final Score = 50/50 Numeric & Qual
        final_score = float(max(0.0, min(100.0, 0.5*numeric_pct + 0.5*qual_pct)))

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(numeric_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": round(final_score, 2),

            # Prediction fields
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            # raw inputs
            "_MCap_M": mc_m, "_Gap_%": gap_pct, "_SI_%": si_pct, "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m, "_Float_M": float_m,
            "_Catalyst": float(catalyst_points), "_Dilution": float(dilution_points),

            # checklist
            "PremarketVerdict": ck["verdict"],
            "PremarketPill": ck["pill"],
            "PremarketGreens": ck["greens"],
            "PremarketReds": ck["reds"],
            "PremarketGood": ck["good"],
            "PremarketWarn": ck["warn"],
            "PremarketRisk": ck["risk"],

            # info lines for expander
            "PM_%_of_Pred": round(pm_pct_of_pred,1) if pm_pct_of_pred==pm_pct_of_pred else "",
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC, cD, cE = st.columns(5)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Numeric % (Smooth)", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qualitative %",      f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",        f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        # We remove PM Float Rotation / PM $Vol / MC / Gap tiles as requested.
        d1, d2 = st.columns(2)
        d1.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d1.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        d2.metric("—", " ")  # spacer for clean layout

        with st.expander("Premarket Checklist (smooth, data-driven)", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            greens = l.get("PremarketGreens",0)
            reds = l.get("PremarketReds",0)
            pill = l.get("PremarketPill","")
            st.markdown(
                f"**Verdict:** {pill if pill else verdict} &nbsp;&nbsp;|&nbsp; ✅ {greens} &nbsp;·&nbsp; 🚫 {reds}",
                unsafe_allow_html=True
            )

            # Requested informational line
            pmpred = l.get("PM_%_of_Pred","")
            if pmpred != "":
                st.markdown(f"**PM Vol / Predicted Day Vol:** {pmpred}%")

            g_col, w_col, r_col = st.columns(3)
            def _ul(items):
                if not items:
                    return "<ul><li><span class='hint'>None</span></li></ul>"
                return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"

            with g_col:
                st.markdown("**Good**")
                st.markdown(_ul(l.get("PremarketGood", [])), unsafe_allow_html=True)
            with w_col:
                st.markdown("**Caution**")
                st.markdown(_ul(l.get("PremarketWarn", [])), unsafe_allow_html=True)
            with r_col:
                st.markdown("**Risk**")
                st.markdown(_ul(l.get("PremarketRisk", [])), unsafe_allow_html=True)

# =========================
# Ranking tab
# =========================
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
            "PremarketVerdict","PremarketReds","PremarketGreens",
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
                "Numeric_%": st.column_config.NumberColumn("Numeric_% (Smooth)", format="%.2f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Premarket Verdict"),
                "PremarketReds": st.column_config.NumberColumn("🚫 Reds"),
                "PremarketGreens": st.column_config.NumberColumn("✅ Greens"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)", format="%.2f"),
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
        st.info("No rows yet. Upload your DB (optional), then add a stock in the **Add Stock** tab.")
