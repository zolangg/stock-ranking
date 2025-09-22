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
# Helpers
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
    "Sheet to learn IQR from (e.g., 'PMH BO FT', 'PMH BO Merged')",
    value="PMH BO FT",
    help="Use your FT (follow-through) sheet for winners-only quartiles."
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
# Learn IQRs from uploaded DB
# =========================
DEFAULT_IQR = {
    # Fallbacks if no file or column missing (units in comments)
    "float_m":   dict(lo=0.5,  q1=5.0,  med=8.5,  q3=12.0, hi=50.0),   # M shrs
    "mcap_m":    dict(lo=5.0,  q1=30.0, med=80.0, q3=150.0,hi=500.0),  # $M
    "atr_usd":   dict(lo=0.10, q1=0.20, med=0.30, q3=0.40, hi=1.00),   # $
    "gap_pct":   dict(lo=30.0, q1=70.0, med=120.0,q3=180.0,hi=280.0),  # %
    "pm_dol_m":  dict(lo=3.0,  q1=7.0,  med=11.0, q3=15.0, hi=40.0),   # $M
    "rvol":      dict(lo=50.0, q1=100.0,med=680.0,q3=1500.0,hi=3000.0),# ×
    "pm_mc_pct": dict(lo=0.5,  q1=1.0,  med=3.0,  q3=6.0,  hi=20.0),   # %
}

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("\n"," ") for c in df.columns]
    return df

def _numify(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(str).str.replace(",","", regex=False).str.replace("'","", regex=False).str.strip()
    s = s.str.replace("%","", regex=False)
    return pd.to_numeric(s, errors="coerce")

def learn_iqr_from_excel(file, sheet_name: str) -> Dict[str, Dict[str, float]]:
    try:
        xls = pd.ExcelFile(file)
    except Exception:
        return DEFAULT_IQR.copy()

    if sheet_name not in xls.sheet_names:
        # try to find something close
        candidates = [s for s in xls.sheet_names if sheet_name.lower() in s.lower()]
        sheet = candidates[0] if candidates else xls.sheet_names[0]
    else:
        sheet = sheet_name

    try:
        raw = pd.read_excel(xls, sheet_name=sheet)
        df = _clean_cols(raw)
    except Exception:
        return DEFAULT_IQR.copy()

    # Map likely columns
    def pick(*names):
        for n in names:
            if n in df.columns: return n
        # fuzzy attempts
        for c in df.columns:
            cc = c.replace(" ", "")
            if any(k in cc for k in ["floatmshares","publicfloat","float(m)"]) and "float" in cc:
                return c
        return None

    col_float = pick("float m shares","public float (m)","float (m)","float_m")
    col_mcap  = pick("marketcap m","market cap (m)","market cap m","mcap")
    col_gap   = pick("gap %","gap%","gap pct","premarket gap %")
    col_pmdol   = pick("pm $vol (m)","pm $ vol (m)","pm dollar vol (m)","premarket $vol (m)")
    col_rvol  = pick("rvol @ bo","rvol","relative volume","rv ol bo","rvol bo")
    col_atr   = pick("atr","atr $","atr (usd)","atr $/day")

    cols = {}
    if col_float: cols["float_m"]  = _numify(df[col_float])
    if col_mcap:  cols["mcap_m"]   = _numify(df[col_mcap])
    if col_gap:
        gap = _numify(df[col_gap])
        # if mostly ratio (e.g., 0.84), convert to %
        if gap.dropna().lt(5).mean() > 0.6: gap = gap * 100.0
        cols["gap_pct"] = gap
    if col_pmdol:    cols["pm_dol_m"] = _numify(df[col_pmdol])
    if col_rvol:   cols["rvol"]     = _numify(df[col_rvol])
    if col_atr:    cols["atr_usd"]  = _numify(df[col_atr])

    num = pd.DataFrame(cols)
    # Derived: PM $Vol / MC %
    if "pm_dol_m" in num.columns and "mcap_m" in num.columns:
        num["pm_mc_pct"] = 100.0 * num["pm_dol_m"] / num["mcap_m"]

    # Drop zeros that are invalid for some features
    for k in ["mcap_m","float_m","atr_usd","pm_dol_m","rvol","gap_pct","pm_mc_pct"]:
        if k in num.columns:
            num.loc[num[k] == 0, k] = np.nan

    def quantiles_for(k: str) -> Dict[str, float]:
        if k not in num.columns: return DEFAULT_IQR[k].copy()
        s = num[k].dropna()
        if len(s) < 10:  # need enough data
            return DEFAULT_IQR[k].copy()
        q1  = float(s.quantile(0.25))
        med = float(s.quantile(0.50))
        q3  = float(s.quantile(0.75))
        # choose loose fences from tails (5th/95th) to allow smooth tails
        lo  = float(s.quantile(0.05))
        hi  = float(s.quantile(0.95))
        if k == "gap_pct":
            # keep an absolute viability ceiling if dataset tail is wild
            hi = min(hi, 600.0)
        return dict(lo=lo, q1=q1, med=med, q3=q3, hi=hi)

    learned = {}
    for k in DEFAULT_IQR.keys():
        learned[k] = quantiles_for(k)

    return learned

# cache learned IQR per session
if "IQR" not in st.session_state:
    st.session_state.IQR = DEFAULT_IQR.copy()

if uploaded is not None:
    st.session_state.IQR = learn_iqr_from_excel(uploaded, learn_sheet)

IQR = st.session_state.IQR  # active windows

# =========================
# Smooth Gaussian scoring (never zero)
# =========================
EPS_FLOOR = 0.02  # minimum per-feature score (2%) so nothing is ever 0

def gauss_iqr_score(x: Optional[float], lo: float, q1: float, med: float, q3: float, hi: float) -> float:
    """
    Smooth 0..1 score that peaks at the median, decays with distance scaled by IQR.
    - Uses Gaussian: exp(-0.5 * ((x-med)/sigma)^2), where sigma = (q3-q1)/1.349 (≈IQR/1.349 ~ std for normal)
    - Outside [lo, hi], still returns a small value (floored by EPS_FLOOR) → never 0.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)): 
        return EPS_FLOOR
    x = float(x)
    iqr = max(1e-9, (q3 - q1))
    sigma = iqr / 1.349  # normal approx: IQR ≈ 1.349σ
    # in case iqr is tiny (degenerate), widen sigma a bit
    sigma = max(sigma, 1e-6)
    z = (x - med) / sigma
    s = math.exp(-0.5 * z * z)
    # softly discourage extreme beyond fences: multiply by taper on each side
    if x < lo:
        # linear distance to lo converted to sigmoid-ish damp
        d = (lo - x) / max(1e-9, abs(hi - lo))
        s *= 1.0 / (1.0 + 15.0 * d)
    elif x > hi:
        d = (x - hi) / max(1e-9, abs(hi - lo))
        s *= 1.0 / (1.0 + 15.0 * d)
    return max(EPS_FLOOR, min(1.0, s))

# Feature weights (sum ~1, renormalized)
W = {
    "float_m":   0.18,
    "mcap_m":    0.15,
    "atr_usd":   0.12,
    "gap_pct":   0.18,
    "pm_dol_m":  0.12,
    "pm_mc_pct": 0.07,   # NEW: PM $Vol / MC %
    "rvol":      0.16,
    "catalyst":  0.01,   # light
    "dilution":  0.01,   # light
}
Wsum = sum(W.values()) or 1.0
for k in W: W[k] = W[k]/Wsum

# =========================
# Checklist & Numeric% (Gaussian over data-driven IQR)
# =========================
def make_premarket_checklist(
    *, float_m: float, mcap_m: float, atr_usd: float,
    gap_pct: float, pm_vol_m: float, pm_dol_m: float,
    rvol: float, catalyst_points: float, dilution_points: float,
    pm_mc_pct: Optional[float], pm_pct_of_pred: Optional[float]
) -> Dict[str, Any]:
    good, warn, risk = [], [], []

    # Per-feature scores
    s_float = gauss_iqr_score(float_m,   **IQR["float_m"])
    s_mcap  = gauss_iqr_score(mcap_m,    **IQR["mcap_m"])
    s_atr   = gauss_iqr_score(atr_usd,   **IQR["atr_usd"])
    s_gap   = gauss_iqr_score(gap_pct,   **IQR["gap_pct"])
    s_pmdol   = gauss_iqr_score(pm_dol_m,  **IQR["pm_dol_m"])
    s_rvol  = gauss_iqr_score(rvol,      **IQR["rvol"])
    s_pmmc  = gauss_iqr_score(pm_mc_pct, **IQR["pm_mc_pct"]) if pm_mc_pct is not None else EPS_FLOOR

    # Catalyst/Dilution → map -1..+1 into 0..1 smoothly around IQRs learned for sliders
    # If you want them excluded from scoring, set weights to 0 above.
    s_cat = gauss_iqr_score(catalyst_points, **IQR.get("catalyst", dict(lo=-1, q1=-0.2, med=0.2, q3=0.6, hi=1)))
    s_dil = gauss_iqr_score(dilution_points, **IQR.get("dilution", dict(lo=-1, q1=-0.2, med=0.0, q3=0.2, hi=1)))

    # Text checklist (IQR-based: inside IQR → Good, near edges → Caution, far outside → Risk)
    def band_text(name, x, key, fmt="{:.2f}"):
        if x is None or (isinstance(x, float) and math.isnan(x)): 
            warn.append(f"{name}: missing")
            return
        p = IQR[key]
        lo, q1, med, q3, hi = p["lo"], p["q1"], p["med"], p["q3"], p["hi"]
        val = fmt.format(x) if isinstance(x, (int, float)) else str(x)
        if q1 <= x <= q3:
            good.append(f"{name} in IQR [{q1:.2f}–{q3:.2f}] (you {val})")
        elif lo <= x <= hi:
            warn.append(f"{name} near edge (IQR [{q1:.2f}–{q3:.2f}]; you {val})")
        else:
            risk.append(f"{name} outside fences (lo {lo:.2f} / hi {hi:.2f}; you {val})")

    band_text("Float (M)",            float_m,   "float_m",   "{:.2f}")
    band_text("Market Cap ($M)",      mcap_m,    "mcap_m",    "{:.0f}")
    band_text("ATR ($)",              atr_usd,   "atr_usd",   "{:.2f}")
    band_text("Gap (%)",              gap_pct,   "gap_pct",   "{:.1f}")
    band_text("Premarket $Vol ($M)",  pm_dol_m,  "pm_dol_m",  "{:.1f}")
    band_text("RVOL (×)",             rvol,      "rvol",      "{:.0f}")
    if pm_mc_pct is not None and not (isinstance(pm_mc_pct,float) and math.isnan(pm_mc_pct)):
        band_text("PM $Vol / MC (%)", pm_mc_pct, "pm_mc_pct", "{:.1f}")

    # Informational metric (requested): PM Vol / Predicted Vol %
    if pm_pct_of_pred is not None and pm_pct_of_pred == pm_pct_of_pred:
        warn.append(f"Info: PM Vol / Predicted Vol = {pm_pct_of_pred:.1f}%")

    # Weighted numeric % (0..100)
    soft = (
        W["float_m"]   * s_float +
        W["mcap_m"]    * s_mcap  +
        W["atr_usd"]   * s_atr   +
        W["gap_pct"]   * s_gap   +
        W["pm_dol_m"]  * s_pmdol   +
        W["pm_mc_pct"] * s_pmmc  +
        W["rvol"]      * s_rvol  +
        W["catalyst"]  * s_cat   +
        W["dilution"]  * s_dil
    )
    numeric_pct = float(max(0.0, min(100.0, 100.0 * soft)))

    # Verdict from the same signal
    if numeric_pct >= 75.0:
        verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
    elif numeric_pct >= 55.0:
        verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
    else:
        verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

    # Greens/Reds counters for UI
    greens = 0; reds = 0
    for key, x in [("float_m", float_m), ("mcap_m", mcap_m), ("atr_usd", atr_usd),
                   ("gap_pct", gap_pct), ("pm_dol_m", pm_dol_m), ("rvol", rvol), ("pm_mc_pct", pm_mc_pct)]:
        if x is None: continue
        p = IQR[key]
        if p["q1"] <= x <= p["q3"]:
            greens += 1
        elif x <= p["lo"] or x >= p["hi"]:
            reds   += 1

    return {
        "greens": greens, "reds": reds,
        "good": good, "warn": warn, "risk": risk,
        "verdict": verdict, "pill": pill,
        "phb_numeric": numeric_pct
    }

# =========================
# Session state for rows
# =========================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

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

        # PM metrics
        pm_mc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")
        pm_pct_of_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        # Qualitative %
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}", 1), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

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

            # diagnostics for display (but not tiles)
            "PM_MC_%": round(pm_mc_pct,1) if pm_mc_pct==pm_mc_pct else "",
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
        cB.metric("Numeric % (IQR-Gauss)", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qualitative %",    f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",      f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        d1, d2 = st.columns(2)
        d1.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d1.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        d2.metric("—", " ")  # spacer to keep layout simple

        # Checklist expander
        with st.expander("Premarket Checklist (data-driven IQR)", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            greens = l.get("PremarketGreens",0)
            reds = l.get("PremarketReds",0)
            pill = l.get("PremarketPill","")
            st.markdown(
                f"**Verdict:** {pill if pill else verdict} &nbsp;&nbsp;|&nbsp; ✅ {greens} &nbsp;·&nbsp; 🚫 {reds}",
                unsafe_allow_html=True
            )

            # Append the requested PM Vol / Predicted Vol % info prominently
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
                "Numeric_%": st.column_config.NumberColumn("Numeric_% (IQR-Gauss)", format="%.2f"),
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
