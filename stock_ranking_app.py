import streamlit as st
import pandas as pd
import math
from typing import Optional

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ---------- Global CSS ----------
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
      .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
      .checklist pre { white-space: pre-wrap; font-size: 0.95rem; line-height: 1.25rem;}
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
      .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0;}
      .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa;}
      .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Comma-friendly numeric input ----------
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    # Remove spaces and apostrophes (1 234,56 or 1'234,56)
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    # If there's a comma but no dot -> treat comma as decimal
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        # Otherwise drop commas as thousands separators
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    """Text input that accepts 5,05  / 1'234,5 / 1 234,5 / 5.05 and returns float."""
    fmt = f"{{:.{decimals}f}}"
    default_str = fmt.format(float(value))
    s = st.text_input(label, default_str, key=key, help=help)
    v = _parse_local_float(s)
    if v is None:
        st.caption('<span class="hint">Enter a number, e.g. 5,05</span>', unsafe_allow_html=True)
        return float(value)
    # Clamp to min/max
    if v < min_value:
        st.caption(f'<span class="hint">Clamped to minimum: {fmt.format(min_value)}</span>', unsafe_allow_html=True)
        v = min_value
    if max_value is not None and v > max_value:
        st.caption(f'<span class="hint">Clamped to maximum: {fmt.format(max_value)}</span>', unsafe_allow_html=True)
        v = max_value
    # Show normalized preview if user typed a comma/formatting
    if ("," in s) or (" " in s) or ("'" in s):
        st.caption(f'<span class="hint">= {fmt.format(v)}</span>', unsafe_allow_html=True)
    return float(v)

# ---------- Markdown table helper ----------
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

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- Qualitative criteria ----------
QUAL_CRITERIA = [
    {
        "name": "GapStruct",
        "question": "Gap & Trend Development:",
        "options": [
            "Gap fully reversed: price loses >80% of gap.",
            "Choppy reversal: price loses 50–80% of gap.",
            "Partial retracement: price loses 25–50% of gap.",
            "Sideways consolidation: gap holds, price within top 25% of gap.",
            "Uptrend with deep pullbacks (>30% retrace).",
            "Uptrend with moderate pullbacks (10–30% retrace).",
            "Clean uptrend, only minor pullbacks (<10%).",
        ],
        "weight": 0.15,
        "help": "How well the gap holds and trends.",
    },
    {
        "name": "LevelStruct",
        "question": "Key Price Levels:",
        "options": [
            "Fails at all major support/resistance; cannot hold any key level.",
            "Briefly holds/reclaims a level but loses it quickly; repeated failures.",
            "Holds one support but unable to break resistance; capped below a key level.",
            "Breaks above resistance but cannot stay; dips below reclaimed level.",
            "Breaks and holds one major level; most resistance remains above.",
            "Breaks and holds several major levels; clears most overhead resistance.",
            "Breaks and holds above all resistance; blue sky.",
        ],
        "weight": 0.15,
        "help": "Break/hold behavior at key levels.",
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "Sharp, accelerating downtrend; new lows repeatedly.",
            "Persistent downtrend; still lower lows.",
            "Downtrend losing momentum; flattening.",
            "Clear base; sideways consolidation.",
            "Bottom confirmed; higher low after base.",
            "Uptrend begins; breaks out of base.",
            "Sustained uptrend; higher highs, blue sky.",
        ],
        "weight": 0.10,
        "help": "Higher-timeframe bias.",
    },
]

# ---------- Sidebar: weights & uncertainty (UNCHANGED) ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(crit["name"], 0.0, 1.0, crit["weight"], 0.01)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider(
    "Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
    help="Estimated std dev of residuals in ln(volume). 0.60 ≈ typical for your sheet."
)

# Normalize blocks separately (UNCHANGED)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------- Numeric bucket scorers (UNCHANGED) ----------
def pts_rvol(x: float) -> int:
    for th, p in [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)]:
        if x < th: return p
    return 7

def pts_atr(x: float) -> int:
    for th, p in [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)]:
        if x < th: return p
    return 7

def pts_si(x: float) -> int:
    for th, p in [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)]:
        if x < th: return p
    return 7

def pts_fr(pm_vol_m: float, float_m: float) -> int:
    if float_m <= 0: return 1
    rot = pm_vol_m / float_m
    for th, p in [(0.01,1),(0.03,2),(0.10,3),(0.25,4),(0.50,5),(1.00,6)]:
        if rot < th: return p
    return 7

def pts_float(float_m: float) -> int:
    if float_m <= 3: return 7
    for th, p in [(200,2),(100,3),(50,4),(35,5),(10,6)]:
        if float_m > th: return p
    return 7

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

# ---------- Day-volume model (millions out) (UNCHANGED) ----------
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

# ---------- FT model params (placeholders) (UNCHANGED) ----------
_FT_INTERCEPT = -0.20
_FT_COEF = {
    'ln_gapf':    1.20,
    'ln_pmvol_f': 0.80,
    'ln_fr':      0.30,
    'ln_pmvol_m': 0.10,
    'ln_mcap':   -0.40,
    'ln_atr':    -0.30,
    'ln_float':  -0.20,
    'catalyst':   0.40,
}
_FT_MEAN  = {k: 0.0 for k in _FT_COEF.keys()}
_FT_SCALE = {k: 1.0 for k in _FT_COEF.keys()}

def _std(x, m, s):
    s = float(s) if s not in (None, 0.0) else 1.0
    return (x - float(m)) / s

def predict_ft_prob_premarket(float_m: float, mcap_m: float, atr_usd: float,
                              gap_pct: float, pm_vol_m: float,
                              pred_vol_m: float, catalyst_flag: int = 0) -> float:
    e = 1e-6
    ln_float   = math.log(max(float_m, e))
    ln_mcap    = math.log(max(mcap_m, e))
    ln_atr     = math.log(max(atr_usd, e))
    ln_gapf    = math.log(max(gap_pct, 0.0)/100.0 + e)
    ln_pmvol_m = math.log(max(pm_vol_m, 0.0) + 1.0)
    fr         = (pm_vol_m / max(float_m, e)) if float_m > 0 else 0.0
    ln_fr      = math.log(fr + 1.0)
    denom      = max(float(pred_vol_m or 0.0), 0.0)
    pm_frac    = (pm_vol_m / denom) if denom > 0 else 0.0
    pm_frac    = max(0.0, min(pm_frac, 5.0))
    ln_pmvol_f = math.log(pm_frac + 1.0)

    lp = _FT_INTERCEPT
    features = {
        'ln_float': ln_float, 'ln_mcap': ln_mcap, 'ln_atr': ln_atr, 'ln_gapf': ln_gapf,
        'ln_pmvol_f': ln_pmvol_f, 'ln_pmvol_m': ln_pmvol_m, 'ln_fr': ln_fr,
        'catalyst': float(catalyst_flag),
    }
    for name, val in features.items():
        if name in _FT_COEF:
            lp += _FT_COEF[name] * _std(val, _FT_MEAN.get(name, 0.0), _FT_SCALE.get(name, 1.0))

    if lp >= 0:
        p = 1.0 / (1.0 + math.exp(-lp))
    else:
        elp = math.exp(lp)
        p = elp / (1.0 + elp)
    return max(0.0, min(1.0, p))

# ---------- PHB Premarket Checklist (NEW, uses numeric inputs only) ----------
PHB_RULES = {
    "FLOAT_MAX": 20.0,           # sweet 5–12M
    "FLOAT_SWEET_LO": 5.0,
    "FLOAT_SWEET_HI": 12.0,
    "MCAP_MAX": 150.0,
    "MCAP_HUGE": 500.0,
    "ATR_MIN": 0.15,             # runners often 0.2–0.4
    "GAP_MIN": 70.0,             # sweet ~100
    "GAP_SWEET_HI": 180.0,
    "GAP_VIABLE_HI": 280.0,      # tails thin above here
    "GAP_OUTLIER": 300.0,        # unproven tail in your sample
    "PM$_MIN": 7.0,
    "PM$_MAX": 30.0,             # >30–40M often frontloads
    "PM_SHARE_MIN": 10.0,        # % of predicted day
    "PM_SHARE_MAX": 20.0,
    "PM_SHARE_FAIL_LOW": 5.0,
    "PM_SHARE_FAIL_HI": 35.0,
    "RVOL_MIN": 100.0,
    "RVOL_MAX": 1500.0,
    "RVOL_WARN_MAX": 3000.0,
}

def make_premarket_checklist(*, float_m: float, mcap_m: float, atr_usd: float,
                             gap_pct: float, pm_vol_m: float, pm_dol_m: float,
                             rvol: float, pm_pct_of_pred: float,
                             catalyst_points: float, dilution_points: float) -> dict:
    R = PHB_RULES
    lines = []
    reds = 0
    greens = 0

    # Catalyst (slider)
    if catalyst_points >= 0.2:
        lines.append("✅ Catalyst: strong / real PR.")
        greens += 1
    elif catalyst_points <= -0.2:
        lines.append("🚫 Catalyst: weak / low-quality.")
        reds += 1
    else:
        lines.append("⚠️ Catalyst: neutral / unknown.")

    # Dilution (slider)
    if dilution_points >= 0.2:
        lines.append("✅ Dilution risk low / clean cap table.")
        greens += 1
    elif dilution_points <= -0.2:
        lines.append("🚫 Dilution / ATM / overhang risk.")
        reds += 1
    else:
        lines.append("⚠️ Dilution: neutral / unknown.")

    # Float
    if float_m <= 0:
        lines.append("⚠️ Float missing.")
    elif R["FLOAT_SWEET_LO"] <= float_m <= R["FLOAT_SWEET_HI"]:
        lines.append(f"✅ Float sweet spot {R['FLOAT_SWEET_LO']:.0f}–{R['FLOAT_SWEET_HI']:.0f}M (you {float_m:.2f}M)."); greens += 1
    elif float_m < R["FLOAT_MAX"]:
        lines.append(f"✅ Float <{R['FLOAT_MAX']:.0f}M (you {float_m:.2f}M)."); greens += 1
    elif float_m >= 50:
        lines.append(f"🚫 Float very large (≥50M; you {float_m:.2f}M)."); reds += 1
    else:
        lines.append(f"⚠️ Float elevated (≥{R['FLOAT_MAX']:.0f}M; you {float_m:.2f}M).")

    # Market cap
    if mcap_m <= 0:
        lines.append("⚠️ Market cap missing.")
    elif mcap_m < R["MCAP_MAX"]:
        lines.append(f"✅ MarketCap <{R['MCAP_MAX']:.0f}M (you {mcap_m:.1f}M)."); greens += 1
    elif mcap_m >= R["MCAP_HUGE"]:
        lines.append(f"🚫 MarketCap huge (≥{R['MCAP_HUGE']:.0f}M; you {mcap_m:.1f}M)."); reds += 1
    else:
        lines.append(f"⚠️ MarketCap elevated (≥{R['MCAP_MAX']:.0f}M; you {mcap_m:.1f}M).")

    # ATR
    if atr_usd >= R["ATR_MIN"]:
        if 0.2 <= atr_usd <= 0.4:
            lines.append(f"✅ ATR runner band 0.2–0.4 (you {atr_usd:.2f})."); greens += 1
        else:
            lines.append(f"✅ ATR ≥{R['ATR_MIN']:.2f} (you {atr_usd:.2f})."); greens += 1
    else:
        lines.append(f"⚠️ ATR thin (<{R['ATR_MIN']:.2f}; you {atr_usd:.2f}).")

    # Gap %
    if gap_pct < R["GAP_MIN"]:
        lines.append(f"🚫 Gap small (<{R['GAP_MIN']:.0f}%; you {gap_pct:.1f}%)."); reds += 1
    elif gap_pct <= R["GAP_SWEET_HI"]:
        lines.append(f"✅ Gap sweet {R['GAP_MIN']:.0f}–{R['GAP_SWEET_HI']:.0f}% (you {gap_pct:.1f}%)."); greens += 1
    elif gap_pct <= R["GAP_VIABLE_HI"]:
        lines.append(f"✅ Gap viable ≤{R['GAP_VIABLE_HI']:.0f}% (you {gap_pct:.1f}%)."); greens += 1
    else:
        lines.append(f"⚠️ Gap outlier >{R['GAP_VIABLE_HI']:.0f}% (you {gap_pct:.1f}%).")
        if gap_pct > R["GAP_OUTLIER"]:
            lines.append(f"⚠️ >{R['GAP_OUTLIER']:.0f}% is unproven tail in sample (exhaustion risk).")

    # Premarket $Volume
    if pm_dol_m < 3:
        lines.append(f"🚫 PM $Vol very thin (<$3M; you ${pm_dol_m:.1f}M)."); reds += 1
    elif R["PM$_MIN"] <= pm_dol_m <= R["PM$_MAX"]:
        lines.append(f"✅ PM $Vol sweet ${R['PM$_MIN']:.0f}–{R['PM$_MAX']:.0f}M (you ${pm_dol_m:.1f}M)."); greens += 1
    elif 5 <= pm_dol_m <= 22:
        lines.append(f"✅ PM $Vol viable ~$5–22M (you ${pm_dol_m:.1f}M)."); greens += 1
    elif pm_dol_m > 40:
        lines.append(f"🚫 PM $Vol bloated (>{40}M; you ${pm_dol_m:.1f}M)."); reds += 1
    else:
        lines.append(f"⚠️ PM $Vol marginal (you ${pm_dol_m:.1f}M).")

    # PM shares as % of predicted day
    if pm_pct_of_pred <= 0:
        lines.append("⚠️ PM % of Predicted: cannot compute.")
    elif R["PM_SHARE_MIN"] <= pm_pct_of_pred <= R["PM_SHARE_MAX"]:
        lines.append(f"✅ PM shares sweet {R['PM_SHARE_MIN']:.0f}–{R['PM_SHARE_MAX']:.0f}% (you {pm_pct_of_pred:.1f}%)."); greens += 1
    elif 7 <= pm_pct_of_pred <= 25:
        lines.append(f"✅ PM shares viable 7–25% (you {pm_pct_of_pred:.1f}%)."); greens += 1
    elif pm_pct_of_pred < R["PM_SHARE_FAIL_LOW"]:
        lines.append(f"🚫 PM shares too thin (<{R['PM_SHARE_FAIL_LOW']:.0f}%; you {pm_pct_of_pred:.1f}%)."); reds += 1
    elif pm_pct_of_pred > R["PM_SHARE_FAIL_HI"]:
        lines.append(f"🚫 PM shares front-loaded (>{R['PM_SHARE_FAIL_HI']:.0f}%; you {pm_pct_of_pred:.1f}%)."); reds += 1
    else:
        lines.append(f"⚠️ PM shares outside sweet band (you {pm_pct_of_pred:.1f}%).")

    # RVOL
    if rvol <= 0:
        lines.append("⚠️ RVOL missing/zero.")
    elif R["RVOL_MIN"] <= rvol <= R["RVOL_MAX"]:
        lines.append(f"✅ RVOL sweet {int(R['RVOL_MIN'])}–{int(R['RVOL_MAX'])}× (you {rvol:.0f}×)."); greens += 1
    elif 70 <= rvol <= 2000:
        lines.append(f"✅ RVOL viable ~70–2000× (you {rvol:.0f}×)."); greens += 1
    elif rvol < 50:
        lines.append(f"🚫 RVOL very low (<50×; you {rvol:.0f}×)."); reds += 1
    else:
        lines.append(f"⚠️ RVOL outside sweet band (you {rvol:.0f}×).")
    if rvol > R["RVOL_WARN_MAX"]:
        lines.append(f"⚠️ RVOL >{int(R['RVOL_WARN_MAX'])}× — blowout/exhaustion risk.")

    # Verdict
    if reds >= 2:
        verdict = "Weak / Avoid"
        pill = '<span class="pill pill-bad">Weak / Avoid</span>'
    elif greens >= 6:
        verdict = "Strong Setup"
        pill = '<span class="pill pill-good">Strong Setup</span>'
    else:
        verdict = "Constructive"
        pill = '<span class="pill pill-warn">Constructive</span>'

    # One-line summary for tables
    summary = f"{verdict}; Float {float_m:.1f}M, MC {mcap_m:.0f}M, Gap {gap_pct:.0f}%, PM$ {pm_dol_m:.1f}M, PM% {pm_pct_of_pred:.0f}%, RVOL {rvol:.0f}×"

    return {
        "lines": lines,
        "verdict": verdict,
        "verdict_pill": pill,
        "greens": greens,
        "reds": reds,
        "summary": summary
    }

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Numeric & Modifiers</div>', unsafe_allow_html=True)

    # Form that clears on submit
    with st.form("add_form", clear_on_submit=True):
        # === Requested order ===
        col1, col2, col3 = st.columns([1.2, 1.2, 0.9])

        # First column: Ticker, Market Cap, Float, SI %, Gap %
        with col1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)

        # Second column: ATR, RVOL, Premarket Volume (shares), Premarket $ Volume
        with col2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)",   0.0, min_value=0.0, decimals=2)

        # Third column: Modifiers right next to numbers
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
                    help=crit.get("help", None)
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    # After submit
    if submitted and ticker:
        # === Day volume prediction (M) (UNCHANGED) ===
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)

        # Confidence bands (millions)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)    # ~68%
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)   # ~95%

        # === Numeric points (UNCHANGED) ===
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # === Qualitative points (UNCHANGED) ===
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}"), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # === Combine + modifiers (UNCHANGED) ===
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + (catalyst_points*10) + (dilution_points*10), 2)
        final_score = max(0.0, min(100.0, final_score))

        # === Diagnostics to save (UNCHANGED) ===
        pm_pct_of_pred   = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x   = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc  = 100.0 * pm_dol_m / mc_m if mc_m > 0 else 0.0  # uses input $Volume (M$)

        # === FT Probability (UNCHANGED) ===
        ft_prob = predict_ft_prob_premarket(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_vol_m=pm_vol_m,
            pred_vol_m=pred_vol_m,
            catalyst_flag=1 if abs(catalyst_points) > 1e-9 else 0
        )
        ft_pct = round(100.0 * ft_prob, 1)
        ft_label = ("High FT" if ft_pct >= 70 else
                    "Moderate FT" if ft_pct >= 55 else
                    "Low FT" if ft_pct >= 40 else
                    "Very Low FT")
        ft_display = f"{ft_pct:.1f}% ({ft_label})"

        # === NEW: Premarket Checklist (data-driven, read-only) ===
        checklist = make_premarket_checklist(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_vol_m=pm_vol_m, pm_dol_m=pm_dol_m,
            rvol=rvol, pm_pct_of_pred=pm_pct_of_pred,
            catalyst_points=catalyst_points, dilution_points=dilution_points
        )

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "OddsScore": final_score,
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            # Prediction fields
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),

            # Ratios
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),

            # FT fields
            "FT": ft_display,
            "FT_Prob_%": ft_pct,
            "FT_Label": ft_label,

            # raw inputs for debug / export
            "_MCap_M": mc_m,
            "_Gap_%": gap_pct,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,
            "_Float_M": float_m,
            "_Catalyst": float(catalyst_points),
            "_Dilution": float(dilution_points),

            # NEW: checklist & summary
            "PremarketVerdict": checklist["verdict"],
            "PremarketSummary": checklist["summary"],
            "PremarketChecklist": "\n".join(checklist["lines"]),
            "PremarketReds": checklist["reds"],
            "PremarketGreens": checklist["greens"],
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
        cB.metric("Numeric Block", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qual Block",    f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",   f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d1.caption("Premarket volume ÷ float.")
        d2.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("Premarket dollar volume ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        d4.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d4.caption("PM volume ÷ predicted day volume × 100.")
        d5.metric("FT Probability", f"{l.get('FT_Prob_%',0):.1f}%")
        d5.caption(f"FT Label: {l.get('FT_Label','—')}")

        # NEW: Premarket Checklist UI
        with st.expander("Premarket Checklist (data-driven)", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            greens = l.get("PremarketGreens",0)
            reds = l.get("PremarketReds",0)
            st.markdown(f"**Verdict:** {verdict} · ✅ {greens} · 🚫 {reds}")
            st.markdown("<div class='checklist'><pre>" + l.get("PremarketChecklist","(no checks)") + "</pre></div>", unsafe_allow_html=True)
            st.caption(f"Summary: {l.get('PremarketSummary','—')}")

# ---------- Ranking tab ----------
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        # NEW: include verdict & concise summary in both views
        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PremarketVerdict","PremarketReds","PremarketGreens",
            "PM$ / MC_%","PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM_%_of_Pred",
            "FT","PremarketSummary"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","FT","PremarketVerdict","PremarketSummary") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Level"),
                "Numeric_%": st.column_config.NumberColumn("Numeric_%", format="%.2f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Premarket Verdict"),
                "PremarketReds": st.column_config.NumberColumn("🚫 Reds"),
                "PremarketGreens": st.column_config.NumberColumn("✅ Greens"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "FT": st.column_config.TextColumn("FT (p/label)"),
                "PremarketSummary": st.column_config.TextColumn("Summary"),
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
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
