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
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
      .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0;}
      .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa;}
      .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
      ul { margin-top: 4px; margin-bottom: 4px; padding-left: 18px; }
      li { margin-bottom: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Comma-friendly numeric input ----------
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

# ---------- Qualitative criteria (unchanged) ----------
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

# ---------- Sidebar: weights & uncertainty (kept for Qual %) ----------
st.sidebar.header("Numeric Weights (legacy display only)")
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

# Normalize qual block (numeric sliders kept but not used for Numeric %)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

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

# ---------- Day-volume model (millions out) ----------
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

# ---------- IQR scoring helpers ----------
def score_iqr(x, lo, q1, med, q3, hi):
    """
    0..1 score:
    - x <= lo or x >= hi -> 0
    - lo..q1: linear 0 -> 0.7
    - q1..q3: peak at med; 0.7 at edges; 1.0 at med (triangular)
    - q3..hi: linear 0.7 -> 0
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    x = float(x)
    if x <= lo or x >= hi:
        return 0.0
    if x < q1:
        return 0.7 * (x - lo) / max(1e-9, (q1 - lo))
    if x > q3:
        return 0.7 * (hi - x) / max(1e-9, (hi - q3))
    # inside IQR: triangular peak at median, 0.7 at q1/q3
    half = max(1e-9, (q3 - q1)/2.0)
    t = min(1.0, abs(x - med) / half)  # 0 at med, 1 at edges
    return 1.0 - 0.3 * t

def clamp01(v): return max(0.0, min(1.0, float(v)))

# ---------- IQR parameterization (tweak from your dataset) ----------
IQR = {
    # value units noted
    "float_m":   dict(lo=0.5,  q1=5.0,  med=8.5,  q3=12.0, hi=50.0),   # M shares
    "mcap_m":    dict(lo=5.0,  q1=30.0, med=80.0, q3=150.0, hi=500.0), # $M
    "atr_usd":   dict(lo=0.10, q1=0.20, med=0.30, q3=0.40, hi=1.00),   # $
    "gap_pct":   dict(lo=30.0, q1=70.0, med=120.0,q3=180.0,hi=280.0),  # %
    "pm_dol_m":  dict(lo=3.0,  q1=7.0,  med=11.0, q3=15.0, hi=40.0),   # $M
    "rvol":      dict(lo=50.0, q1=100.0,med=680.0,q3=1500.0,hi=3000.0),# ×
    # soft sliders:
    "catalyst":  dict(lo=-1.0,q1=-0.2,med=0.2, q3=0.6,  hi=1.0),       # slider -1..+1
    "dilution":  dict(lo=-1.0,q1=-0.2,med=0.0, q3=0.2,  hi=1.0),       # higher -> better
}

# feature weights (sum ≈ 1; renormalized)
W = {
    "float_m": 0.18,
    "mcap_m":  0.15,
    "atr_usd": 0.12,
    "gap_pct": 0.18,
    "pm_dol_m":0.15,
    "rvol":    0.17,
    "catalyst":0.03,
    "dilution":0.02,
}
W_sum = sum(W.values()) or 1.0
for k in W: W[k] = W[k] / W_sum

# hard caps for tail risks
HARD_CAPS = {
    "gap_abs_hi": 500.0,   # %; cap Numeric if above
    "rvol_blow":  6000.0,  # ×
    "pm$_double": 80.0,    # $M: ~2× high fence for PM$
    "cap_at_gap": 65.0,    # % max after extreme gap
    "cap_at_rvol":65.0,    # % max after blowout rvol
    "cap_at_pm$": 70.0,    # % max after very bloated PM$
}

def phb_checklist_iqr(float_m, mcap_m, atr_usd, gap_pct, pm_dol_m, rvol,
                      catalyst_points, dilution_points, pm_dollar_vs_mc) -> dict:
    good, warn, risk = [], [], []

    s_float = score_iqr(float_m, **IQR["float_m"])
    if float_m is None or float_m <= 0:
        warn.append("Float: missing")
    elif float_m < IQR["float_m"]["q1"] or float_m > IQR["float_m"]["q3"]:
        warn.append(f"Float outside IQR (you {float_m:.2f}M)")
    else:
        good.append(f"Float in IQR 5–12M (you {float_m:.2f}M)")

    s_mcap = score_iqr(mcap_m, **IQR["mcap_m"])
    if mcap_m is None or mcap_m <= 0:
        warn.append("MarketCap: missing")
    elif mcap_m < IQR["mcap_m"]["q1"] or mcap_m > IQR["mcap_m"]["q3"]:
        warn.append(f"MarketCap outside IQR (you {mcap_m:.0f}M)")
    else:
        good.append(f"MarketCap in IQR 30–150M (you {mcap_m:.0f}M)")

    s_atr = score_iqr(atr_usd, **IQR["atr_usd"])
    if atr_usd is None or atr_usd <= 0:
        warn.append("ATR: missing")
    elif 0.20 <= atr_usd <= 0.40:
        good.append(f"ATR in IQR 0.20–0.40 (you ${atr_usd:.2f})")
    elif atr_usd < 0.10:
        risk.append(f"ATR very thin (<$0.10; you ${atr_usd:.2f})")
    else:
        warn.append(f"ATR OK (you ${atr_usd:.2f})")

    s_gap = score_iqr(gap_pct, **IQR["gap_pct"])
    if gap_pct is None or gap_pct <= 0:
        warn.append("Gap %: missing")
    elif IQR["gap_pct"]["q1"] <= gap_pct <= IQR["gap_pct"]["q3"]:
        good.append(f"Gap in IQR 70–180% (you {gap_pct:.1f}%)")
    elif gap_pct < IQR["gap_pct"]["lo"]:
        risk.append(f"Gap weak (<{IQR['gap_pct']['lo']:.0f}% ; you {gap_pct:.1f}%)")
    elif gap_pct > IQR["gap_pct"]["hi"]:
        risk.append(f"Gap extreme (>{IQR['gap_pct']['hi']:.0f}% ; you {gap_pct:.1f}%)")
    else:
        warn.append(f"Gap viable (you {gap_pct:.1f}%)")

    s_pm = score_iqr(pm_dol_m, **IQR["pm_dol_m"])
    if pm_dol_m is None or pm_dol_m < 0:
        warn.append("PM $Vol: missing")
    elif IQR["pm_dol_m"]["q1"] <= pm_dol_m <= IQR["pm_dol_m"]["q3"]:
        good.append(f"PM $Vol in IQR $7–15M (you ${pm_dol_m:.1f}M)")
    elif pm_dol_m < IQR["pm_dol_m"]["lo"]:
        risk.append(f"PM $Vol thin (<${IQR['pm_dol_m']['lo']:.0f}M; you ${pm_dol_m:.1f}M)")
    elif pm_dol_m > IQR["pm_dol_m"]["hi"]:
        risk.append(f"PM $Vol bloated (>${IQR['pm_dol_m']['hi']:.0f}M; you ${pm_dol_m:.1f}M)")
    else:
        warn.append(f"PM $Vol viable (you ${pm_dol_m:.1f}M)")

    s_rvol = score_iqr(rvol, **IQR["rvol"])
    if rvol is None or rvol <= 0:
        warn.append("RVOL: missing")
    elif IQR["rvol"]["q1"] <= rvol <= IQR["rvol"]["q3"]:
        good.append(f"RVOL in IQR {int(IQR['rvol']['q1'])}–{int(IQR['rvol']['q3'])}× (you {int(rvol)}×)")
    elif rvol < IQR["rvol"]["lo"]:
        risk.append(f"RVOL too low (<{int(IQR['rvol']['lo'])}×; you {int(rvol)}×)")
    elif rvol > IQR["rvol"]["hi"]:
        risk.append(f"RVOL extreme (>{int(IQR['rvol']['hi'])}×; you {int(rvol)}×)")
    else:
        warn.append(f"RVOL viable (you {int(rvol)}×)")

    # Catalyst / Dilution as soft features scored by IQR too
    s_cat = score_iqr(catalyst_points, **IQR["catalyst"])
    s_dil = score_iqr(dilution_points, **IQR["dilution"])
    if catalyst_points >= 0.2: good.append("Catalyst: strong/real PR")
    elif catalyst_points <= -0.2: warn.append("Catalyst: weak/low-quality")
    else: warn.append("Catalyst: neutral/unknown")

    if dilution_points <= -0.2: risk.append("Dilution/ATM/overhang")
    elif dilution_points >= 0.2: good.append("Clean cap table / low dilution")
    else: warn.append("Dilution: neutral/unknown")

    # Optional neutral info
    if pm_dollar_vs_mc is not None and pm_dollar_vs_mc == pm_dollar_vs_mc:
        warn.append(f"Info: PM $Vol / MC = {pm_dollar_vs_mc:.1f}% (threshold TBD)")

    # Weighted Numeric % from IQR scores
    soft = (
        W["float_m"]  * s_float +
        W["mcap_m"]   * s_mcap  +
        W["atr_usd"]  * s_atr   +
        W["gap_pct"]  * s_gap   +
        W["pm_dol_m"] * s_pm    +
        W["rvol"]     * s_rvol  +
        W["catalyst"] * clamp01((s_cat)) +
        W["dilution"] * clamp01((s_dil))
    )
    numeric_pct = float(max(0.0, min(100.0, 100.0 * soft)))

    # Hard caps for extreme tails
    cap = 100.0
    if gap_pct and gap_pct >= HARD_CAPS["gap_abs_hi"]:
        cap = min(cap, HARD_CAPS["cap_at_gap"])
    if rvol and rvol >= HARD_CAPS["rvol_blow"]:
        cap = min(cap, HARD_CAPS["cap_at_rvol"])
    if pm_dol_m and pm_dol_m >= HARD_CAPS["pm$_double"]:
        cap = min(cap, HARD_CAPS["cap_at_pm$"])
    numeric_pct = min(numeric_pct, cap)

    # Verdict from numeric %
    if numeric_pct >= 75.0:
        verdict = "Strong Setup"; pill = '<span class="pill pill-good">Strong Setup</span>'
    elif numeric_pct >= 55.0:
        verdict = "Constructive"; pill = '<span class="pill pill-warn">Constructive</span>'
    else:
        verdict = "Weak / Avoid"; pill = '<span class="pill pill-bad">Weak / Avoid</span>'

    # Greens/Reds for counters (IQR-based)
    greens = 0
    reds = 0
    # inside IQR => green
    for name, val in [("float_m", float_m), ("mcap_m", mcap_m), ("atr_usd", atr_usd),
                      ("gap_pct", gap_pct), ("pm_dol_m", pm_dol_m), ("rvol", rvol)]:
        if val is None: continue
        p = IQR[name]
        if p["q1"] <= val <= p["q3"]:
            greens += 1
        elif val <= p["lo"] or val >= p["hi"]:
            reds += 1
    if catalyst_points >= IQR["catalyst"]["med"]: greens += 1
    if dilution_points <= IQR["dilution"]["q1"]:  reds += 1

    return {
        "greens": greens, "reds": reds,
        "good": good, "warn": warn, "risk": risk,
        "verdict": verdict, "pill": pill,
        "phb_numeric": numeric_pct
    }

# ---------- Tabs ----------
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
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)",   0.0, min_value=0.0, decimals=2)

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

    if submitted and ticker:
        # Predicted day volume (M) + CIs
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        # Qualitative %
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}", 1), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # PM $Vol / MC % (diagnostic)
        pm_dollar_vs_mc = 100.0 * pm_dol_m / mc_m if mc_m > 0 else float("nan")

        # IQR checklist + Numeric %
        ck = phb_checklist_iqr(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_dol_m=pm_dol_m, rvol=rvol,
            catalyst_points=catalyst_points, dilution_points=dilution_points,
            pm_dollar_vs_mc=pm_dollar_vs_mc
        )
        numeric_pct = ck["phb_numeric"]

        # Final Score = 50/50 Numeric (IQR) & Qualitative
        final_score = float(max(0.0, min(100.0, 0.5*numeric_pct + 0.5*qual_pct)))

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(numeric_pct, 2),  # IQR-based
            "Qual_%": round(qual_pct, 2),
            "FinalScore": round(final_score, 2),

            # Prediction fields
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            # Diagnostics
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1) if pm_dollar_vs_mc == pm_dollar_vs_mc else "",
            "PM_FloatRot_x": round(pm_vol_m / float_m, 3) if float_m > 0 else 0.0,

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
        cB.metric("Numeric % (IQR)", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qualitative %",    f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",      f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d1.caption("Premarket volume ÷ float.")
        pm_mc = l.get("PM$ / MC_%","")
        d2.metric("PM $Vol / MC", f"{pm_mc if pm_mc != '' else '—'}")
        d2.caption("Premarket dollar volume ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        d4.metric("Gap %", f"{l.get('_Gap_%',0):.1f}%")
        d4.caption("Open gap vs prior close.")

        with st.expander("Premarket Checklist (IQR-based)", expanded=True):
            verdict = l.get("PremarketVerdict","—")
            greens = l.get("PremarketGreens",0)
            reds = l.get("PremarketReds",0)
            pill = l.get("PremarketPill","")
            st.markdown(
                f"**Verdict:** {pill if pill else verdict} &nbsp;&nbsp;|&nbsp; ✅ {greens} &nbsp;·&nbsp; 🚫 {reds}",
                unsafe_allow_html=True
            )

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

# ---------- Ranking tab ----------
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
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Level"),
                "Numeric_%": st.column_config.NumberColumn("Numeric_% (IQR)", format="%.2f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Premarket Verdict"),
                "PremarketReds": st.column_config.NumberColumn("🚫 Reds"),
                "PremarketGreens": st.column_config.NumberColumn("✅ Greens"),
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
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
