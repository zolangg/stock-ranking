import streamlit as st
import pandas as pd

st.set_page_config(page_title="Premarket Stock Ranking Tool", layout="wide")
st.header("Premarket Stock Ranking Tool")

# -------------------------------
# Criteria (7-point scales) - unchanged
# -------------------------------
CRITERIA = [
    {
        "name": "RVOL",
        "question": "Relative Volume (RVOL):",
        "options": [
            "3.0 – 4.0","4.0 – 5.0","5.0 – 7.0","7.0 – 10.0",
            "10.0 – 15.0","15.0 – 25.0","Over 25.0",
        ],
        "weight": 0.15,
    },
    {
        "name": "ATR",
        "question": "Average True Range (ATR) in USD:",
        "options": [
            "Under 0.05","0.05 – 0.10","0.10 – 0.20","0.20 – 0.35",
            "0.35 – 0.60","0.60 – 1.00","Over 1.00",
        ],
        "weight": 0.10,
    },
    {
        "name": "Float",
        "question": "Public Float (shares):",
        "options": [
            "Over 200M","100M – 200M","50M – 100M","25M – 50M",
            "10M – 25M","3M – 10M","Under 3M",
        ],
        "weight": 0.05,
    },
    {
        "name": "FloatPct",
        "question": "Premarket Volume as % of Float:",
        "options": [
            "Under 1%","1% – 3%","3% – 10%","10% – 25%",
            "25% – 50%","50% – 100%","Over 100%",
        ],
        "weight": 0.10,
    },
    {
        "name": "ShortInterest",
        "question": "Short Interest (% of float):",
        "options": [
            "Under 2%","2% – 5%","5% – 10%","10% – 15%",
            "15% – 20%","20% – 30%","Over 30%",
        ],
        "weight": 0.08,
    },
    {
        "name": "GapStruct",
        "question": "Gap & Trend Development:",
        "options": [
            "Gap fully reversed: price loses >80% of gap.",
            "Choppy reversal: price loses 50–80% of gap.",
            "Partial retracement: price loses 25–50% of gap.",
            "Sideways consolidation: gap holds, price moves within top 25% of gap.",
            "Uptrend with deep pullbacks (>30% retrace).",
            "Uptrend with moderate pullbacks (10–30% retrace).",
            "Clean uptrend, only minor pullbacks (<10%).",
        ],
        "weight": 0.15,
    },
    {
        "name": "LevelStruct",
        "question": "Key Price Levels:",
        "options": [
            "Fails at all major support/resistance; cannot hold any key level.",
            "Briefly holds or reclaims a level but loses it quickly; repeated failures.",
            "Holds one support but unable to break any resistance; capped below a key level.",
            "Breaks above resistance but cannot stay above; repeatedly dips below reclaimed level.",
            "Breaks and holds one major level; most resistance remains above.",
            "Breaks and holds several major levels; clears most overhead resistance.",
            "Breaks and holds above all resistance; blue sky, no levels overhead.",
        ],
        "weight": 0.15,
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "Sharp, accelerating downtrend; price makes new lows repeatedly.",
            "Persistent downtrend; decline continues but slows, still making lower lows.",
            "Downtrend loses momentum; price begins to flatten, lower lows shallow.",
            "Clear base forms; price consolidates sideways, volatility drops.",
            "Bottom confirmed; price sets a clear higher low after base.",
            "Uptrend begins; price breaks out of base and forms higher highs.",
            "Sustained uptrend; price makes consecutive higher highs, blue sky above.",
        ],
        "weight": 0.10,
    },
]

# -------------------------------
# Sidebar: weight controls
# -------------------------------
st.sidebar.header("Set Criteria Weights")
weights = {}
recalc = False
for crit in CRITERIA:
    new_weight = st.sidebar.slider(
        label=crit["question"],
        min_value=0.0,
        max_value=0.5,
        value=crit["weight"],
        step=0.01,
        key=f"weight_{crit['name']}"
    )
    if "weights" not in st.session_state:
        st.session_state["weights"] = {}
    if st.session_state["weights"].get(crit["name"], crit["weight"]) != new_weight:
        recalc = True
    weights[crit["name"]] = new_weight
    st.session_state["weights"][crit["name"]] = new_weight

news_weight = st.sidebar.slider(
    label="Catalyst (News) Weight (× on slider value)",
    min_value=0.0, max_value=2.0, value=1.0, step=0.05, key="news_weight"
)
if "prev_news_weight" not in st.session_state or st.session_state["prev_news_weight"] != news_weight:
    recalc = True
st.session_state["prev_news_weight"] = news_weight

dilution_weight = st.sidebar.slider(
    label="Dilution Impact Weight (× on slider value)",
    min_value=0.0, max_value=2.0, value=1.0, step=0.05, key="dilution_weight"
)
if "prev_dilution_weight" not in st.session_state or st.session_state["prev_dilution_weight"] != dilution_weight:
    recalc = True
st.session_state["prev_dilution_weight"] = dilution_weight

# NEW: PM Odds weights + thresholds
st.sidebar.header("PM Odds (Numeric) — Weight & Thresholds")
pm_weight = st.sidebar.slider(
    "PMOdds Weight (applied to 0–7 PMOdds subscore)",
    min_value=0.0, max_value=1.5, value=0.40, step=0.05, key="pm_weight"
)
if "prev_pm_weight" not in st.session_state or st.session_state["prev_pm_weight"] != pm_weight:
    recalc = True
st.session_state["prev_pm_weight"] = pm_weight

# Thresholds for scaling (feel free to tweak)
th_pct_target_hi = st.sidebar.slider("PM Vol % of Target: High marker", 0.05, 0.30, 0.15, 0.01)
th_dv_mc_hi      = st.sidebar.slider("$ DollarVol / MarketCap: High marker", 0.01, 0.20, 0.10, 0.01)
th_pct_float_hi  = st.sidebar.slider("PM Vol % of Float: High marker", 0.02, 0.50, 0.10, 0.01)

# -------------------------------
# Grading
# -------------------------------
def heat_level(score: float) -> str:
    if score >= 6.0:
        return "A++"
    elif score >= 5.5:
        return "A+"
    elif score >= 4.8:
        return "A"
    elif score >= 3.9:
        return "B"
    elif score >= 2.9:
        return "C"
    else:
        return "D"

# -------------------------------
# Session state
# -------------------------------
if "stock_scores" not in st.session_state:
    st.session_state.stock_scores = []

# -------------------------------
# Recalculate scores on weight changes
# -------------------------------
if recalc and st.session_state.stock_scores:
    for stock in st.session_state.stock_scores:
        base_score = sum(
            stock.get(crit['name'], 0) * weights.get(crit['name'], crit['weight'])
            for crit in CRITERIA
        )
        # Recompute PMOdds if inputs exist
        pm_odds = stock.get("PMOdds", 0.0)
        stock["Score"] = round(
            base_score
            + stock.get("Catalyst", 0.0) * news_weight
            + stock.get("Dilution", 0.0) * dilution_weight
            + pm_odds * pm_weight,
            2
        )

# -------------------------------
# Helpers for PMOdds scaling (0..7)
# -------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def scale_linear(x, hi, max_points):
    """Map x from 0..hi to 0..max_points (cap)."""
    if x is None:
        return 0.0
    return clamp((x / hi) * max_points, 0.0, max_points)

# -------------------------------
# Input form
# -------------------------------
with st.form(key="stock_form", clear_on_submit=True):
    ticker = st.text_input("Stock ticker (symbol)", max_chars=10).strip().upper()

    criteria_points = {}
    for crit in CRITERIA:
        idx = st.radio(
            crit["question"],
            options=list(enumerate(crit["options"], 1)),
            format_func=lambda x: x[1],
            key=crit["name"]
        )
        criteria_points[crit["name"]] = idx[0]  # 1..7

    # Catalyst & Dilution sliders (existing)
    catalyst_points = st.slider(
        "Catalyst Impact (−1.0 = Strong Negative … +1.0 = Strong Positive)",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05, key="catalyst_slider"
    )
    dilution_points = st.slider(
        "Dilution Impact (−1.0 = Heavy Overhang … +1.0 = Supportive)",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05, key="dilution_slider"
    )

    st.markdown("### PM Odds (numeric inputs)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        pm_vol_m = st.number_input("PM Volume (M shares)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
    with col2:
        pm_price = st.number_input("PM VWAP / Avg Price ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    with col3:
        mcap_m = st.number_input("Market Cap (M $)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    with col4:
        float_m = st.number_input("Float (M shares)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
    with col5:
        target_m = st.number_input("Target Daily Volume (M)", min_value=1.0, value=150.0, step=10.0, format="%.0f")

    submit = st.form_submit_button("Save Stock & Add to Ranking")

def compute_pm_odds(pm_vol_m, pm_price, mcap_m, float_m, target_m, catalyst_points):
    # Safeguards
    target_m = target_m if target_m and target_m > 0 else None
    mcap_m   = mcap_m   if mcap_m   and mcap_m   > 0 else None
    float_m  = float_m  if float_m  and float_m  > 0 else None
    pm_vol_m = pm_vol_m if pm_vol_m and pm_vol_m > 0 else 0.0
    pm_price = pm_price if pm_price and pm_price > 0 else 0.0

    # Metrics
    pct_target = (pm_vol_m / target_m) if target_m else None                # 0..1+
    dv_over_mc = ((pm_vol_m * 1e6 * pm_price) / (mcap_m * 1e6)) if mcap_m else None
    pct_float  = (pm_vol_m / float_m) if float_m else None

    # Points (cap each bucket)
    pts_target = scale_linear(pct_target, th_pct_target_hi, 3.0)  # 0..3
    pts_dvmc   = scale_linear(dv_over_mc, th_dv_mc_hi,     2.0)  # 0..2
    pts_pff    = scale_linear(pct_float,  th_pct_float_hi, 2.0)  # 0..2

    # Catalyst boost (only positive catalyst adds up to +1)
    cat_boost = clamp(max(0.0, catalyst_points), 0.0, 1.0)       # 0..1

    pm_odds = clamp(pts_target + pts_dvmc + pts_pff + cat_boost, 0.0, 7.0)
    return pm_odds, pct_target, dv_over_mc, pct_float

if submit and ticker:
    # Base criteria score
    base_score = sum(criteria_points[crit['name']] * weights[crit['name']] for crit in CRITERIA)

    # PMOdds calc
    pm_odds, pct_target, dv_over_mc, pct_float = compute_pm_odds(
        pm_vol_m, pm_price, mcap_m, float_m, target_m, catalyst_points
    )

    score_total = round(
        base_score
        + catalyst_points * news_weight
        + dilution_points * dilution_weight
        + pm_odds * pm_weight,
        2
    )

    stock_entry = {
        "Ticker": ticker,
        **criteria_points,            # each criterion 1..7
        "Catalyst": round(catalyst_points, 2),
        "Dilution": round(dilution_points, 2),
        # PM metrics & subscore
        "PM Vol (M)": round(pm_vol_m, 2),
        "PM Price": round(pm_price, 2),
        "MktCap (M)": round(mcap_m, 2),
        "Float (M)": round(float_m, 2),
        "Target (M)": round(target_m, 2),
        "PM%Target": round((pct_target or 0)*100, 2),
        "$Vol/MC %": round((dv_over_mc or 0)*100, 2),
        "PM%Float": round((pct_float or 0)*100, 2),
        "PMOdds": round(pm_odds, 2),
        "Score": score_total
    }
    st.session_state.stock_scores.append(stock_entry)
    st.success(f"Stock {ticker} saved!")

# -------------------------------
# Table & CSV
# -------------------------------
st.write("---")
st.header("Current Stock Ranking")

if st.session_state.stock_scores:
    df = pd.DataFrame(st.session_state.stock_scores)

    # Ensure numeric types present
    for c in ["Score","Catalyst","Dilution","PMOdds","PM Vol (M)","PM Price","MktCap (M)","Float (M)","Target (M)","PM%Target","$Vol/MC %","PM%Float"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Grading
    df["Level"] = df["Score"].apply(heat_level)

    ordered_cols = [
        "Ticker","Score","Level",
        "PMOdds","PM%Target","$Vol/MC %","PM%Float",
        "PM Vol (M)","PM Price","MktCap (M)","Float (M)","Target (M)",
        "RVOL","ATR","Float","FloatPct","ShortInterest","GapStruct","LevelStruct","Monthly",
        "Catalyst","Dilution"
    ]
    existing_cols = [c for c in ordered_cols if c in df.columns]
    df = df[existing_cols]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_order=existing_cols,
        column_config={
            "Ticker": st.column_config.Column(label="Ticker", pinned="left"),
            "Score":  st.column_config.Column(label="Score",  pinned="left"),
            "Level":  st.column_config.Column(label="Level",  pinned="left"),
        }
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranking as CSV",
        data=csv,
        file_name="stock_ranking.csv",
        mime="text/csv"
    )
else:
    st.info("No stocks have been ranked yet. Please fill out the form above!")
