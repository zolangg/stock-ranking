import streamlit as st
import pandas as pd

st.header("Premarket Stock Ranking Tool")

CATALYSTS = [
    # ... (dein Catalyst-Block wie gehabt, unverändert) ...
    # Hier für Kürze ausgelassen, da von dir vollständig.
]

CRITERIA = [
    {
        "name": "RVOL",
        "question": "Relative Volume (RVOL):",
        "options": [
            "3.0 - 4.0",
            "4.0 – 5.0",
            "5.0 - 7.0",
            "7.0 - 10.0",
            "10.0 – 15.0",
            "15.0 – 25.0",
            "Over 25.0",
        ],
        "weight": 0.15,
    },
    {
        "name": "ATR",
        "question": "Average True Range (ATR) in USD:",
        "options": [
            "Under 0.05",
            "0.05 – 0.10",
            "0.10 – 0.20",
            "0.20 - 0.35",
            "0.35 - 0.60",
            "0.60 – 1.00",
            "Over 1.00",
        ],
        "weight": 0.1,
    },
    {
        "name": "Float",
        "question": "Public Float (shares):",
        "options": [
            "Over 200M",
            "100M - 200M",
            "50M – 100M",
            "25M – 50M",
            "10M – 25M",
            "3M - 10M",
            "Under 3M",
        ],
        "weight": 0.05,
    },
    {
        "name": "FloatPct",
        "question": "Premarket Volume as % of Float:",
        "options": [
            "Under 1%",
            "1% - 3%",
            "3% - 10%",
            "10% – 25%",
            "25% – 50%",
            "50% – 100%",
            "Over 100%",
        ],
        "weight": 0.1,
    },
    {
        "name": "GapStruct",
        "question": "Gap & Trend Development:",
        "options": [
            "Immediate full reversal after gap; price gives back all gains in one move down.",
            "Choppy reversal after gap; price trends down and gives back most of the gains.",
            "Sideways range after gap; no clear trend up or down.",
            "Uptrend with deep, frequent pullbacks.",
            "Uptrend with moderate pullbacks.",
            "Steady uptrend; only minor pullbacks.",
            "Exceptionally clean uptrend.",
        ],
        "weight": 0.15,
    },
    {
        "name": "LevelStruct",
        "question": "Price Levels:",
        "options": [
            "Fails at every key level; unable to hold or reclaim any support.",
            "Briefly reclaims or holds a key level but quickly loses it again.",
            "Holds some levels but unable to break through any major resistance.",
            "Breaks key level but cannot hold above; price keeps dipping below the level.",
            "Breaks and holds above a key level; most resistance levels remain above.",
            "Breaks and holds above multiple key levels; most resistance levels are cleared.",
            "Breaks and holds above all previous resistance; blue sky with no levels overhead.",
        ],
        "weight": 0.15,
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "Sharp, accelerating downtrend; new lows form repeatedly, no support visible.",
            "Persistent but steady downtrend; price continues lower but the decline slows slightly.",
            "Downtrend loses momentum; price begins to flatten out, lower lows are shallow.",
            "Base formation: price consolidates sideways after downtrend, volatility reduces.",
            "Bottom confirmed: price sets a clear higher low, structure shifts neutral.",
            "Uptrend initiates: breakout from base, first higher highs established.",
            "Sustained uptrend: price makes consecutive higher highs, all resistance is cleared.",
        ],
        "weight": 0.10,
    },
    {
        "name": "DilutionRisk",
        "question": "Dilution & Overhead Supply Risk (SEC Filings):",
        "options": [
            "High risk: Recent S-1/F-1 or EFFECT for offering, active ATM or pending S-1/F-1, large shelf not yet used, many in-the-money warrants, urgent cash need. Massive overhang possible.",
            "ATM or Shelf registered and active, several warrant/convertible filings near current price, cash need in the next 3–6 months.",
            "ATM/Shelf active but only partially used, some warrants/options in or near the money, moderate cash need in 6–9 months.",
            "Shelf/ATM mostly exhausted or expiring, few new warrants/options above price, cash buffer for 9–12 months.",
            "Minimal overhang: Only a few OTM warrants/options, no active ATM, shelf nearly expired, company is cash-positive.",
            "No active ATM, no shelf, no EFFECT, very few warrants/options, no recent dilution filings, cash position is strong.",
            "No shelf, ATM, warrants, or converts; no dilution-related filings, strong cash position. No overhang or dilution risk.",
        ],
        "weight": 0.08,
    },
    {
        "name": "Spread",
        "question": "Bid-Ask Spread:",
        "options": [
            "Over 5%",
            "4%–5%",
            "3%–4%",
            "2%–3%",
            "1%–2%",
            "0.5%–1%",
            "Under 0.5%",
        ],
        "weight": 0.05,
    },
]

# --- Sidebar for weights ---
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
    label="Catalyst (News) Weight",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05,
    key="news_weight"
)
if "prev_news_weight" not in st.session_state or st.session_state["prev_news_weight"] != news_weight:
    recalc = True
st.session_state["prev_news_weight"] = news_weight

def heat_level(score):
    if score >= 6.0:
        return "A++"
    elif score >= 5.5:
        return "A+"
    elif score >= 4.8:
        return "A"
    elif score >= 4.0:
        return "B"
    elif score >= 3.5:
        return "C"
    else:
        return "D"

if "stock_scores" not in st.session_state:
    st.session_state.stock_scores = []

# --- RECALC on weight/news_weight change ---
if recalc and st.session_state.stock_scores:
    for stock in st.session_state.stock_scores:
        base_score = sum(
            stock[crit['name']] * weights[crit['name']] for crit in CRITERIA
        )
        stock["Score"] = round(base_score + stock["Catalyst"] * news_weight, 2)

# --- Stock input form ---
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
        criteria_points[crit["name"]] = idx[0]

    selected_catalysts = st.multiselect(
        "Select all relevant catalysts/news/technical/price triggers (multiple allowed):",
        options=[cat["name"] for cat in CATALYSTS]
    )
    catalyst_score = sum(cat["score"] for cat in CATALYSTS if cat["name"] in selected_catalysts)
    catalyst_points = round(catalyst_score, 2)
    submit = st.form_submit_button("Save Stock & Add to Ranking")

if submit and ticker:
    base_score = sum(
        criteria_points[crit['name']] * weights[crit['name']] for crit in CRITERIA
    )
    score_normalized = round(base_score + catalyst_points * news_weight, 2)
    stock_entry = {
        "Ticker": ticker,
        **criteria_points,
        "Catalyst": catalyst_points,
        "Score": score_normalized
    }
    st.session_state.stock_scores.append(stock_entry)
    st.success(f"Stock {ticker} saved!")

st.write("---")
st.header("Current Stock Ranking")

if st.session_state.stock_scores:
    df = pd.DataFrame(st.session_state.stock_scores)
    df["Score"] = df["Score"].astype(float).round(2)
    if "Catalyst" in df.columns:
        df["Catalyst"] = df["Catalyst"].astype(float).round(2)
    df["Level"] = df["Score"].apply(heat_level)

    ordered_cols = [
        "Ticker", "Score", "Level", "RVOL", "ATR", "Float", "FloatPct", "GapStruct",
        "LevelStruct", "Monthly", "DilutionRisk", "Spread", "Catalyst"
    ]
    
    st.dataframe(
        df[ordered_cols],
        use_container_width=True,
        hide_index=True,
        column_order=ordered_cols,
        column_config={
            "Ticker": st.column_config.Column(
                label="Ticker",
                pinned="left"
            ),
            "Score": st.column_config.Column(
                label="Score",
                pinned="left"
            ),
            "Level": st.column_config.Column(
                label="Level",
                pinned="left"
            ),
        }
    )

    csv = df[ordered_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranking as CSV",
        data=csv,
        file_name="stock_ranking.csv",
        mime="text/csv"
    )
else:
    st.info("No stocks have been ranked yet. Please fill out the form above!")
