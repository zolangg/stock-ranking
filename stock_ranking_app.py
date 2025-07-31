import streamlit as st
import pandas as pd

st.header("Premarket Stock Ranking Tool")

CATALYSTS = [
    # --- POSITIVE CATALYSTS ---
    {"name": "Unusually good/bad earnings report (surprise!)", "score": 1.0},
    {"name": "Better/worse than expected guidance reported", "score": 0.9},
    {"name": "Market can't put a ceiling on earnings (growth stocks)", "score": 0.9},
    {"name": "Acquisition announced or confirmed", "score": 1.0},
    {"name": "Acquisition rumors or media speculation", "score": 0.7},
    {"name": "New or delayed product announcement", "score": 0.8},
    {"name": "Positive/negative results of ongoing study announced", "score": 0.8},
    {"name": "Positive/negative results of independent study announced", "score": 0.9},
    {"name": "Positive/negative results of completed study announced", "score": 0.9},
    {"name": "Positive Phase 1 trial results announced", "score": 0.7},
    {"name": "Positive Phase 2 trial results announced", "score": 0.8},
    {"name": "Positive Phase 3 trial results announced", "score": 0.9},
    {"name": "Positive Phase 4 (post-market) results announced", "score": 0.7},
    {"name": "New drug or treatment announced / New clinical data released", "score": 0.9},
    {"name": "Emergency Use Authorization (EUA) granted", "score": 1.0},
    {"name": "FDA approval granted", "score": 1.0},
    {"name": "FDA meeting scheduled or results announced", "score": 0.9},
    {"name": "FDA Fast Track designation granted", "score": 0.8},
    {"name": "FDA Breakthrough Therapy designation granted", "score": 0.9},
    {"name": "FDA Priority Review designation granted", "score": 0.8},
    {"name": "FDA Orphan Drug designation granted", "score": 0.7},
    {"name": "FDA Accelerated Approval pathway granted", "score": 0.8},
    {"name": "EMA (European Medicines Agency) approval granted", "score": 0.9},
    {"name": "EMA PRIME designation granted", "score": 0.8},
    {"name": "MHRA (UK) approval granted", "score": 0.8},
    {"name": "Health Canada approval granted", "score": 0.8},
    {"name": "Company moves into a hot new sector", "score": 0.7},
    {"name": "Company gains significant market share", "score": 0.7},
    {"name": "Collaboration or separation with a major company", "score": 0.7},
    {"name": "Large contract win or cancellation announced", "score": 0.8},
    {"name": "Major contract renewal or extension", "score": 0.7},
    {"name": "Major customer announced or named", "score": 0.7},
    {"name": "New funding round", "score": 0.5},
    {"name": "Cost cuts", "score": 0.4},
    {"name": "Net margin improvement", "score": 0.4},
    {"name": "Analyst upgrades or downgrades", "score": 0.5},
    {"name": "CEO or major insider buying reported", "score": 0.6},
    {"name": "Macroeconomic news", "score": 0.6},
    {"name": "Major strategic partnership or licensing deal announced", "score": 0.8},
    {"name": "Joint venture announced", "score": 0.7},
    {"name": "Reverse merger announced or completed", "score": 0.8},
    {"name": "Significant SEC filing or institutional buy", "score": 0.6},
    {"name": "Added to major index (e.g., S&P, Nasdaq, Russell)", "score": 0.7},
    {"name": "International market expansion or approval", "score": 0.7},
    {"name": "Significant patent approval or litigation win", "score": 0.7},
    {"name": "First day trading after IPO or uplisting", "score": 0.8},
    {"name": "Successful up-listing to major exchange", "score": 0.8},
    {"name": "Major regulatory clearance outside US", "score": 0.8},
    {"name": "Breakthrough Device/Drug Designation awarded", "score": 0.8},
    {"name": "Dividend initiation or significant increase", "score": 0.5},
    {"name": "Major grant awarded", "score": 0.6},
    {"name": "Showcased at a prestigious event", "score": 0.3},
    {"name": "Honored for excellence or innovation", "score": 0.2},
    {"name": "Debt paid off", "score": 0.3},
    {"name": "Anchor/Sympathy Play", "score": 0.2},
    {"name": "Sector-wide sympathy move", "score": 0.5},
    {"name": "Share buyback program announced or expanded", "score": 0.6},
    {"name": "Unusual short interest or squeeze alert", "score": 0.9},

    # --- TECHNICAL / PRICE ---
    {"name": "Anchored/2-Day VWAP", "score": 0.4},
    {"name": "Moving Average", "score": 0.3},
    {"name": "Candlestick Pattern", "score": 0.3},
    {"name": "Chart Pattern", "score": 0.4},
    {"name": "Trendline", "score": 0.3},
    {"name": "Volume", "score": 0.5},
    {"name": "Unusual options volume", "score": 0.5},
    {"name": "Break-out or break-down angle", "score": 0.4},
    {"name": "Key support/resistance level", "score": 0.3},
    {"name": "Overnight gap of ±3% and at least 10% of ADV", "score": 0.7},
    {"name": "Held bid in uptrend visible in Level 2/T&S", "score": 0.6},
    {"name": "Gap fill", "score": 0.3},
    {"name": "Relevant recent history (last 2Y)", "score": 0.2},
    {"name": "Full/Half/Quarter price level", "score": 0.2},
    {"name": "Overextension", "score": 0.3},
    {"name": "Beaten down stock", "score": 0.3},
    {"name": "Break of all-time/52W high or low", "score": 0.7},
    {"name": "Breakout", "score": 0.7},

    # --- NEGATIVE CATALYSTS ---
    {"name": "Dilutive offering/reverse split/supply", "score": -0.8},
    {"name": "Negative trial results announced", "score": -0.9},
    {"name": "FDA Complete Response Letter (CRL) received", "score": -1.0},
    {"name": "Clinical hold imposed", "score": -0.8},
    {"name": "Product recall or safety warning issued", "score": -0.7},
    {"name": "Major cyberattack or data breach disclosed", "score": -0.7},
    {"name": "Report alleging misconduct", "score": -0.5},
    {"name": "Significant restructuring or layoffs announced", "score": 0.4},  # can be negative or positive!
    {"name": "Major government regulation or policy announced", "score": 0.7}, # can be negative or positive!
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
        "weight": 0.18,
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
        "weight": 0.08,
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
        "weight": 0.07,
    },
    {
        "name": "FloatPct",
        "question": "Premarket Volume as % of Float:",
        "options": [
            "Under 10%",
            "10% – 30%",
            "30% – 60%",
            "60% – 100%",
            "Over 100%",
        ],
        "weight": 0.10,
    },
    {
    "name": "PreMarket",
    "question": "Premarket Price Structure (Gap + Trend development):",
    "options": [
        "Immediate full reversal after gap; price gives back all gains in one move down.",
        "Choppy reversal after gap; price trends down with high volatility and overlap.",
        "Sideways, volatile range after gap; no clear trend up or down.",
        "Uptrend with deep, frequent pullbacks; volatile upward move.",
        "Uptrend with moderate pullbacks; some noise, but trend holds.",
        "Steady uptrend; higher highs and lows, only minor pullbacks.",
        "Exceptionally clean uptrend; multiple strong legs up, minimal overlap or pullbacks.",
    ],
    "weight": 0.12,
    },
    {
    "name": "Technicals",
    "question": "Technical Structure (Levels, Breaks, and Holds):",
    "options": [
        "Fails at every key level; unable to hold or reclaim any support/resistance.",
        "Briefly reclaims or holds a level but quickly loses it again; repeated failures.",
        "Holds some levels but unable to break through any major resistance.",
        "Breaks key level but cannot stay above; frequent false breaks and rejections.",
        "Breaks and holds above one significant level; most resistance levels remain above.",
        "Breaks and holds above multiple significant levels; most resistance levels are cleared.",
        "Breaks and holds above all previous resistance; blue sky with no levels overhead.",
    ],
    "weight": 0.12,
    },
    {
    "name": "Monthly",
    "question": "Monthly/Weekly Chart Context (Trend, Volume, Structure):",
    "options": [
        "Sustained, heavy downtrend with low volume; no signs of bottoming.",
        "Steep downtrend, occasional volume spikes, but no real base or reversal.",
        "Downtrend slows; price starts to form a base, volume still low.",
        "Base formation becomes clear; sideways action with rising volume.",
        "Bottom confirmed; first higher low appears, clear uptick in volume.",
        "Uprise begins; price breaks out of base, first higher highs and strong volume surge.",
        "Clear uptrend; price moves above all previous resistance, blue sky on the long-term chart.",
    ],
    "weight": 0.06,
    },
    {
    "name": "VolProfile",
    "question": "Volume Profile (Clusters, Distribution, Overhead):",
    "options": [
        "Major volume clusters are above current price; heavy overhead resistance.",
        "Multiple volume clusters above price; significant resistance on the way up.",
        "Volume is evenly distributed with no dominant cluster; no clear support or resistance.",
        "Volume clusters are mostly above price, but some support is forming below.",
        "Main volume clusters are at current price; support and resistance are mixed.",
        "Major volume clusters are below current price; strong support established underneath.",
        "All major volume clusters are well below price; no overhead resistance, price is in Blue Sky.",
    ],
    "weight": 0.06,
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

st.sidebar.header("Set Criteria Weights")

weights = {}
for crit in CRITERIA:
    weights[crit["name"]] = st.sidebar.slider(
        label=crit["question"],
        min_value=0.0,
        max_value=0.5,
        value=crit["weight"],
        step=0.01
    )
    
news_weight = st.sidebar.slider(
    label="Catalyst (News) Weight",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05
)

def heat_level(score):
    if score >= 4.5:
        return "A+"
    elif score >= 4.0:
        return "A"
    elif score >= 3.7:
        return "B"
    elif score >= 3.3:
        return "C"
    else:
        return "D"

if "stock_scores" not in st.session_state:
    st.session_state.stock_scores = []

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
    # Unbegrenzter Catalyst-Score: einfach Summe, kein max!
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
        "Ticker", "RVOL", "ATR", "Float", "FloatPct", "PreMarket", "Technicals",
        "Monthly", "VolProfile", "Spread", "Catalyst", "Score", "Level"
    ]
    
    st.dataframe(
        df[ordered_cols],
        use_container_width=True,
        hide_index=True,
        column_order=ordered_cols,
        column_config={
            "Ticker": st.column_config.Column(
                label="Ticker",
                width="small",
                required=True,
                disabled=True,
                help="Stock symbol",
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
