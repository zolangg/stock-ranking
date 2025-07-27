import streamlit as st
import pandas as pd

st.header("Premarket Stock Ranking Tool")

CATALYSTS = [
    {"name": "Unusually good/bad earnings report (surprise!)", "score": 1.0},
    {"name": "Better/worse than expected guidance reported", "score": 0.9},
    {"name": "Market can't put a ceiling on earnings (growth stocks)", "score": 0.9},
    {"name": "New or delayed product announcement", "score": 0.8},
    {"name": "Positive/negative results of ongoing study announced", "score": 0.8},
    {"name": "Positive/negative results of independent study announced", "score": 0.9},
    {"name": "Positive/negative results of completed study announced", "score": 0.9},
    {"name": "Company moves into a hot new sector", "score": 0.7},
    {"name": "Company gains significant market share", "score": 0.7},
    {"name": "Collaboration or separation with a major company", "score": 0.7},
    {"name": "Dilutive offering/reverse split/supply", "score": -0.8},
    {"name": "Government regulatory announcement (positive/negative)", "score": 1.0},
    {"name": "Large contract win or cancellation announced", "score": 0.8},
    {"name": "New funding round", "score": 0.5},
    {"name": "Cost cuts", "score": 0.4},
    {"name": "Net margin improvement", "score": 0.4},
    {"name": "Analyst upgrades or downgrades", "score": 0.5},
    {"name": "Macroeconomic news", "score": 0.6},
    {"name": "Management changes", "score": 0.3},
    {"name": "Dividend announcement", "score": 0.2},
    {"name": "Report alleging misconduct", "score": -0.5},
    {"name": "Showcased at a prestigious event", "score": 0.3},
    {"name": "Honored for excellence or innovation", "score": 0.2},
    {"name": "Debt paid off", "score": 0.3},
    {"name": "Anchor/Sympathy Play", "score": 0.2},
    # TECHNICAL
    {"name": "Anchored/2-Day VWAP", "score": 0.4},
    {"name": "Moving Average", "score": 0.3},
    {"name": "Candlestick Pattern", "score": 0.3},
    {"name": "Chart Pattern", "score": 0.4},
    {"name": "Trendline", "score": 0.3},
    {"name": "Volume", "score": 0.5},
    {"name": "Unusual options volume", "score": 0.5},
    {"name": "Break-out or break-down angle", "score": 0.4},
    {"name": "Key support/resistance level", "score": 0.3},
    # PRICE
    {"name": "Overnight gap of ±3% and at least 10% of ADV", "score": 0.7},
    {"name": "Held bid in uptrend visible in Level 2/T&S", "score": 0.6},
    {"name": "Gap fill", "score": 0.3},
    {"name": "Relevant recent history (last 2Y)", "score": 0.2},
    {"name": "Full/Half/Quarter price level", "score": 0.2},
    {"name": "Overextension", "score": 0.3},
    {"name": "Beaten down stock", "score": 0.3},
    {"name": "Break of all-time/52W high or low", "score": 0.7},
    {"name": "Breakout", "score": 0.7},
]

CRITERIA = [
    {
        "name": "RVOL",
        "question": "Relative Volume (RVOL):",
        "options": [
            "1 – Under 1.0 (very low, no real attention)",
            "2 – 1.0–2.0 (low, weak setup)",
            "3 – 2.0–5.0 (moderate, typical smallcap premarket)",
            "4 – 5.0–10.0 (high, strong in-play stock)",
            "5 – Over 10.0 (exceptional volume, extreme attention)",
        ],
        "weight": 0.18,
    },
    {
        "name": "ATR",
        "question": "Average True Range (ATR) in USD:",
        "options": [
            "1 – Less than 0.10 (no range, not tradeable)",
            "2 – 0.10 – 0.20 (tight, limited opportunity)",
            "3 – 0.20 – 0.50 (average, can work)",
            "4 – 0.50 – 1.00 (wide, high potential)",
            "5 – Over 1.00 (huge range, big moves possible)",
        ],
        "weight": 0.08,
    },
    {
        "name": "Float",
        "question": "Public Float (shares):",
        "options": [
            "1 – >100M (very high, hard to move)",
            "2 – 50M–100M (high, slow mover)",
            "3 – 25M–50M (medium, can move)",
            "4 – 10M–25M (low, can squeeze)",
            "5 – <10M (ultra low, explosive potential)",
        ],
        "weight": 0.09,
    },
    {
        "name": "FloatPct",
        "question": "Premarket Volume as % of Float:",
        "options": [
            "1 – <2% (very low, weak setup)",
            "2 – 2%–10% (low, not in-play)",
            "3 – 10%–30% (solid, gaining traction)",
            "4 – 30%–100% (strong, clear in-play momentum)",
            "5 – Over 100% (exceptional: full float rotation, extreme momentum)",
        ],
        "weight": 0.10,
    },
    {
        "name": "PreMarket",
        "question": "Premarket Price Structure:",
        "options": [
            "1 – Flat/no action",
            "2 – Choppy, random, no levels",
            "3 – Gap with volume, but messy",
            "4 – Clean move, minor noise",
            "5 – Strong trend, clean triggers",
        ],
        "weight": 0.10,
    },
    {
        "name": "Technicals",
        "question": "Technical Setup:",
        "options": [
            "1 – No setup, heavy overhead",
            "2 – Many resistances, messy",
            "3 – Some overhead, average",
            "4 – Clear pattern, low resistance",
            "5 – Perfect: no resistance, blue-sky/gap-up",
        ],
        "weight": 0.10,
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "1 – No context, dead/sideways",
            "2 – Old downtrend, no volume",
            "3 – Downtrend with volume, many levels",
            "4 – Recent volume surge, minor old resistance",
            "5 – Breakout: fresh volume, clean chart",
        ],
        "weight": 0.08,
    },
    {
        "name": "VolProfile",
        "question": "Volume Profile:",
        "options": [
            "1 – Flat/no structure",
            "2 – Choppy, many clusters",
            "3 – Many clusters, some resistance",
            "4 – Good clusters, little overhead",
            "5 – Clean cluster, no overhead",
        ],
        "weight": 0.08,
    },
    {
        "name": "Spread",
        "question": "Bid-Ask Spread:",
        "options": [
            "1 – >3% (very wide, untradeable)",
            "2 – 2%–3% (wide, risky)",
            "3 – 1%–2% (average, manageable)",
            "4 – 0.5%–1% (tight, good fills)",
            "5 – <0.5% (super tight, ideal)",
        ],
        "weight": 0.09,
    },
]
CATALYST_WEIGHT = 0.10

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
        st.markdown(f"##### {crit['question']}")
        idx = st.radio(
            "",
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
    catalyst_score = min(max(catalyst_score, 0), 1.0)
    catalyst_points = round(catalyst_score * 5, 2)
    submit = st.form_submit_button("Save Stock & Add to Ranking")

if submit and ticker:
    base_score = sum(
        criteria_points[crit['name']] * crit['weight'] for crit in CRITERIA
    )
    weighted_score = base_score + catalyst_points * CATALYST_WEIGHT
    score_normalized = round(weighted_score, 2)
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

    # --- Delete row ---
    st.write("---")
    st.header("Remove Stock from Ranking")

    ticker_list = [entry["Ticker"] for entry in st.session_state.stock_scores]
    if len(ticker_list) > 0:
        delete_ticker = st.selectbox("Select ticker to remove", ticker_list, key="delete_ticker")
        if st.button("Delete this stock from ranking"):
            del_idx = ticker_list.index(delete_ticker)
            st.session_state.stock_scores.pop(del_idx)
            st.success(f"Stock {delete_ticker} removed! Table and export updated.")
    else:
        st.info("No stocks saved yet — nothing to remove.")

else:
    st.info("No stocks have been ranked yet. Please fill out the form above!")