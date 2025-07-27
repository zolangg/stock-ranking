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
            "Under 1.0 (very low, no real attention)",
            "1.0 – 2.0 (low, weak setup)",
            "2.0 – 5.0 (moderate, typical smallcap premarket)",
            "5.0 – 10.0 (high, strong in-play stock)",
            "Over 10.0 (exceptional volume, extreme attention)",
        ],
        "weight": 0.20,
    },
    {
        "name": "ATR",
        "question": "Average True Range (ATR) in USD:",
        "options": [
            "Less than 0.10 (no range, not tradeable)",
            "0.10 – 0.20 (tight, limited opportunity)",
            "0.20 – 0.50 (average, can work)",
            "0.50 – 1.00 (wide, high potential)",
            "Over 1.00 (huge range, big moves possible)",
        ],
        "weight": 0.09,
    },
    {
        "name": "Float",
        "question": "Public Float (shares):",
        "options": [
            ">100M (very high, hard to move)",
            "50M – 100M (high, slow mover)",
            "25M – 50M (medium, can move)",
            "10M – 25M (low, can squeeze)",
            "<10M (ultra low, explosive potential)",
        ],
        "weight": 0.10,
    },
    {
        "name": "FloatPct",
        "question": "Premarket Volume as % of Float:",
        "options": [
            "<2% (very low, weak setup)",
            "2% – 10% (low, not in-play)",
            "10% – 30% (solid, gaining traction)",
            "30% – 100% (strong, clear in-play momentum)",
            "Over 100% (exceptional: full float rotation, extreme momentum)",
        ],
        "weight": 0.11,
    },
    {
    "name": "PreMarket",
    "question": "Premarket Price Structure:",
    "options": [
        "Flat/no action, no real price movement or liquidity",
        "Very choppy, random price action, no defined levels or trend",
        "Gapped up or down with significant premarket volume, but price action is messy, overlapping, or inconsistent",
        "Clean, directional move in premarket, clear trend, but with minor noise or some unclear moments",
        "Strong, continuous premarket trend with obvious and clean trigger levels, easy to spot inflection points",
    ],
    "weight": 0.11,
    },
    {
        "name": "Technicals",
        "question": "Technical Setup:",
        "options": [
            "No recognizable setup, heavy overhead resistance, chart is unattractive",
            "Many resistance levels above, price action is messy or mixed, setup is not clean",
            "Some overhead levels present but manageable, pattern is average or unclear",
            "Clear chart pattern (e.g. flag, breakout setup), minimal overhead resistance, structure is easy to interpret",
            "Perfect technicals: no resistance above, clean breakout or gap-up, 'blue sky' setup with obvious potential",
        ],
        "weight": 0.11,
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "No meaningful context, stock is flat or stuck in a sideways range for months",
            "Persistent old downtrend, no significant volume, chart is weak",
            "Stock is in a downtrend but shows increased volume, many old support/resistance levels still in play",
            "Recent major volume surge, only minor old resistance levels remain, chart is turning bullish",
            "Clean breakout from all previous levels, new volume highs, fresh uptrend, no visible resistance above",
        ],
        "weight": 0.09,
    },
    {
        "name": "VolProfile",
        "question": "Volume Profile:",
        "options": [
            "Flat volume profile, no visible structure or key levels",
            "Very choppy, many volume clusters, no dominant levels, hard to read",
            "Several volume clusters present, some act as resistance but the profile is still tradeable",
            "Good, clear volume clusters with little overhead, price can move cleanly through important levels",
            "Clean, dominant volume cluster supporting current price, no overhead resistance, easy for price to move up",
        ],
        "weight": 0.09,
    },
    {
        "name": "Spread",
        "question": "Bid-Ask Spread:",
        "options": [
            ">3% (very wide, untradeable)",
            "2% – 3% (wide, risky)",
            "1% – 2% (average, manageable)",
            "0.5% – 1% (tight, good fills)",
            "<0.5% (super tight, ideal)",
        ],
        "weight": 0.1,
    },
]

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
        criteria_points[crit['name']] * crit['weight'] for crit in CRITERIA
    )
    # Kein Gewicht, keine Begrenzung!
    score_normalized = round(base_score + catalyst_points, 2)
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