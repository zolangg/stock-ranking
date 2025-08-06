import streamlit as st
import pandas as pd

st.header("Premarket Stock Ranking Tool")

CATALYSTS = [
    # --- EARNINGS & GUIDANCE ---
    {"name": "Earnings report (beat/miss/surprise)", "score": 1.0},
    {"name": "Guidance raised/lowered", "score": 0.9},
    {"name": "Earnings/guidance pre-announcement", "score": 0.7},
    {"name": "Analyst upgrade/downgrade", "score": 0.5},
    {"name": "Initiation of coverage by analyst", "score": 0.4},
    {"name": "Reinstated dividend", "score": 0.6},

    # --- M&A, CORPORATE ACTION ---
    {"name": "Acquisition announced or confirmed", "score": 1.0},
    {"name": "Acquisition rumors or speculation", "score": 0.7},
    {"name": "Strategic partnership/licensing deal", "score": 0.8},
    {"name": "Collaboration with major company", "score": 0.7},
    {"name": "Joint venture announced", "score": 0.7},
    {"name": "Merger or spin-off", "score": 0.9},
    {"name": "Reverse merger completed", "score": 0.8},
    {"name": "Reverse merger target rumor", "score": 0.7},
    {"name": "Major customer/contract win", "score": 0.8},
    {"name": "Major contract renewal/extension", "score": 0.7},
    {"name": "Large government or commercial contract", "score": 0.8},
    {"name": "Added to major index (S&P, Nasdaq, Russell, etc.)", "score": 0.7},
    {"name": "Share buyback program announced", "score": 0.6},
    {"name": "Dividend initiation or increase", "score": 0.5},
    {"name": "Special dividend", "score": 0.5},
    {"name": "Stock split/reverse split", "score": 0.5},
    {"name": "IPO or uplisting to major exchange", "score": 0.8},
    {"name": "First day trading after IPO/uplisting", "score": 0.8},
    {"name": "Successful up-listing", "score": 0.8},

    # --- INSIDER, OWNERSHIP, ACTIVISM ---
    {"name": "CEO or major insider buying", "score": 0.6},
    {"name": "Insider selling", "score": -0.7},
    {"name": "Management/CEO change", "score": 0.3},
    {"name": "Change in major shareholders", "score": 0.4},
    {"name": "13D/G activist position disclosed", "score": 0.8},
    {"name": "Insider lockup expiration", "score": -0.6},
    {"name": "Shareholder activism", "score": 0.7},

    # --- FUNDING, DEBT, FINANCE ---
    {"name": "New funding round", "score": 0.5},
    {"name": "ATM offering announced", "score": -0.8},
    {"name": "ATM program terminated", "score": 0.5},
    {"name": "Dilutive offering (public/PIPE/convertible)", "score": -0.8},
    {"name": "Debt refinancing/extension", "score": 0.5},
    {"name": "Significant debt reduction", "score": 0.3},
    {"name": "Bankruptcy/restructuring filing", "score": -1.0},
    {"name": "Going concern warning removed", "score": 0.6},
    {"name": "Shelf registration withdrawn", "score": 0.6},
    {"name": "Debt paid off", "score": 0.3},

    # --- PRODUCT, BUSINESS, GROWTH ---
    {"name": "New product/feature launch", "score": 0.8},
    {"name": "Product recall or safety warning", "score": -0.7},
    {"name": "Product recall expansion", "score": -0.7},
    {"name": "Rebranding", "score": 0.3},
    {"name": "Geographic expansion", "score": 0.6},
    {"name": "Facility opening/closing", "score": 0.3},
    {"name": "Major supply chain news", "score": 0.6},
    {"name": "International market expansion/approval", "score": 0.7},
    {"name": "Company moves into hot new sector", "score": 0.7},
    {"name": "Sector-wide sympathy move", "score": 0.5},
    {"name": "Anchor/Sympathy play", "score": 0.2},

    # --- TECH/AI, CLOUD, PLATFORM ---
    {"name": "AI product announcement", "score": 0.7},
    {"name": "Cloud migration news", "score": 0.4},
    {"name": "Platform milestone achieved", "score": 0.5},
    {"name": "Key integration/partnership", "score": 0.7},
    {"name": "Major app or platform launch", "score": 0.7},
    {"name": "NFT/crypto product news", "score": 0.6},

    # --- REGULATORY/APPROVALS (BIOTECH/PHARMA/TECH) ---
    {"name": "FDA approval granted", "score": 1.0},
    {"name": "FDA Fast Track/Breakthrough/Orphan/Accelerated Approval", "score": 0.8},
    {"name": "Positive/negative FDA panel/advisory meeting", "score": 0.7},
    {"name": "Emergency Use Authorization (EUA) granted", "score": 1.0},
    {"name": "FDA meeting scheduled or results announced", "score": 0.9},
    {"name": "Positive/negative EMA/MHRA/Health Canada approval", "score": 0.8},
    {"name": "Major regulatory clearance outside US", "score": 0.8},
    {"name": "New or delayed product approval/launch", "score": 0.8},
    {"name": "Regulatory approval for new market", "score": 0.7},
    {"name": "Sanctions/tariffs impact", "score": -0.8},

    # --- DATA & STUDY RESULTS (BIOTECH/MEDTECH) ---
    {"name": "Positive/negative Phase 1/2/3/4 trial results", "score": 0.7},
    {"name": "New clinical data released", "score": 0.9},
    {"name": "Study readout: topline or peer-reviewed publication", "score": 0.7},
    {"name": "Positive/negative results of independent or ongoing study", "score": 0.8},
    {"name": "Patent award or litigation win", "score": 0.7},
    {"name": "Major patent application filed", "score": 0.4},
    {"name": "IP portfolio acquisition/licensing", "score": 0.7},

    # --- LITIGATION & LEGAL ---
    {"name": "Lawsuit filed", "score": -0.6},
    {"name": "Lawsuit settlement", "score": 0.4},
    {"name": "ITC/USPTO decision", "score": 0.5},
    {"name": "Trade secrets ruling", "score": 0.5},
    {"name": "Major litigation loss", "score": -0.7},

    # --- SHORT SQUEEZE, SOCIAL MEDIA & TECHNICAL ---
    {"name": "Unusual short interest/squeeze alert", "score": 0.9},
    {"name": "Trending on social media (Reddit, X, etc)", "score": 0.6},
    {"name": "Added to meme stock watchlist", "score": 0.5},
    {"name": "Viral PR/news coverage", "score": 0.6},
    {"name": "Technical breakout or all-time high/52w high", "score": 0.7},
    {"name": "Anchored/2-Day VWAP reclaim", "score": 0.4},
    {"name": "Moving average crossover", "score": 0.3},
    {"name": "Volume/relative volume surge", "score": 0.5},
    {"name": "Unusual options activity", "score": 0.5},
    {"name": "Breakout", "score": 0.7},
    {"name": "Overnight gap of ±3% and at least 10% of ADV", "score": 0.7},
    {"name": "Held bid in uptrend (L2/T&S)", "score": 0.6},
    {"name": "Gap fill", "score": 0.3},
    {"name": "Relevant recent price/volume history", "score": 0.2},
    {"name": "Overextension", "score": 0.3},
    {"name": "Beaten down stock", "score": 0.3},

    # --- EVENTS & RECOGNITION ---
    {"name": "Showcased at prestigious event/trade show", "score": 0.3},
    {"name": "Honored for excellence or innovation", "score": 0.2},
    {"name": "Major grant or prize awarded", "score": 0.6},

    # --- NEGATIVE EVENTS ---
    {"name": "Negative trial results announced", "score": -0.9},
    {"name": "FDA Complete Response Letter (CRL) received", "score": -1.0},
    {"name": "Clinical hold imposed", "score": -0.8},
    {"name": "Product recall or safety warning", "score": -0.7},
    {"name": "Cyberattack/data breach", "score": -0.7},
    {"name": "Report alleging misconduct/fraud", "score": -0.5},
    {"name": "Significant restructuring or layoffs", "score": -0.4},
    {"name": "Major government regulation or policy", "score": -0.7},
    {"name": "Major litigation loss", "score": -0.7},

    # --- OTHER / MISC ---
    {"name": "Debt paid off", "score": 0.3},
    {"name": "Significant institutional buy/SEC filing", "score": 0.6},
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
            "Gap fully reversed: price loses >80% of gap.",
            "Choppy reversal: price loses 50–80% of gap.",
            "Partial retracement: price loses 25–50% of gap.",
            "Sideways consolidation: gap holds, price moves within top 25% of gap.",
            "Uptrend with deep pullbacks (>30% retrace or ≥1.5× ATR).",
            "Uptrend with moderate pullbacks (10–30% retrace or ~1× ATR).",
            "Clean uptrend, only minor pullbacks (<10% or <0.5× ATR).",
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
    {
    "name": "DilutionRisk",
    "question": "Dilution & Overhead Supply Risk (SEC Filings):",
    "options": [
        "Active S-1/F-1 or ATM, large shelf unused, many in-the-money warrants, urgent cash need.",
        "Open ATM/shelf with capacity, pending offering, warrants/converts near price, cash needed <6 months.",
        "ATM/shelf partly used, some warrants/options near price, cash runway 6–9 months.",
        "Shelf/ATM mostly used or expiring, few new warrants/options above price, cash for 9–12 months.",
        "Minimal overhang: few OTM warrants/options, no active ATM, shelf nearly expired, cash for 1+ year.",
        "No active ATM/shelf, only rare OTM warrants/options, no recent filings, strong cash.",
        "No shelf, ATM, warrants, converts, or dilution filings; cash-rich, no overhang.",
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
    if score >= 6.5:
        return "A++"
    elif score >= 6:
        return "A+"
    elif score >= 5.3:
        return "A"
    elif score >= 4.5:
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
