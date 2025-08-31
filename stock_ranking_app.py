import streamlit as st
import pandas as pd

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# -------------------------------
# Qualitative criteria (kept)
# -------------------------------
QUAL_CRITERIA = [
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
        "help": "How well the gap holds and trends. Higher choices = cleaner hold/trend.",
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
        "help": "How many key levels it breaks and holds. Higher choices = stronger structure.",
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
        "help": "Bigger-picture bias. Higher choices = healthier higher-timeframe context.",
    },
]

# -------------------------------
# Sidebar: Weights & Modifiers
# -------------------------------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01, key="w_rvol",
                            help="Relative Volume weight. RVOL = current volume ÷ typical (e.g., 10 = 10× usual).")
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01, key="w_atr",
                            help="Average True Range weight. Example: 0.40 means ~40¢ daily range.")
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01, key="w_si",
                            help="Short Interest weight (as % of float).")
w_fr    = st.sidebar.slider("PM Float Rotation (%)", 0.0, 1.0, 0.45, 0.01, key="w_fr",
                            help="Weight for Premarket Float Rotation = PM Volume ÷ Float × 100.")
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01, key="w_float",
                            help="Weight for float size points (smaller float → more points).")

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}",
        help=f"Weight for: {crit['name']}."
    )

st.sidebar.header("Modifiers")
news_weight = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05, key="news_weight",
                                help="Multiplier applied to the Catalyst slider value (−1.0…+1.0) × 10.")
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05, key="dil_weight",
                                    help="Multiplier applied to the Dilution slider value (−1.0…+1.0) × 10.")

# Normalize blocks separately
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# -------------------------------
# Mappers & labels
# -------------------------------
def pts_rvol(x: float) -> int:
    cuts = [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_atr(x: float) -> int:
    cuts = [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_si(x: float) -> int:
    cuts = [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_fr(pm_vol_m: float, float_m: float) -> int:
    if float_m <= 0:
        return 1
    pct = 100.0 * pm_vol_m / float_m
    cuts = [(1,1),(3,2),(10,3),(25,4),(50,5),(100,6)]
    for th, p in cuts:
        if pct < th: return p
    return 7

def pts_float(float_m: float) -> int:
    cuts = [(200,2),(100,3),(50,4),(35,5),(10,6)]
    if float_m <= 3: return 7
    for th, p in cuts:
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

# -------------------------------
# Session state
# -------------------------------
if "rows" not in st.session_state:
    st.session_state.rows = []
if "last" not in st.session_state:
    st.session_state.last = None
if "flash" not in st.session_state:
    st.session_state.flash = None

# Show flash from previous submit
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# -------------------------------
# Tabs: Add / Ranking
# -------------------------------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Enter Inputs")

    c_top = st.columns([1.2, 1.2, 1.0])

    # Basics
    with c_top[0]:
        st.markdown("**Basics**")
        ticker   = st.text_input("Ticker", "", key="in_ticker",
                                 help="Stock symbol, e.g., **BSLK**.")
        rvol     = st.number_input("RVOL", min_value=0.0, value=5.0, step=0.1, key="in_rvol",
                                   help="Relative Volume = current volume ÷ typical volume. Example: **10** means 10× usual.")
        atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.40, step=0.01, format="%.2f", key="in_atr",
                                   help="Average True Range in dollars. Example: **0.40** ≈ 40¢ daily range.")

    # Float / SI / PM volume + NEW: Target, PM VWAP, Market Cap
    with c_top[1]:
        st.markdown("**Float, SI & Volume**")
        float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=25.0, step=1.0, key="in_float_m",
                                   help="Tradable shares (in millions). Example: **25** = 25,000,000 shares.")
        si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=12.0, step=0.5, key="in_si_pct",
                                   help="Shorted shares as a % of float. Example: **12** = 12%.")
        pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=5.0, step=0.1, key="in_pm_vol_m",
                                   help="Shares traded in premarket (in millions). Example: **5** = 5,000,000.")
        target_vol_m = st.number_input("Target Day Volume (Millions)", min_value=1.0, value=150.0, step=5.0, key="in_target_vol_m",
                                       help="Your day-volume goal for the ticker, e.g., **150**–**200**M.")
    with c_top[2]:
        st.markdown("**Price, Cap & Modifiers**")
        pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=5.00, step=0.05, format="%.2f", key="in_pm_vwap",
                                   help="Average premarket price (VWAP) to convert PM volume → **$ volume**.")
        mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=100.0, step=5.0, key="in_mc_m",
                                   help="Approximate market cap in **millions** of USD.")
        catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05, key="in_catalyst",
                                    help="Strength of news/catalyst. **+1.0** strong positive (FDA, earnings beat), **−1.0** strong negative.")
        dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05, key="in_dilution",
                                    help="Dilution/overhang context. **−1.0** heavy ATM/S-1, **+1.0** supportive (ATM ended, buyback).")

    st.markdown("---")
    st.markdown("**Qualitative Context**")

    q_cols = st.columns(3)
    qual_points = {}
    for i, crit in enumerate(QUAL_CRITERIA):
        with q_cols[i % 3]:
            choice = st.radio(
                crit["question"],
                options=list(enumerate(crit["options"], 1)),
                format_func=lambda x: x[1],
                key=f"qual_{crit['name']}",
                help=crit.get("help", None)
            )
            qual_points[crit["name"]] = choice[0]  # 1..7

    submitted = st.button("Add / Score", type="primary", use_container_width=True)

    if submitted and ticker:
        # ---- Points (numeric) ----
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)

        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # ---- Points (qualitative) ----
        qual_0_7 = sum(q_weights[c["name"]] * qual_points[c["name"]] for c in QUAL_CRITERIA)
        qual_pct = (d := (qual_0_7/7.0)*100.0)

        # ---- Combine + modifiers ----
        combo_pct = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)

        # ---- Diagnostics ----
        pm_pct_target = 100.0 * pm_vol_m / target_vol_m if target_vol_m > 0 else 0.0
        pm_float_pct  = 100.0 * pm_vol_m / float_m     if float_m     > 0 else 0.0
        pm_dollar_vol_m = pm_vol_m * pm_vwap  # in $ millions, since pm_vol_m is in millions
        pm_dollar_vs_mc_pct = 100.0 * pm_dollar_vol_m / mc_m if mc_m > 0 else 0.0

        # Save row (include numeric OddsScore for sorting)
        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "OddsScore": final_score,
            "Level": grade(final_score),
        }
        st.session_state.rows.append(row)

        # Save last (for preview)
        st.session_state.last = {
            "Ticker": ticker,
            "Numeric_%": round(num_pct,2),
            "Qual_%": round(qual_pct,2),
            "Catalyst": round(catalyst_points,2),
            "Dilution": round(dilution_points,2),
            "Final": final_score,
            "Level": grade(final_score),
            "Odds": odds_label(final_score),
            # diagnostics
            "PM_Target_%": round(pm_pct_target,1),
            "PM_Float_%": round(pm_float_pct,1),
            "PM_$Vol_M": round(pm_dollar_vol_m,2),
            "PM$ / MC_%": round(pm_dollar_vs_mc_pct,1),
        }

        # Flash & reset inputs
        st.session_state.flash = f"Saved {ticker} – Final Score {final_score} ({row['Level']})"
        for k in [
            "in_ticker","in_rvol","in_atr","in_float_m","in_si_pct","in_pm_vol_m",
            "in_target_vol_m","in_pm_vwap","in_mc_m","in_catalyst","in_dilution"
        ] + [f"qual_{c['name']}" for c in QUAL_CRITERIA]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # Preview card for last added item
    if st.session_state.last:
        st.markdown("---")
        l = st.session_state.last
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l["Ticker"])
        cB.metric("Numeric Block", f'{l["Numeric_%"]}%')
        cC.metric("Qual Block", f'{l["Qual_%"]}%')
        cD.metric("Final Score", f'{l["Final"]} ({l["Level"]})')
       

        st.markdown("##### Premarket Diagnostics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM % of Target", f'{l["PM_Target_%"]}%')
        d1.caption("PM volume ÷ target day volume × 100.")
        d2.metric("PM Float %", f'{l["PM_Float_%"]}%')
        d2.caption("PM volume ÷ float × 100.")
        d3.metric("PM $Vol", f'{l["PM_$Vol_M"]}')
        d3.caption("PM Vol × PM VWAP.")
        d4.metric("PM $Vol / MC", f'{l["PM$ / MC_%"]}%')
        d4.caption("PM dollar volume ÷ market cap × 100.")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.sort_values("Final", ascending=False).reset_index(drop=True)

        st.dataframe(
            df[["Ticker","Odds","Level","Final","Numeric_%","Qual_%"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(
                    "Ticker",
                    help="Stock symbol you entered."
                ),
                "Odds": st.column_config.TextColumn(
                    "Odds",
                    help=("Qualitative label derived from Final Score:\n"
                          "• Very High (≥85)\n• High (≥70)\n• Moderate (≥55)\n• Low (≥40)\n• Very Low (<40)")
                ),
                "Level": st.column_config.TextColumn(
                    "Level",
                    help=("Letter grade from Final Score:\n"
                          "A++ (≥85), A+ (≥80), A (≥70), B (≥60), C (≥45), D (<45)")
                ),
                 "Final":     st.column_config.NumberColumn("Final",     format="%.2f"),
                 "Numeric_%": st.column_config.NumberColumn("Numeric %", format="%.2f"),
                 "Qual_%":    st.column_config.NumberColumn("Qual %",    format="%.2f"),
            }
        )

        st.download_button(
            "Download CSV",
            df[["Ticker","Odds","Level"]].to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        c1, c2 = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = None
                st.rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
