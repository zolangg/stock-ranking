import streamlit as st
import pandas as pd

# Compat helper: use st.rerun if available, else experimental_rerun if present
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        
# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# -------------------------------
# Session state init
# -------------------------------
if "rows" not in st.session_state:
    st.session_state.rows = []
if "last" not in st.session_state:
    st.session_state.last = None
if "flash" not in st.session_state:
    st.session_state.flash = None

# Flash from prior submit
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

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

# -------------------------------
# Sidebar: Weights & Modifiers
# -------------------------------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01, key="w_rvol", help="Relative volume weight.")
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01, key="w_atr",  help="ATR weight.")
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01, key="w_si", help="Short interest weight.")
w_fr    = st.sidebar.slider("PM Float Rotation (%)", 0.0, 1.0, 0.45, 0.01, key="w_fr", help="PM float rotation weight.")
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01, key="w_float", help="Float size points weight.")

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}",
        help=f"Weight for {crit['name']}."
    )

st.sidebar.header("Modifiers")
news_weight = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05, key="news_weight",
                                help="Multiplier on Catalyst slider value.")
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05, key="dil_weight",
                                    help="Multiplier on Dilution slider value.")

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
# Tabs
# -------------------------------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # ---------- OPTION A: form that clears on submit ----------
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=5.0, step=0.1)
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.40, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=25.0, step=1.0)
        
        # Float / SI / PM volume + Target
        with c_top[1]:
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=12.0, step=0.5)
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=5.0, step=0.1)
            target_vol_m = st.number_input("Target Day Volume (Millions)", min_value=1.0, value=150.0, step=5.0)
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=5.00, step=0.05, format="%.2f")
        
        # Price, Cap & Modifiers
        with c_top[2]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=100.0, step=5.0)
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.subheader("**Qualitative Context**")

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

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    # ---------- Scoring after submit ----------
    if submitted and ticker:
        # Numeric points
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)

        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # Qualitative points
        qual_0_7 = sum(q_weights[c["name"]] * qual_points[c["name"]] for c in QUAL_CRITERIA)
        qual_pct = (qual_0_7/7.0)*100.0

        # Combine + modifiers
        combo_pct = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)

        # Diagnostics (for preview)
        pm_pct_target = 100.0 * pm_vol_m / target_vol_m if target_vol_m > 0 else 0.0
        pm_float_pct  = 100.0 * pm_vol_m / float_m     if float_m     > 0 else 0.0
        pm_dollar_vol_m = pm_vol_m * pm_vwap
        pm_dollar_vs_mc_pct = 100.0 * pm_dollar_vol_m / mc_m if mc_m > 0 else 0.0

        # Save row
        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "OddsScore": final_score,
            "Level": grade(final_score),
        }
        st.session_state.rows.append(row)

        # Save last for preview card
        st.session_state.last = {
            "Ticker": ticker,
            "Numeric_%": round(num_pct,2),
            "Qual_%": round(qual_pct,2),
            "Catalyst": round(catalyst_points,2),
            "Dilution": round(dilution_points,2),
            "Final": final_score,
            "Level": grade(final_score),
            "Odds": odds_label(final_score),
            "PM_Target_%": round(pm_pct_target,1),
            "PM_Float_%": round(pm_float_pct,1),
            "PM_$Vol_M": round(pm_dollar_vol_m,2),
            "PM$ / MC_%": round(pm_dollar_vs_mc_pct,1),
        }

        st.session_state.flash = f"Saved {ticker} – Final Score {final_score} ({row['Level']})"
        do_rerun()

    # Preview card (after rerun)
    if st.session_state.last:
        st.markdown("---")
        l = st.session_state.last
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l["Ticker"])
        cB.metric("Numeric Block", f'{l["Numeric_%"]}%')
        cC.metric("Qual Block", f'{l["Qual_%"]}%')
        cD.metric("Final Score", f'{l["Final"]} ({l["Level"]})')

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM % of Target", f'{l["PM_Target_%"]}%')
        d1.caption("PM volume ÷ target day volume × 100.")
        d2.metric("PM Float %", f'{l["PM_Float_%"]}%')
        d2.caption("PM volume ÷ float × 100.")
        d3.metric("PM $Vol (M)", f'{l["PM_$Vol_M"]}')
        d3.caption("PM Vol × PM VWAP (in $ millions).")
        d4.metric("PM $Vol / MC", f'{l["PM$ / MC_%"]}%')
        d4.caption("PM dollar volume ÷ market cap × 100.")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.sort_values("OddsScore", ascending=False).reset_index(drop=True)

        st.dataframe(
            df[["Ticker","Odds","Level"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Symbol."),
                "Odds": st.column_config.TextColumn(
                    "Odds",
                    help="Qualitative label from Final Score."
                ),
                "Level": st.column_config.TextColumn(
                    "Level",
                    help="Letter grade from Final Score."
                ),
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
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
