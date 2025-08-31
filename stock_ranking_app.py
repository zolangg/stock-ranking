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
# Sidebar: Weights & Modifiers
# -------------------------------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (%)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01
    )

st.sidebar.header("Modifiers")
news_weight = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

# Normalize numeric & qualitative blocks separately
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# -------------------------------
# Point mappers (for preview metrics)
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
    if float_m <= 0: return 1
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

# -------------------------------
# Odds engine (heuristics)
# -------------------------------
def odds_from_pm_heuristics(pm_vol_m: float,
                            float_m: float,
                            pm_avg_price: float,
                            mktcap_m: float,
                            target_day_vol_m: float,
                            catalyst_points: float):
    pm_share_pct = 100.0 * (pm_vol_m / max(1e-9, target_day_vol_m))
    if pm_share_pct < 1:   s1 = 5
    elif pm_share_pct < 3: s1 = 20
    elif pm_share_pct < 10:s1 = 40
    elif pm_share_pct < 15:s1 = 65
    elif pm_share_pct < 25:s1 = 80
    else:                  s1 = 95

    pm_dolv = pm_vol_m * 1_000_000 * pm_avg_price
    mc_dol  = mktcap_m * 1_000_000
    ratio_pct = 100.0 * (pm_dolv / max(1e-9, mc_dol))
    if ratio_pct < 1:    s2 = 5
    elif ratio_pct < 3:  s2 = 25
    elif ratio_pct < 5:  s2 = 45
    elif ratio_pct < 10: s2 = 70
    elif ratio_pct < 20: s2 = 85
    else:                s2 = 95

    fr_pct = 100.0 * (pm_vol_m / max(1e-9, float_m))
    if fr_pct < 1:    s3 = 10
    elif fr_pct < 3:  s3 = 25
    elif fr_pct < 10: s3 = 45
    elif fr_pct < 25: s3 = 65
    elif fr_pct < 50: s3 = 80
    elif fr_pct < 100:s3 = 90
    else:             s3 = 100

    base = 0.4*s1 + 0.3*s2 + 0.3*s3
    score = min(100.0, max(0.0, base + catalyst_points * 10.0))

    label = ("Very High Odds" if score >= 85 else
             "High Odds"      if score >= 70 else
             "Moderate Odds"  if score >= 55 else
             "Low Odds"       if score >= 40 else
             "Very Low Odds")
    return score, label

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

# -------------------------------
# Tabs: Add / Ranking
# -------------------------------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Enter Inputs")

    c_top = st.columns([1.2, 1.2, 1.0, 1.0])

    with c_top[0]:
        st.markdown("**Basics**")
        ticker        = st.text_input("Ticker", "").strip().upper()
        rvol          = st.number_input("RVOL", min_value=0.0, value=5.0, step=0.1)
        atr_usd       = st.number_input("ATR ($)", min_value=0.0, value=0.40, step=0.01, format="%.2f")

    with c_top[1]:
        st.markdown("**Float & SI**")
        float_m       = st.number_input("Public Float (Millions)", min_value=0.0, value=25.0, step=1.0)
        si_pct        = st.number_input("Short Interest (% of float)", min_value=0.0, value=12.0, step=0.5)
        pm_vol_m      = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=5.0, step=0.1)

    with c_top[2]:
        st.markdown("**PM Heuristic Inputs**")
        pm_avg_price  = st.number_input("PM Avg Price ($)", min_value=0.0, value=5.00, step=0.05, format="%.2f")
        mktcap_m      = st.number_input("Market Cap ($ Millions)", min_value=0.0, value=80.0, step=1.0)
        target_day_vol_m = st.number_input("Target Day Volume (Millions)", min_value=1.0, value=150.0, step=1.0)

    with c_top[3]:
        st.markdown("**Context Modifiers**")
        catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
        dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

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
                key=f"qual_{crit['name']}"
            )
            qual_points[crit["name"]] = choice[0]  # 1..7

    submitted = st.button("Add / Score", type="primary", use_container_width=True)

    if submitted and ticker:
        # Numeric/qual preview
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0
        qual_0_7 = sum(q_weights[c["name"]] * qual_points[c["name"]] for c in QUAL_CRITERIA)
        qual_pct = (qual_0_7/7.0)*100.0
        preview_score = 0.5*num_pct + 0.5*qual_pct
        preview_score = preview_score + catalyst_points*news_weight*10 + dilution_points*dilution_weight*10

        # Odds derived from PM heuristics (+ catalyst)
        base_odds, odds_label = odds_from_pm_heuristics(
            pm_vol_m=pm_vol_m,
            float_m=float_m,
            pm_avg_price=pm_avg_price,
            mktcap_m=mktcap_m,
            target_day_vol_m=target_day_vol_m,
            catalyst_points=catalyst_points * news_weight
        )
        odds_score = min(100.0, max(0.0, base_odds + dilution_points * dilution_weight * 5.0))

        # Append row (new schema)
        st.session_state.rows.append({
            "Ticker": ticker,
            "Odds": odds_label,
            "OddsScore": round(odds_score, 2),
            "Level": grade(odds_score),
        })

        # Last card snapshot (safe keys)
        st.session_state.last = {
            "Ticker": ticker,
            "Numeric_%": round(num_pct,2),
            "Qual_%": round(qual_pct,2),
            "Preview_Score": round(preview_score,2),
            "OddsScore": round(odds_score,2),
            "Odds": odds_label,
            "Level": grade(odds_score),
            "PM%Target": round(100.0 * pm_vol_m / max(1e-9, target_day_vol_m), 2),
            "PM$ / MC%": round(100.0 * (pm_vol_m*1_000_000*pm_avg_price) / max(1e-9, mktcap_m*1_000_000), 2),
            "PM Float %": round(100.0 * pm_vol_m / max(1e-9, float_m), 2),
        }

        st.success(f"Saved {ticker} – {odds_label} (OddsScore {round(odds_score,2)})")

    # Preview card (robust to old schema)
    l = st.session_state.last or {}
    if l:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Ticker", l.get("Ticker","—"))
        c2.metric("Odds", f"{l.get('Odds','—')} ({l.get('OddsScore','—')})")
        c3.metric("PM % of Target", f"{l.get('PM%Target','—')}%")
        c4.metric("PM$ / MC", f"{l.get('PM$ / MC%','—')}%")
        cA, cB, cC = st.columns(3)
        cA.metric("PM Float %", f"{l.get('PM Float %','—')}%")
        cB.metric("Numeric Block", f"{l.get('Numeric_%','—')}%")
        cC.metric("Qual Block", f"{l.get('Qual_%','—')}%")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)

        # Ensure new columns exist for any older rows
        for col in ["Odds","OddsScore","Level"]:
            if col not in df.columns:
                df[col] = None

        # Sort safely even if some OddsScore are missing
        df["OddsScore"] = pd.to_numeric(df["OddsScore"], errors="coerce")
        df = df.sort_values("OddsScore", ascending=False, na_position="last").reset_index(drop=True)

        st.dataframe(
            df[["Ticker","Odds","OddsScore","Level"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download CSV",
            df[["Ticker","Odds","OddsScore","Level"]].to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        c1, c2 = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = None
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
