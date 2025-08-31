import streamlit as st
import pandas as pd

st.set_page_config(page_title="Premarket Stock Ranking – Hybrid", layout="wide")
st.header("Premarket Stock Ranking – Hybrid (Numeric + Qualitative)")

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
# Sidebar weights (numeric block)
# -------------------------------
st.sidebar.header("Weights – Numeric metrics")
w_rvol  = st.sidebar.slider("Weight: RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("Weight: ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Weight: Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("Weight: PM Float Rotation (%)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Weight: Public Float (informational or penalty)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.header("Weights – Qualitative criteria")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        f"Weight: {crit['name']}", 0.0, 1.0, crit["weight"], 0.01
    )

st.sidebar.header("Modifiers")
news_weight = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

# Normalize numeric+qual group weights separately to preserve proportions
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# -------------------------------
# Mapping helpers (numeric → 1..7)
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

# Optional: Float penalty (smaller float = higher points)
def pts_float(float_m: float) -> int:
    cuts = [(200,2),(100,3),(50,4),(35,5),(10,6)]
    if float_m <= 3: return 7
    for th, p in cuts:
        if float_m > th: return p
    return 7

# -------------------------------
# Grading
# -------------------------------
def grade(score_pct: float) -> str:
    return "A++" if score_pct >= 85 else "A+" if score_pct >= 80 else "A" if score_pct >= 70 else "B" if score_pct >= 60 else "C" if score_pct >= 45 else "D"

# -------------------------------
# Session state
# -------------------------------
if "rows" not in st.session_state:
    st.session_state.rows = []

# -------------------------------
# Input form
# -------------------------------
with st.form("row_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2,1.2,1.2])

    with c1:
        ticker   = st.text_input("Ticker", "").strip().upper()
        rvol     = st.number_input("RVOL", min_value=0.0, value=5.0, step=0.1)
        atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.40, step=0.01, format="%.2f")

    with c2:
        float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=25.0, step=1.0)
        si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=12.0, step=0.5)

    with c3:
        pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=5.0, step=0.1)
        catalyst_points = st.slider("Catalyst Impact (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
        dilution_points = st.slider("Dilution Impact (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

    st.subheader("Qualitative (keep original picks)")
    qual_points = {}
    for crit in QUAL_CRITERIA:
        choice = st.radio(
            crit["question"],
            options=list(enumerate(crit["options"], 1)),
            format_func=lambda x: x[1],
            key=f"qual_{crit['name']}"
        )
        qual_points[crit["name"]] = choice[0]  # 1..7

    submitted = st.form_submit_button("Add / Score")

# -------------------------------
# Scoring
# -------------------------------
if submitted and ticker:
    p_rvol  = pts_rvol(rvol)
    p_atr   = pts_atr(atr_usd)
    p_si    = pts_si(si_pct)
    p_fr    = pts_fr(pm_vol_m, float_m)
    p_float = pts_float(float_m)

    # numeric block (scaled to 0–100)
    num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
    num_pct = (num_0_7/7.0)*100.0

    # qualitative block (normalize by the weights you set)
    qual_0_7 = 0.0
    for crit in QUAL_CRITERIA:
        qual_0_7 += q_weights[crit["name"]] * qual_points[crit["name"]]
    qual_pct = (qual_0_7/7.0)*100.0

    # combine blocks 50/50 by default (tweak here if you want)
    combo_pct = 0.5*num_pct + 0.5*qual_pct

    # add modifiers
    final_score = round(combo_pct + catalyst_points*news_weight*10 + dilution_points*dilution_weight*10, 2)

    row = {
        "Ticker": ticker,
        # numeric
        "RVOL": rvol, "RVOL_pts": p_rvol,
        "ATR_$": atr_usd, "ATR_pts": p_atr,
        "Float_M": float_m, "Float_pts": p_float,
        "SI_%": si_pct, "SI_pts": p_si,
        "PM_Vol_M": pm_vol_m,
        "PM_FloatRot_%": (0.0 if float_m<=0 else round(100.0*pm_vol_m/float_m, 2)), "FR_pts": p_fr,
        # qualitative picks
        **{f"{k}_pts": v for k,v in qual_points.items()},
        # block scores
        "Numeric_%": round(num_pct,2),
        "Qual_%": round(qual_pct,2),
        "Catalyst": round(catalyst_points,2),
        "Dilution": round(dilution_points,2),
        "Score": final_score,
        "Level": grade(final_score),
    }
    st.session_state.rows.append(row)
    st.success(f"Saved {ticker} – Final Score {final_score} ({row['Level']})")

# -------------------------------
# Table & export
# -------------------------------
st.write("---")
st.subheader("Ranking")
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)

    cols = [
        "Ticker","Score","Level",
        "Numeric_%","Qual_%",
        "RVOL","RVOL_pts","ATR_$","ATR_pts",
        "Float_M","Float_pts","SI_%","SI_pts",
        "PM_Vol_M","PM_FloatRot_%","FR_pts",
        "GapStruct_pts","LevelStruct_pts","Monthly_pts",
        "Catalyst","Dilution",
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df[cols].to_csv(index=False).encode("utf-8"),
                       "ranking_hybrid.csv", "text/csv")
else:
    st.info("Add a stock above to see the ranking.")
