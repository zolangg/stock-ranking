import streamlit as st
import pandas as pd

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ---------- Markdown table helper ----------
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if not keep: return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy().fillna("")
    header = "| " + " | ".join(keep) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = []
        for c in keep:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}

# ---------- Qualitative criteria ----------
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
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Context:",
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
    },
]

# ---------- Sidebar: weights ----------
st.sidebar.header("Numeric Weights")
w_mc     = st.sidebar.slider("Market Cap",           0.0, 1.0, 0.10, 0.01)
w_float  = st.sidebar.slider("Public Float",         0.0, 1.0, 0.05, 0.01)
w_si     = st.sidebar.slider("Short Interest",       0.0, 1.0, 0.15, 0.01)
w_gap    = st.sidebar.slider("Gap %",                0.0, 1.0, 0.15, 0.01)
w_atr    = st.sidebar.slider("ATR ($)",              0.0, 1.0, 0.10, 0.01)
w_rvol   = st.sidebar.slider("RVOL",                 0.0, 1.0, 0.20, 0.01)
w_pmvol  = st.sidebar.slider("Premarket Volume (M)", 0.0, 1.0, 0.15, 0.01)

st.sidebar.header("Qualitative Weights")
q_weights = {crit["name"]: st.sidebar.slider(crit["name"], 0.0, 1.0, crit["weight"], 0.01) for crit in QUAL_CRITERIA}

# normalize weights
num_sum = max(1e-9, w_mc+w_float+w_si+w_gap+w_atr+w_rvol+w_pmvol)
w_mc, w_float, w_si, w_gap, w_atr, w_rvol, w_pmvol = [w/num_sum for w in (w_mc, w_float, w_si, w_gap, w_atr, w_rvol, w_pmvol)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights: q_weights[k] /= qual_sum

# ---------- Grading helper ----------
def grade(score: float) -> str:
    return (
        "A++" if score >= 90 else
        "A+"  if score >= 85 else
        "A"   if score >= 75 else
        "B"   if score >= 65 else
        "C"   if score >= 50 else "D"
    )

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    with st.form("add_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = st.number_input("Market Cap (M $)", min_value=0.0, value=0.0, step=0.01)
            float_m  = st.number_input("Public Float (M)", min_value=0.0, value=0.0, step=0.01)
            si_pct   = st.number_input("Short Interest (%)", min_value=0.0, value=0.0, step=0.01)

        with c2:
            gap_pct  = st.number_input("Gap %", min_value=0.0, value=0.0, step=0.1)
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01)
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01)
            pm_vol_m = st.number_input("Premarket Volume (M)", min_value=0.0, value=0.0, step=0.01)

        st.markdown("---")
        st.subheader("Qualitative Context")
        for crit in QUAL_CRITERIA:
            st.radio(
                crit["question"],
                options=list(enumerate(crit["options"], start=1)),
                format_func=lambda x: f"{x[0]}. {x[1]}",
                key=f"qual_{crit['name']}"
            )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # numeric score
        num_score = w_mc*mc_m + w_float*float_m + w_si*si_pct + w_gap*gap_pct + w_atr*atr_usd + w_rvol*rvol + w_pmvol*pm_vol_m
        num_pct = 100.0 * num_score / (num_score+1e-9)

        # qualitative score
        qual_score = 0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}")[0]
            qual_score += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_score/7.0)*100.0

        final_score = round(0.5*num_pct + 0.5*qual_pct, 2)
        grade_label = grade(final_score)

        row = {
            "Ticker": ticker,
            "Numeric_%": round(num_pct,2),
            "Qual_%": round(qual_pct,2),
            "FinalScore": final_score,
            "Grade": grade_label,
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        do_rerun()

    # preview
    l = st.session_state.last
    if l:
        st.markdown("---")
        a, b, c, d, e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Numeric Score", f"{l.get('Numeric_%',0):.2f}%")
        c.metric("Qualitative Score", f"{l.get('Qual_%',0):.2f}%")
        d.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        e.metric("Grade", l.get("Grade","—"))

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "ranking.csv")
        st.code(df_to_markdown_table(df, ["Ticker","Numeric_%","Qual_%","FinalScore","Grade"]), language="markdown")
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
