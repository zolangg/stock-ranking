# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Helpers ==============================
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Convert dataframe to markdown table with 2 decimals for floats"""
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy()
    header = "| " + " | ".join(keep) + " |"
    sep = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = []
        for c in keep:
            v = row[c]
            if isinstance(v, (int, float, np.floating)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ============================== Session state ==============================
if "rows" not in st.session_state:
    st.session_state.rows = []
if "summary" not in st.session_state:
    st.session_state.summary = []

# ============================== Qualitative criteria ==============================
QUAL_CRITERIA = [
    {"name": "GapStruct", "question": "Gap & Trend Development:",
     "options": ["Gap fully reversed", "Choppy reversal", "Partial retracement",
                 "Sideways consolidation", "Uptrend deep pullbacks",
                 "Uptrend moderate pullbacks", "Clean uptrend minor pullbacks"]},
    {"name": "LevelStruct", "question": "Key Price Levels:",
     "options": ["Fails at levels", "Briefly reclaims but loses", "Holds one support",
                 "Breaks above but loses", "Breaks and holds one",
                 "Breaks and holds several", "Blue sky breakout"]},
    {"name": "Monthly", "question": "Monthly/Weekly Context:",
     "options": ["Sharp downtrend", "Persistent downtrend", "Flattening",
                 "Sideways base", "Bottom confirmed", "Uptrend begins", "Sustained uptrend"]},
]

# ============================== Tabs ==============================
tab_input, tab_tables = st.tabs(["➕ Manual Input", "📊 Tables"])

# ------------------------------------------------------------------
# Manual Input Tab
# ------------------------------------------------------------------
with tab_input:
    st.subheader("Enter Stock Data")
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

        with c1:
            ticker = st.text_input("Ticker", "").strip().upper()
            mc_m = st.number_input("Market Cap (M$)", 0.0, step=0.01)
            float_m = st.number_input("Public Float (M)", 0.0, step=0.01)
            si_pct = st.number_input("Short Interest (%)", 0.0, step=0.01)
            gap_pct = st.number_input("Gap %", 0.0, step=0.1)

        with c2:
            atr_usd = st.number_input("ATR ($)", 0.0, step=0.01)
            rvol = st.number_input("RVOL", 0.0, step=0.01)
            pm_vol = st.number_input("Premarket Volume (M)", 0.0, step=0.01)
            pm_dol = st.number_input("Premarket $ Volume (M$)", 0.0, step=0.01)

        with c3:
            catalyst = st.slider("Catalyst (−1 … +1)", -1.0, 1.0, 0.0, 0.1)
            dilution = st.slider("Dilution (0 … 1)", 0.0, 1.0, 0.0, 0.1)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        qual_short = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                sel = st.radio(crit["question"],
                               options=list(enumerate(crit["options"], 1)),
                               format_func=lambda x: x[1])
                qual_short[crit["name"]] = sel[1]  # store short label

        submitted = st.form_submit_button("Add Stock", use_container_width=True)

    if submitted and ticker:
        fr = pm_vol / float_m if float_m > 0 else 0
        pmmc = (pm_dol / mc_m * 100) if mc_m > 0 else 0

        row = {
            "Ticker": ticker,
            "MarketCap_M$": mc_m,
            "Float_M": float_m,
            "ShortInt_%": si_pct,
            "Gap_%": gap_pct,
            "ATR_$": atr_usd,
            "RVOL": rvol,
            "PM_Vol_M": pm_vol,
            "PM_$Vol_M$": pm_dol,
            "FR_x": fr,
            "PM$Vol/MC_%": pmmc,
            "Catalyst": catalyst,
            "Dilution": dilution,
        }
        row.update(qual_short)
        st.session_state.rows.append(row)

        # also summary row
        ft_like = sum([fr > 0.5, gap_pct > 5, pmmc > 2, catalyst > 0])
        fail_like = 4 - ft_like
        st.session_state.summary.append({"Ticker": ticker, "FT_like": ft_like, "Fail_like": fail_like})

        do_rerun()

# ------------------------------------------------------------------
# Tables Tab
# ------------------------------------------------------------------
with tab_tables:
    st.subheader("All Entered Stocks")

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### 📋 Markdown Export")
        st.code(df_to_markdown_table(df, df.columns.tolist()), language="markdown")

        st.markdown("### 📊 Summary (FT-like vs Fail-like signals)")
        df_sum = pd.DataFrame(st.session_state.summary)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        st.download_button("Download Full CSV", df.to_csv(index=False).encode("utf-8"),
                           "stocks.csv", "text/csv", use_container_width=True)
    else:
        st.info("No rows yet. Add a stock in the **Manual Input** tab.")
