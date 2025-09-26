import streamlit as st
import pandas as pd

# ---------- Page ----------
st.set_page_config(page_title="Premarket Table", layout="wide")
st.title("Premarket Table")

# ---------- Markdown table helper (always 2 decimals) ----------
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
            if isinstance(v, (int, float)):
                cells.append(f"{float(v):.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []

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
    },
]

# ---------- Input Form ----------
st.subheader("Add Stock")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])

    with c1:
        ticker  = st.text_input("Ticker", "").strip().upper()
        mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
        si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01, format="%.2f")

    with c2:
        gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")
        atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
        rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")

    with c3:
        pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
        catalyst = st.slider("Catalyst (−1…+1)", -1.0, 1.0, 0.0, 0.05)
        dilution = st.slider("Dilution (−1…+1)", -1.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.subheader("Qualitative Context")
    q_cols = st.columns(3)
    qual_levels = {}
    qual_texts  = {}
    for i, crit in enumerate(QUAL_CRITERIA):
        with q_cols[i % 3]:
            choice = st.radio(
                crit["question"],
                options=list(enumerate(crit["options"], 1)),   # (1..7, text)
                format_func=lambda x: f"{x[0]}. {x[1]}",
                key=f"qual_{crit['name']}"
            )
            level, text = choice
            qual_levels[crit["name"]] = float(level)
            qual_texts[crit["name"]]  = text

    submitted = st.form_submit_button("Add", use_container_width=True)

# ---------- Add Row ----------
if submitted and ticker:
    # Derived metrics
    fr_x = (pm_vol/float_m) if float_m > 0 else 0.0                    # PM Float Rotation ×
    pmmc_pct = (pm_dol/mc_m*100.0) if mc_m > 0 else 0.0                # PM $Vol / MC %

    # Simple qualitative % (equal-weight mean of 3 levels; 1..7 → 0..100)
    levels = list(qual_levels.values())
    qual_pct = (sum(levels)/len(levels)/7.0*100.0) if levels else 0.0

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
        "Catalyst": catalyst,
        "Dilution": dilution,
        # Derived
        "FR_x": fr_x,
        "PM$Vol/MC_%": pmmc_pct,
        # Qualitative
        "GapStruct_Level": qual_levels.get("GapStruct", 0.0),
        "GapStruct_Text":  qual_texts.get("GapStruct", ""),
        "LevelStruct_Level": qual_levels.get("LevelStruct", 0.0),
        "LevelStruct_Text":  qual_texts.get("LevelStruct", ""),
        "Monthly_Level": qual_levels.get("Monthly", 0.0),
        "Monthly_Text":  qual_texts.get("Monthly", ""),
        "Qualitative_%": qual_pct,
    }
    st.session_state.rows.append(row)
    do_rerun()

# ---------- Table ----------
st.subheader("Table")
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)

    # Display
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "MarketCap_M$": st.column_config.NumberColumn("Market Cap (M$)", format="%.2f"),
            "Float_M": st.column_config.NumberColumn("Public Float (M)", format="%.2f"),
            "ShortInt_%": st.column_config.NumberColumn("Short Interest (%)", format="%.2f"),
            "Gap_%": st.column_config.NumberColumn("Gap %", format="%.2f"),
            "ATR_$": st.column_config.NumberColumn("ATR ($)", format="%.2f"),
            "RVOL": st.column_config.NumberColumn("RVOL", format="%.2f"),
            "PM_Vol_M": st.column_config.NumberColumn("Premarket Vol (M)", format="%.2f"),
            "PM_$Vol_M$": st.column_config.NumberColumn("Premarket $Vol (M$)", format="%.2f"),
            "Catalyst": st.column_config.NumberColumn("Catalyst", format="%.2f"),
            "Dilution": st.column_config.NumberColumn("Dilution", format="%.2f"),
            "FR_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.2f"),
            "PM$Vol/MC_%": st.column_config.NumberColumn("PM $Vol / MC (%)", format="%.2f"),
            "GapStruct_Level": st.column_config.NumberColumn("GapStruct (1–7)", format="%.2f"),
            "LevelStruct_Level": st.column_config.NumberColumn("LevelStruct (1–7)", format="%.2f"),
            "Monthly_Level": st.column_config.NumberColumn("Monthly (1–7)", format="%.2f"),
            "Qualitative_%": st.column_config.NumberColumn("Qualitative %", format="%.2f"),
        }
    )

    # Actions
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "premarket_table.csv",
            "text/csv",
            use_container_width=True
        )
    with c2:
        if st.button("Clear Table", use_container_width=True):
            st.session_state.rows = []
            do_rerun()

    # Markdown export (concise)
    st.markdown("### 📋 Table (Markdown)")
    cols_order = [
        "Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
        "PM_Vol_M","PM_$Vol_M$","Catalyst","Dilution","FR_x","PM$Vol/MC_%","Qualitative_%",
        "GapStruct_Level","LevelStruct_Level","Monthly_Level"
    ]
    st.code(df_to_markdown_table(df, cols_order), language="markdown")
else:
    st.info("Add a stock above to populate the table.")
