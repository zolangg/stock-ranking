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
            if isinstance(v, float) or isinstance(v, int):
                cells.append(f"{float(v):.2f}")
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

# ---------- Input Form ----------
st.subheader("Add Stock")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

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

    submitted = st.form_submit_button("Add", use_container_width=True)

if submitted and ticker:
    fr = (pm_vol/float_m) if float_m > 0 else 0.0                       # PM Float Rotation ×
    pmmc_pct = (pm_dol/mc_m*100.0) if mc_m > 0 else 0.0                  # PM $Vol / MC %
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
        "FR_x": fr,
        "PM$Vol/MC_%": pmmc_pct,
    }
    st.session_state.rows.append(row)
    do_rerun()

# ---------- Table ----------
st.subheader("Table")
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)

    # Display table (two-decimal formatting)
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
        }
    )

    # Actions
    c1, c2, c3 = st.columns([0.3, 0.3, 0.4])
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

    # Markdown export
    st.markdown("### 📋 Table (Markdown)")
    cols_order = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                  "PM_Vol_M","PM_$Vol_M$","Catalyst","Dilution","FR_x","PM$Vol/MC_%"]
    st.code(df_to_markdown_table(df, cols_order), language="markdown")
else:
    st.info("Add a stock above to populate the table.")
