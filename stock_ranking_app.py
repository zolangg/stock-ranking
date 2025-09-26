# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Helpers ==============================
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Convert dataframe to markdown with 2 decimals for numerics."""
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy()
    for c in keep:
        if pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].astype(float).map(lambda v: f"{v:.2f}")
        else:
            sub[c] = sub[c].astype(str)
    header = "| " + " | ".join(keep) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = [str(row[c]) for c in keep]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ============================== Session State ==============================
if "rows" not in st.session_state:
    st.session_state.rows = []   # manual entries

# ============================== Qualitative Block (full text) ==============================
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

# ============================== Tabs ==============================
tab_input, tab_tables = st.tabs(["➕ Manual Input", "📊 Tables & Divergences"])

# ------------------------------------------------------------------
# TAB 1 — Manual Input (numeric + catalyst/dilution sliders + full qualitative)
# ------------------------------------------------------------------
with tab_input:
    st.subheader("Enter Stock Data")
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

        with c1:
            ticker  = st.text_input("Ticker", "").strip().upper()
            mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
            float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
            si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01, format="%.2f")
            gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")  # e.g., 45 = +45%

        with c2:
            atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
            rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")
            pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
            pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")

        with c3:
            catalyst = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.1)
            dilution = st.slider("Dilution (0.0 … 1.0)", 0.0, 1.0, 0.0, 0.1)

        st.markdown("---")
        st.subheader("Qualitative Context (full text)")
        q_cols = st.columns(3)
        qual_vals = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                sel = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}"
                )
                # store both label and 1-based index
                qual_vals[f"Q_{crit['name']}"] = sel[1]
                qual_vals[f"_Q_{crit['name']}_idx"] = sel[0]

        submitted = st.form_submit_button("Add Stock", use_container_width=True)

    if submitted and ticker:
        # Derived metrics
        fr = (pm_vol / float_m) if float_m > 0 else 0.0
        pmmc = (pm_dol / mc_m * 100.0) if mc_m > 0 else 0.0

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
        row.update(qual_vals)
        st.session_state.rows.append(row)
        st.success(f"Saved {ticker}.")
        do_rerun()

# ------------------------------------------------------------------
# TAB 2 — Tables & Divergences
# ------------------------------------------------------------------
with tab_tables:
    st.subheader("All Entered Stocks")

    if not st.session_state.rows:
        st.info("No rows yet. Add a stock in the **Manual Input** tab.")
    else:
        df = pd.DataFrame(st.session_state.rows)

        # Display table
        show_cols = [
            "Ticker",
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%",
            "Catalyst","Dilution",
            "Q_GapStruct","Q_LevelStruct","Q_Monthly",
        ]
        for c in show_cols:
            if c not in df.columns: df[c] = ""
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in show_cols if c not in ["Ticker","Q_GapStruct","Q_LevelStruct","Q_Monthly"]}
        st.dataframe(
            df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={**num_cfg}
        )

        # Downloads & Markdown
        st.download_button(
            "Download CSV",
            df[show_cols].to_csv(index=False).encode("utf-8"),
            "stocks_table.csv",
            "text/csv",
            use_container_width=True
        )
        st.markdown("### 📋 Markdown Export")
        st.code(df_to_markdown_table(df, show_cols), language="markdown")

        st.markdown("---")
        st.subheader("Top-K Divergent Variables (across all entered stocks)")
        TOPK = st.slider("How many variables to list", 3, 12, 8, 1)
        # Only consider numeric vars for divergence
        num_vars = [
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"
        ]
        avail = [v for v in num_vars if v in df.columns]
        if len(df) >= 2 and avail:
            sub = df[avail].apply(pd.to_numeric, errors="coerce")
            stats = []
            for v in avail:
                col = sub[v].dropna()
                if col.size >= 2:
                    rng = float(col.max() - col.min())
                    std = float(col.std(ddof=1))
                    med = float(col.median())
                    cv  = float(std / (abs(med) + 1e-9))  # coefficient of variation proxy
                    stats.append({"Variable": v, "Range": rng, "Std": std, "Median": med, "CV~": cv})
            if stats:
                dfd = pd.DataFrame(stats).sort_values(["Range","Std"], ascending=False).head(TOPK)
                cfg = {
                    "Variable": st.column_config.TextColumn("Variable"),
                    "Range": st.column_config.NumberColumn("Range", format="%.2f"),
                    "Std":   st.column_config.NumberColumn("Std",   format="%.2f"),
                    "Median":st.column_config.NumberColumn("Median",format="%.2f"),
                    "CV~":   st.column_config.NumberColumn("CV~",   format="%.2f"),
                }
                st.dataframe(dfd, use_container_width=True, hide_index=True, column_config=cfg)
                st.markdown("**Markdown**")
                st.code(df_to_markdown_table(dfd, list(dfd.columns)), language="markdown")
            else:
                st.info("Not enough numeric data to compute divergences.")
        else:
            st.info("Add at least two stocks to compute divergences.")

        st.markdown("---")
        st.subheader("Pairwise Differences (select two tickers)")
        tickers = df["Ticker"].astype(str).unique().tolist()
        if len(tickers) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                tA = st.selectbox("Ticker A", tickers, index=0, key="pair_A")
            with c2:
                tB = st.selectbox("Ticker B", [t for t in tickers if t != tA], index=0, key="pair_B")

            a = df[df["Ticker"] == tA].iloc[-1]  # last entry of that ticker
            b = df[df["Ticker"] == tB].iloc[-1]
            rows = []
            for v in avail:
                va = pd.to_numeric(a.get(v), errors="coerce")
                vb = pd.to_numeric(b.get(v), errors="coerce")
                if pd.notna(va) and pd.notna(vb):
                    diff = float(va - vb)
                    ad   = float(abs(diff))
                    rows.append({"Variable": v, f"{tA}": va, f"{tB}": vb, "Δ(A−B)": diff, "|Δ|": ad})
            if rows:
                dd = pd.DataFrame(rows).sort_values("|Δ|", ascending=False)
                cfg = {
                    "Variable": st.column_config.TextColumn("Variable"),
                    f"{tA}": st.column_config.NumberColumn(f"{tA}", format="%.2f"),
                    f"{tB}": st.column_config.NumberColumn(f"{tB}", format="%.2f"),
                    "Δ(A−B)": st.column_config.NumberColumn("Δ(A−B)", format="%.2f"),
                    "|Δ|": st.column_config.NumberColumn("|Δ|", format="%.2f"),
                }
                st.dataframe(dd, use_container_width=True, hide_index=True, column_config=cfg)
                st.markdown("**Markdown**")
                st.code(df_to_markdown_table(dd, list(dd.columns)), language="markdown")
            else:
                st.info("No overlapping numeric variables between the selected tickers.")
        else:
            st.info("Need at least two distinct tickers for pairwise comparison.")

        st.markdown("---")
        if st.button("Clear All Rows", use_container_width=True):
            st.session_state.rows = []
            do_rerun()
