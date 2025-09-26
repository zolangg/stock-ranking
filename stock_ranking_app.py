import streamlit as st
import pandas as pd
import re

# ---------- Page ----------
st.set_page_config(page_title="Premarket Table + Models", layout="wide")
st.title("Premarket Table + Models")

# ---------- Markdown table helper (2 decimals everywhere numeric) ----------
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

# ---------- Qualitative criteria (short labels for compact table) ----------
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
        "short": [
            "Full reversal", "Choppy reversal", "25–50% retrace",
            "Sideways (top 25%)", "Uptrend (deep PBs)", "Uptrend (mod PBs)",
            "Clean uptrend",
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
        "short": [
            "Fails levels", "Brief hold; loses", "Holds support; capped",
            "Breaks then loses", "Holds 1 major", "Holds several", "Above all",
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
        "short": [
            "Accel downtrend", "Downtrend", "Flattening",
            "Base", "Higher low", "BO from base", "Sustained uptrend",
        ],
    },
]

# ---------- Tabs ----------
tab_manual, tab_models = st.tabs(["📝 Manual Table", "📥 Upload & Models"])

# =============================================================================
# TAB 1 — Manual Table (no ranking)
# =============================================================================
with tab_manual:
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
        qual_tags = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),  # (1..7, long)
                    format_func=lambda x: f"{x[0]}. {x[1]}",
                    key=f"qual_{crit['name']}"
                )
                lvl = int(choice[0])
                qual_tags[crit["name"]] = f"{lvl}-{crit['short'][lvl-1]}"

        submitted = st.form_submit_button("Add", use_container_width=True)

    if submitted and ticker:
        fr_x = (pm_vol/float_m) if float_m > 0 else 0.0
        pmmc_pct = (pm_dol/mc_m*100.0) if mc_m > 0 else 0.0
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
            "FR_x": fr_x,
            "PM$Vol/MC_%": pmmc_pct,
            "GapStruct": qual_tags.get("GapStruct",""),
            "LevelStruct": qual_tags.get("LevelStruct",""),
            "Monthly": qual_tags.get("Monthly",""),
        }
        st.session_state.rows.append(row)
        do_rerun()

    st.subheader("Table")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
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
                "GapStruct": st.column_config.TextColumn("GapStruct (lvl-tag)"),
                "LevelStruct": st.column_config.TextColumn("LevelStruct (lvl-tag)"),
                "Monthly": st.column_config.TextColumn("Monthly (lvl-tag)"),
            }
        )

        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "premarket_table.csv", "text/csv",
                use_container_width=True
            )
        with c2:
            if st.button("Clear Table", use_container_width=True):
                st.session_state.rows = []
                do_rerun()

        st.markdown("### 📋 Table (Markdown)")
        cols = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                "PM_Vol_M","PM_$Vol_M$","Catalyst","Dilution","FR_x","PM$Vol/MC_%",
                "GapStruct","LevelStruct","Monthly"]
        st.code(df_to_markdown_table(df, cols), language="markdown")
    else:
        st.info("Add a stock above to populate the table.")

# =============================================================================
# TAB 2 — Upload & Models (build “model stock” rows from your DB)
# =============================================================================
with tab_models:
    st.subheader("Upload workbook")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    sheet_main = st.text_input("Data sheet name", "PMH BO Merged")
    sheet_legend = st.text_input("Legend sheet name (optional)", "Legend")
    build_btn = st.button("Build Models", use_container_width=True)

    # --- helpers: column picking with fuzzy match ---
    def _norm(s: str) -> str:
        s = re.sub(r"\s+", " ", str(s).strip().lower())
        return (s.replace("$","").replace("%","").replace("("," ").replace(")"," ")
                 .replace("’","").replace("'",""))

    def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = list(df.columns)
        norm = {c: _norm(c) for c in cols}
        # exact normalized match
        for cand in candidates:
            n = _norm(cand)
            for c in cols:
                if norm[c] == n: return c
        # substring fallback
        for cand in candidates:
            n = _norm(cand)
            for c in cols:
                if n in norm[c]: return c
        return None

    def build_model_rows(df: pd.DataFrame) -> dict:
        """
        Returns dict of DataFrames:
          - 'FT1': one-row model built from FT==1 subset
          - 'FT0': one-row model from FT==0 subset (if present)
          - 'MaxPush': one-row model from top decile of 'Max Push Daily' (if present)
        """
        out = {}

        # map columns
        col_ft    = pick_col(df, ["ft","FT"])
        col_mc    = pick_col(df, ["market cap (m)","mcap m","mcap","marketcap m"])
        col_float = pick_col(df, ["float m shares","public float (m)","float (m)","float"])
        col_si    = pick_col(df, ["short interest %","short float %","si","short interest (float) %"])
        col_gap   = pick_col(df, ["gap %","gap%","premarket gap","gap"])
        col_atr   = pick_col(df, ["atr","atr $","atr$","atr (usd)"])
        col_rvol  = pick_col(df, ["rvol @ bo","rvol","relative volume"])
        col_pmvol = pick_col(df, ["pm vol (m)","pm volume (m)","premarket vol (m)","pm shares (m)"])
        col_pmdol = pick_col(df, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
        col_cat   = pick_col(df, ["catalyst","news","pr"])
        col_dil   = pick_col(df, ["dilution","offer","atm flag","atm"])
        col_push  = pick_col(df, ["max push daily","max push %", "maxpush", "max push"])

        need = [col_mc, col_float, col_si, col_gap, col_atr, col_rvol, col_pmvol, col_pmdol]
        if not all(c is not None for c in need):
            missing = [n for n, c in zip(
                ["MarketCap","Float","SI","Gap%","ATR","RVOL","PM Vol (M)","PM $Vol (M)"], need
            ) if c is None]
            st.error(f"Missing required columns in sheet: {', '.join(missing)}")
            return out

        # numeric frame
        use_cols = { 
            "MarketCap_M$": col_mc, "Float_M": col_float, "ShortInt_%": col_si,
            "Gap_%": col_gap, "ATR_$": col_atr, "RVOL": col_rvol,
            "PM_Vol_M": col_pmvol, "PM_$Vol_M$": col_pmdol
        }
        base = pd.DataFrame({
            k: pd.to_numeric(df[v], errors="coerce") for k, v in use_cols.items()
        })
        if col_cat:
            base["Catalyst"] = pd.to_numeric(df[col_cat], errors="coerce").fillna(0.0).clip(-1, 1)
        else:
            base["Catalyst"] = 0.0
        if col_dil:
            base["Dilution"] = pd.to_numeric(df[col_dil], errors="coerce").fillna(0.0).clip(-1, 1)
        else:
            base["Dilution"] = 0.0

        # derived
        base["FR_x"] = base.apply(lambda r: (r["PM_Vol_M"]/r["Float_M"]) if r["Float_M"]>0 else 0.0, axis=1)
        base["PM$Vol/MC_%"] = base.apply(lambda r: (r["PM_$Vol_M$"]/r["MarketCap_M$"]*100.0) if r["MarketCap_M$"]>0 else 0.0, axis=1)

        # FT filters
        if col_ft and col_ft in df.columns:
            ft_series = pd.to_numeric(df[col_ft], errors="coerce")
            mask_ft1 = (ft_series == 1)
            mask_ft0 = (ft_series == 0)
        else:
            mask_ft1 = pd.Series([False]*len(df))
            mask_ft0 = pd.Series([False]*len(df))

        # helper to compute medians safely
        def model_from_subset(sub: pd.DataFrame, label: str) -> pd.DataFrame | None:
            if sub.empty: 
                return None
            med = sub.median(numeric_only=True)
            row = pd.DataFrame([{
                "Ticker": f"MODEL_{label}",
                "MarketCap_M$": med.get("MarketCap_M$", 0.0),
                "Float_M": med.get("Float_M", 0.0),
                "ShortInt_%": med.get("ShortInt_%", 0.0),
                "Gap_%": med.get("Gap_%", 0.0),
                "ATR_$": med.get("ATR_$", 0.0),
                "RVOL": med.get("RVOL", 0.0),
                "PM_Vol_M": med.get("PM_Vol_M", 0.0),
                "PM_$Vol_M$": med.get("PM_$Vol_M$", 0.0),
                "Catalyst": med.get("Catalyst", 0.0),
                "Dilution": med.get("Dilution", 0.0),
                "FR_x": med.get("FR_x", 0.0),
                "PM$Vol/MC_%": med.get("PM$Vol/MC_%", 0.0),
                # Qualitative left blank (not derivable)
                "GapStruct": "",
                "LevelStruct": "",
                "Monthly": "",
            }])
            return row

        # FT=1
        sub_ft1 = base[mask_ft1] if mask_ft1.any() else pd.DataFrame(columns=base.columns)
        m_ft1 = model_from_subset(sub_ft1, "FT1")
        if m_ft1 is not None: out["FT1"] = m_ft1

        # FT=0
        sub_ft0 = base[mask_ft0] if mask_ft0.any() else pd.DataFrame(columns=base.columns)
        m_ft0 = model_from_subset(sub_ft0, "FT0")
        if m_ft0 is not None: out["FT0"] = m_ft0

        # Max Push Daily (top decile)
        if col_push:
            push_vals = pd.to_numeric(df[col_push], errors="coerce")
            if push_vals.notna().any():
                thr = push_vals.quantile(0.90)
                mask_push = push_vals >= thr
                sub_push = base[mask_push.fillna(False)]
                m_push = model_from_subset(sub_push, "MaxPush")
                if m_push is not None:
                    out["MaxPush"] = m_push

        return out

    if build_btn:
        if not uploaded:
            st.error("Upload a workbook first.")
        else:
            try:
                xls = pd.ExcelFile(uploaded)
                if sheet_main not in xls.sheet_names:
                    st.error(f"Sheet '{sheet_main}' not found. Available: {xls.sheet_names}")
                else:
                    raw = pd.read_excel(xls, sheet_main)
                    models = build_model_rows(raw)

                    if not models:
                        st.warning("No models could be built from the uploaded sheet (check required columns).")
                    else:
                        st.success(f"Built {len(models)} model table(s).")

                        # Show each model table (same columns & formatting as manual)
                        col_cfg = {
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
                            "GapStruct": st.column_config.TextColumn("GapStruct (lvl-tag)"),
                            "LevelStruct": st.column_config.TextColumn("LevelStruct (lvl-tag)"),
                            "Monthly": st.column_config.TextColumn("Monthly (lvl-tag)"),
                        }
                        cols_order = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                                      "PM_Vol_M","PM_$Vol_M$","Catalyst","Dilution","FR_x","PM$Vol/MC_%",
                                      "GapStruct","LevelStruct","Monthly"]

                        for name, dfm in models.items():
                            st.markdown(f"#### Model — {name}")
                            st.dataframe(dfm, use_container_width=True, hide_index=True, column_config=col_cfg)

                            c1, c2 = st.columns([0.35, 0.65])
                            with c1:
                                st.download_button(
                                    f"Download CSV ({name})",
                                    dfm.to_csv(index=False).encode("utf-8"),
                                    f"model_{name}.csv","text/csv",
                                    use_container_width=True
                                )
                            with c2:
                                st.markdown("**Markdown**")
                                st.code(df_to_markdown_table(dfm, cols_order), language="markdown")

            except Exception as e:
                st.error(f"Failed to build models: {e}")
