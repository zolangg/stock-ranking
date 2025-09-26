# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re

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
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    nm = {c:_norm(c) for c in cols}
    # exact
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c
    # contains
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]:
                return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        ss = str(s).strip().replace(" ", "")
        if "," in ss and "." not in ss:  # EU decimals
            ss = ss.replace(",", ".")
        else:
            ss = ss.replace(",", "")
        return float(ss)
    except Exception:
        return np.nan

# Qualitative config (full text + short labels for table)
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
            "Full reverse",
            "Choppy rev",
            "Partial retrace",
            "Sideways top 25%",
            "Uptrend deep PB",
            "Uptrend mod PB",
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
            "Fails levels",
            "Brief reclaim → fail",
            "Holds support only",
            "Breaks then loses",
            "Holds 1 major",
            "Holds several",
            "Blue sky",
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
            "Accel down",
            "Downtrend",
            "Flattening",
            "Base",
            "Bottomed",
            "Uptrend start",
            "Uptrend sustained",
        ],
    },
]

def _short_qual(name: str, idx1_based: int) -> str:
    for crit in QUAL_CRITERIA:
        if crit["name"] == name:
            i = max(1, min(idx1_based, len(crit["short"]))) - 1
            return crit["short"][i]
    return "—"

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []      # manual rows ONLY
if "last" not in st.session_state: st.session_state.last = {}      # last manual row
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl": DataFrame, "var_list": [...]}

# ============================== Tabs ==============================
tab_input, tab_tables = st.tabs(["➕ Manual Input", "📊 Tables & Models"])

# ============================== ➕ Manual Input ==============================
with tab_input:
    st.markdown("### Add Stock")
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

        with c1:
            ticker  = st.text_input("Ticker", "").strip().upper()
            mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
            float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
            si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01, format="%.2f")
            gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")  # real percent e.g. 100 = +100%

        with c2:
            atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
            rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")
            pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
            pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")

        with c3:
            # Probability sliders (as requested)
            catalyst = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.1)
            dilution = st.slider("Dilution (0.0 … 1.0)", 0.0, 1.0, 0.0, 0.1)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        qual_selected = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}"
                )
                q_idx = choice[0] if isinstance(choice, tuple) else int(choice)
                qual_selected[crit["name"]] = q_idx

        submitted = st.form_submit_button("Add to Table", use_container_width=True)

    if submitted and ticker:
        # Derived metrics
        fr = (pm_vol / float_m) if float_m > 0 else 0.0
        pmmc = (pm_dol / mc_m * 100.0) if mc_m > 0 else 0.0

        # Short qualitative labels for table
        q_gap   = _short_qual("GapStruct",   qual_selected.get("GapStruct", 1))
        q_level = _short_qual("LevelStruct", qual_selected.get("LevelStruct", 1))
        q_mtf   = _short_qual("Monthly",     qual_selected.get("Monthly", 1))

        row = {
            "Ticker": ticker,
            # Numeric
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
            # Sliders
            "Catalyst": catalyst,
            "Dilution": dilution,
            # Qualitative (short)
            "Q_GapStruct": q_gap,
            "Q_LevelStruct": q_level,
            "Q_Monthly": q_mtf,
            # raw idx (optional)
            "_Q_GapStruct_idx": qual_selected.get("GapStruct", 1),
            "_Q_LevelStruct_idx": qual_selected.get("LevelStruct", 1),
            "_Q_Monthly_idx": qual_selected.get("Monthly", 1),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker}.")
        do_rerun()

    # Minimal preview of last added
    l = st.session_state.last
    if l:
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("FR ×", f"{l.get('FR_x',0):.2f}")
        cC.metric("PM $Vol / MC %", f"{l.get('PM$Vol/MC_%',0):.2f}")
        cD.metric("Gap %", f"{l.get('Gap_%',0):.2f}")

# ============================== 📊 Tables & Models ==============================
with tab_tables:
    st.markdown("### Upload Database (Excel) → Build Model Stocks (FT=1 & FT=0 medians only)")
    uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl", label_visibility="collapsed")
    group_field = st.text_input("Group field (must have 0/1 values, e.g., 'FT')", value="FT", key="db_group")
    build_btn = st.button("Build model stocks (medians for FT=1 and FT=0)", use_container_width=True, key="db_build_btn")

    # ---- Build model stocks: only store medians by group; DO NOT mix uploaded rows into manual table ----
    if build_btn:
        if not uploaded:
            st.error("Please upload an Excel workbook first.")
        else:
            try:
                xls = pd.ExcelFile(uploaded)
                # choose first non-legend sheet
                sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
                sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
                raw = pd.read_excel(xls, sheet)

                # map columns
                col_group = _pick(raw, [group_field])
                if col_group is None:
                    st.error(f"Group field '{group_field}' not found in sheet '{sheet}'.")
                else:
                    df = pd.DataFrame()
                    df["GroupRaw"] = raw[col_group]

                    def add_num(df, name, src_candidates):
                        src = _pick(raw, src_candidates)
                        if src:
                            df[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                    add_num(df, "MarketCap_M$", ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap"])
                    add_num(df, "Float_M",      ["float m","public float (m)","float_m","float (m)","float m shares"])
                    add_num(df, "ShortInt_%",   ["shortint %","short interest %","short float %","si","short interest (float) %"])
                    add_num(df, "Gap_%",        ["gap %","gap%","premarket gap","gap"])
                    add_num(df, "ATR_$",        ["atr $","atr$","atr (usd)","atr"])
                    add_num(df, "RVOL",         ["rvol","relative volume","rvol @ bo"])
                    add_num(df, "PM_Vol_M",     ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)"])
                    add_num(df, "PM_$Vol_M$",   ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)"])

                    # derived
                    if {"PM_Vol_M","Float_M"}.issubset(df.columns):
                        df["FR_x"] = (df["PM_Vol_M"] / df["Float_M"]).replace([np.inf,-np.inf], np.nan)
                    if {"PM_$Vol_M$","MarketCap_M$"}.issubset(df.columns):
                        df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df["MarketCap_M$"] * 100.0).replace([np.inf,-np.inf], np.nan)

                    # reduce to binary groups FT=1 vs FT=0 (or general 1 vs 0)
                    def _to_binary(v):
                        sv = str(v).strip().lower()
                        if sv in {"1","true","yes","y","t"}: return 1
                        if sv in {"0","false","no","n","f"}: return 0
                        try:
                            fv = float(sv)
                            return 1 if fv >= 0.5 else 0
                        except:
                            return np.nan

                    df["FT01"] = df["GroupRaw"].map(_to_binary)
                    df = df[df["FT01"].isin([0,1])]
                    if df.empty or df["FT01"].nunique() < 2:
                        st.error("Could not find both FT=1 and FT=0 rows in the DB. Please check the group column.")
                    else:
                        df["Group"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

                        # medians per group
                        var_list = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%"]
                        gmed = df.groupby("Group")[var_list].median(numeric_only=True).T  # variables as rows, columns FT=0/FT=1

                        st.session_state.models = {"models_tbl": gmed, "var_list": var_list}
                        st.success(f"Built model stocks: columns in medians table = {list(gmed.columns)}")
                        do_rerun()

            except Exception as e:
                st.error("Loading/processing failed.")
                st.exception(e)

    # ---------- Manual Table (ONLY manual rows) ----------
    st.markdown("### Manual Table")
    if st.session_state.rows:
        df_rows = pd.DataFrame(st.session_state.rows)
        show_cols = [
            "Ticker",
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%",
            "Catalyst","Dilution",
            "Q_GapStruct","Q_LevelStruct","Q_Monthly",
        ]
        for c in show_cols:
            if c not in df_rows.columns:
                df_rows[c] = ""  # pad missing

        num_cols_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                        for c in show_cols if c not in ["Ticker","Q_GapStruct","Q_LevelStruct","Q_Monthly"]}
        st.dataframe(df_rows[show_cols], use_container_width=True, hide_index=True,
                     column_config={**num_cols_cfg})

        st.download_button("Download Manual Table CSV",
                           df_rows[show_cols].to_csv(index=False).encode("utf-8"),
                           "manual_table.csv","text/csv",use_container_width=True)

        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(df_rows, show_cols), language="markdown")
    else:
        st.info("No manual rows yet. Add a stock in **Manual Input**.")

    # ---------- Divergence (FT=1 vs FT=0 only) ----------
    st.markdown("### Divergence (FT=1 vs FT=0 medians)")
    models_data = st.session_state.models
    if models_data:
        models_tbl: pd.DataFrame = models_data["models_tbl"]  # rows=variables, cols=FT=0/FT=1
        needed = {"FT=1","FT=0"}.issubset(models_tbl.columns)
        if not needed:
            st.warning("Need both FT=1 and FT=0 medians from DB to show divergence.")
        else:
            diffs = models_tbl.copy()
            diffs["Δ |FT=1 − FT=0|"] = (models_tbl["FT=1"] - models_tbl["FT=0"]).abs()
            dfd = diffs.sort_values("Δ |FT=1 − FT=0|", ascending=False)
            TOP_K = st.slider("Top-K divergent variables", 3, min(12, len(dfd)), min(8, len(dfd)), 1, key="topk_div")
            show_df = dfd.head(TOP_K).reset_index().rename(columns={"index":"Variable"})

            cfg = {
                "Variable": st.column_config.TextColumn("Variable"),
                "FT=1": st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
                "FT=0": st.column_config.NumberColumn("FT=0 (median)", format="%.2f"),
                "Δ |FT=1 − FT=0|": st.column_config.NumberColumn("Δ |FT=1 − FT=0|", format="%.2f"),
            }
            st.dataframe(show_df, use_container_width=True, hide_index=True, column_config=cfg)

            st.markdown("**Markdown**")
            st.code(df_to_markdown_table(show_df, list(show_df.columns)), language="markdown")
    else:
        st.info("Upload DB and build FT=1/FT=0 medians to see divergence.")

    # ---------- Pairwise Differences: Manual vs Model Stocks ----------
    st.markdown("### Pairwise Differences (Manual Ticker vs FT=1 & FT=0 medians)")
    if st.session_state.rows and st.session_state.models and {"FT=1","FT=0"}.issubset(st.session_state.models["models_tbl"].columns):
        base = pd.DataFrame(st.session_state.rows)
        tickers = base["Ticker"].dropna().astype(str).unique().tolist() if "Ticker" in base.columns else []
        tickers = [t for t in tickers if t != ""]
        if tickers:
            tA = st.selectbox("Manual Ticker", tickers, index=0, key="pair_manual_vs_models")
            a = base[base["Ticker"] == tA].iloc[-1]

            models_tbl = st.session_state.models["models_tbl"]  # rows=variables
            num_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"]

            rows = []
            for v in num_vars:
                va = pd.to_numeric(a.get(v), errors="coerce")
                v1 = models_tbl.loc[v, "FT=1"] if (v in models_tbl.index) and ("FT=1" in models_tbl.columns) else np.nan
                v0 = models_tbl.loc[v, "FT=0"] if (v in models_tbl.index) and ("FT=0" in models_tbl.columns) else np.nan
                if pd.notna(va) and (pd.notna(v1) or pd.notna(v0)):
                    rows.append({
                        "Variable": v,
                        f"{tA}": va,
                        "FT=1": v1 if pd.notna(v1) else np.nan,
                        "FT=0": v0 if pd.notna(v0) else np.nan,
                        "Δ vs FT=1": (va - v1) if pd.notna(v1) else np.nan,
                        "Δ vs FT=0": (va - v0) if pd.notna(v0) else np.nan,
                    })
            if rows:
                dd = pd.DataFrame(rows)
                # Order by whichever absolute delta is larger per row
                dd["_max_abs"] = np.nanmax(np.vstack([
                    dd["Δ vs FT=1"].abs().fillna(-np.inf).to_numpy(),
                    dd["Δ vs FT=0"].abs().fillna(-np.inf).to_numpy()
                ]), axis=0)
                dd = dd.sort_values("_max_abs", ascending=False).drop(columns=["_max_abs"])

                cfg = {
                    "Variable": st.column_config.TextColumn("Variable"),
                    f"{tA}":   st.column_config.NumberColumn(f"{tA}",   format="%.2f"),
                    "FT=1":    st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
                    "FT=0":    st.column_config.NumberColumn("FT=0 (median)", format="%.2f"),
                    "Δ vs FT=1": st.column_config.NumberColumn("Δ vs FT=1", format="%.2f"),
                    "Δ vs FT=0": st.column_config.NumberColumn("Δ vs FT=0", format="%.2f"),
                }
                st.dataframe(dd, use_container_width=True, hide_index=True, column_config=cfg)
                st.markdown("**Markdown**")
                st.code(df_to_markdown_table(dd, list(dd.columns)), language="markdown")
            else:
                st.info("No overlapping variables between manual ticker and model medians.")
        else:
            st.info("Add at least one manual ticker to compare.")
    else:
        st.info("Need manual rows and built FT=1/FT=0 medians to run this comparison.")

    # ---------- Alignment Summary (Manual vs FT=1 & FT=0) ----------
    st.markdown("### Alignment Summary (manual stocks closest to FT=1 vs FT=0)")
    def compute_alignment_counts_vs_binary(stock_row: dict, models_tbl: pd.DataFrame) -> dict:
        if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
            return {}
        groups_cols = ["FT=1","FT=0"]
        cand_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"]
        common_vars = [v for v in cand_vars if (v in stock_row) and (v in models_tbl.index)]
        counts = {g: 0 for g in groups_cols}
        used = 0
        for v in common_vars:
            x = stock_row.get(v, None)
            try:
                xv = float(x)
            except Exception:
                xv = np.nan
            if not np.isfinite(xv):
                continue
            med = models_tbl.loc[v, groups_cols].astype(float).dropna()
            if med.empty:
                continue
            diffs = (med - xv).abs()
            nearest = diffs.idxmin()
            counts[nearest] = counts.get(nearest, 0) + 1
            used += 1
        counts["N_Vars_Used"] = used
        return counts

    if st.session_state.rows and st.session_state.models and {"FT=1","FT=0"}.issubset(st.session_state.models["models_tbl"].columns):
        models_tbl = st.session_state.models["models_tbl"]
        summary_rows = []
        for row in st.session_state.rows:
            counts = compute_alignment_counts_vs_binary(row, models_tbl)
            if not counts:
                continue
            out = {"Ticker": row.get("Ticker","—"), "N_Vars_Used": counts.pop("N_Vars_Used", 0)}
            out["Like_FT=1"] = counts.get("FT=1", 0)
            out["Like_FT=0"] = counts.get("FT=0", 0)
            if out["Like_FT=1"] > out["Like_FT=0"]:
                out["Lean"] = "FT=1"
            elif out["Like_FT=0"] > out["Like_FT=1"]:
                out["Lean"] = "FT=0"
            else:
                out["Lean"] = "Tie"
            summary_rows.append(out)

        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            cfg = {
                "Ticker": st.column_config.TextColumn("Ticker"),
                "N_Vars_Used": st.column_config.NumberColumn("N Vars Used", format="%.2f"),
                "Like_FT=1": st.column_config.NumberColumn("Like FT=1", format="%.2f"),
                "Like_FT=0": st.column_config.NumberColumn("Like FT=0", format="%.2f"),
                "Lean": st.column_config.TextColumn("Lean"),
            }
            st.dataframe(sum_df, use_container_width=True, hide_index=True, column_config=cfg)

            st.markdown("**Markdown**")
            st.code(df_to_markdown_table(sum_df, list(sum_df.columns)), language="markdown")

            st.download_button("Download Alignment CSV",
                               sum_df.to_csv(index=False).encode("utf-8"),
                               "alignment_summary.csv", "text/csv",
                               use_container_width=True)
        else:
            st.info("Add at least one manual stock to compute alignment.")
    else:
        st.info("Upload DB (to build FT=1/FT=0 medians) and add manual stocks to see alignment.")

    # Clear
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Manual Rows", use_container_width=True):
            st.session_state.rows = []
            do_rerun()
    with c2:
        if st.button("Clear Model Medians (FT=1/FT=0)", use_container_width=True):
            st.session_state.models = {}
            do_rerun()
