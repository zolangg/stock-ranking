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
if "rows" not in st.session_state: st.session_state.rows = []      # manual + uploaded rows for table
if "last" not in st.session_state: st.session_state.last = {}      # last manual row
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl": DataFrame, "var_list": [...]}
if "groups" not in st.session_state: st.session_state.groups = {}  # group -> df (uploaded)

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
            # Sliders per your request
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
                # store 1-based index
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
            # Qual raw selections (1..7) if needed later
            "_Q_GapStruct_idx": qual_selected.get("GapStruct", 1),
            "_Q_LevelStruct_idx": qual_selected.get("LevelStruct", 1),
            "_Q_Monthly_idx": qual_selected.get("Monthly", 1),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker}.")
        do_rerun()

    # Minimal preview of last added (no ranking)
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
    st.markdown("### Upload Database (Excel) → Build Model Stocks")
    uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl", label_visibility="collapsed")
    group_field = st.text_input("Group field (e.g., 'FT')", value="FT", key="db_group")
    build_btn = st.button("Build model stocks (medians per group)", use_container_width=True, key="db_build_btn")

    if build_btn:
        if not uploaded:
            st.error("Please upload an Excel workbook first.")
        else:
            try:
                xls = pd.ExcelFile(uploaded)
                # Pick first non-legend sheet if present
                sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
                sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
                raw = pd.read_excel(xls, sheet)

                # ------- Map columns from DB -------
                col_ticker= _pick(raw, ["ticker","symbol"])
                col_group = _pick(raw, [group_field])

                col_mc    = _pick(raw, ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap"])
                col_float = _pick(raw, ["float m","public float (m)","float_m","float (m)","float m shares"])
                col_si    = _pick(raw, ["shortint %","short interest %","short float %","si","short interest (float) %"])
                col_gap   = _pick(raw, ["gap %","gap%","premarket gap","gap"])
                col_atr   = _pick(raw, ["atr $","atr$","atr (usd)","atr"])
                col_rvol  = _pick(raw, ["rvol","relative volume","rvol @ bo"])
                col_pm    = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)"])
                col_pmd   = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)"])
                col_cat   = _pick(raw, ["catalyst","news","pr"])

                if col_group is None:
                    st.error(f"Group field '{group_field}' not found in sheet '{sheet}'.")
                else:
                    df = pd.DataFrame()
                    df["Group"] = raw[col_group].astype(str)

                    if col_ticker: df["Ticker"] = raw[col_ticker].astype(str).str.upper()

                    def add_num(name, src):
                        if src:
                            df[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                    add_num("MarketCap_M$", col_mc)
                    add_num("Float_M",      col_float)
                    add_num("ShortInt_%",   col_si)
                    add_num("Gap_%",        col_gap)     # real % (e.g., 100 = +100)
                    add_num("ATR_$",        col_atr)
                    add_num("RVOL",         col_rvol)
                    add_num("PM_Vol_M",     col_pm)
                    add_num("PM_$Vol_M$",   col_pmd)

                    # Catalyst in DB is Yes/No or 1/0 → map to 0/1 for models if present
                    if col_cat:
                        s = raw[col_cat].astype(str).str.strip().str.lower()
                        df["Catalyst01"] = s.map({
                            "yes":1,"y":1,"true":1,"1":1,
                            "no":0,"n":0,"false":0,"0":0
                        })
                        # if numeric strings:
                        df["Catalyst01"] = df["Catalyst01"].fillna(pd.to_numeric(s, errors="coerce")).fillna(0).astype(int)

                    # Derived
                    if {"PM_Vol_M","Float_M"}.issubset(df.columns):
                        df["FR_x"] = (df["PM_Vol_M"] / df["Float_M"]).replace([np.inf,-np.inf], np.nan)
                    if {"PM_$Vol_M$","MarketCap_M$"}.issubset(df.columns):
                        df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df["MarketCap_M$"] * 100.0).replace([np.inf,-np.inf], np.nan)

                    # Split groups & build medians table
                    groups = {}
                    for gname, gdf in df.groupby("Group"):
                        groups[str(gname)] = gdf.drop(columns=["Group"]).copy()

                    var_list = [c for c in df.columns if c not in ["Group","Ticker"]]
                    medians = []
                    for gname, gdf in groups.items():
                        meds = gdf[var_list].median(numeric_only=True)
                        meds["Group"] = gname
                        medians.append(meds)
                    models_tbl = pd.DataFrame(medians).set_index("Group").T  # variables as rows

                    st.session_state.groups = groups
                    st.session_state.models = {"models_tbl": models_tbl, "var_list": var_list}

                    # Also append uploaded rows to main table (display)
                    st.session_state.rows.extend(df.to_dict(orient="records"))

                    st.success(f"Built model stocks from sheet '{sheet}' with groups: {list(groups.keys())}")
                    do_rerun()

            except Exception as e:
                st.error("Loading/processing failed.")
                st.exception(e)

    # ---------- Current Manual + Uploaded Table ----------
    st.markdown("### Current Table (Manual + Uploaded)")
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

        st.download_button("Download Table CSV",
                           df_rows[show_cols].to_csv(index=False).encode("utf-8"),
                           "stocks_table.csv","text/csv",use_container_width=True)

        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(df_rows, show_cols), language="markdown")
    else:
        st.info("No rows yet. Add a stock in **Manual Input** or upload a DB above.")

    # ---------- Divergence across Model Stocks ----------
    st.markdown("### Divergence across Model Stocks (Top-K by median differences)")
    TOP_K = st.slider("Top-K divergent variables", 3, 12, 8, 1, key="topk_div")
    models_data = st.session_state.models
    groups = st.session_state.groups

    if models_data and groups:
        models_tbl: pd.DataFrame = models_data["models_tbl"]

        group_names = list(models_tbl.columns)
        preferred = [g for g in ["FT=1","1","True","Yes","MaxPush","FT=0","0","False","No"] if g in group_names]
        rest = [g for g in group_names if g not in preferred]
        ordered_groups = preferred + rest
        models_tbl_disp = models_tbl.reindex(columns=ordered_groups)

        # Compute max absolute pairwise median difference per variable
        diffs = []
        for v in models_tbl_disp.index:
            vals = models_tbl_disp.loc[v].astype(float)
            if vals.notna().sum() < 2:
                continue
            mat = vals.dropna()
            arr = mat.values
            cols = list(mat.index)
            max_diff = 0.0
            best_pair = ("","")
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    d = abs(arr[i] - arr[j])
                    if d > max_diff:
                        max_diff = d
                        best_pair = (cols[i], cols[j])
            diffs.append({"Variable": v, "MaxΔMedian": max_diff, "Pair": f"{best_pair[0]} vs {best_pair[1]}"})

        if diffs:
            dfd = pd.DataFrame(diffs).sort_values("MaxΔMedian", ascending=False).head(TOP_K)
            med_view = models_tbl_disp.loc[dfd["Variable"]].reset_index().rename(columns={"index":"Variable"})
            show_df = pd.merge(dfd, med_view, on="Variable", how="left")

            num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                       for c in show_df.columns if c not in ["Variable","Pair"]}
            col_cfg = {"Variable": st.column_config.TextColumn("Variable"),
                       "Pair": st.column_config.TextColumn("Strongest Pair"), **num_cfg}
            st.dataframe(show_df, use_container_width=True, hide_index=True, column_config=col_cfg)
            st.caption("Top-K variables with the largest absolute median differences across model groups.")
            st.markdown("**Markdown**")
            st.code(df_to_markdown_table(show_df, list(show_df.columns)), language="markdown")
        else:
            st.info("Not enough overlap across groups to compute divergences.")
    else:
        st.info("Upload the DB and build model stocks to see divergences.")

    # ---------- Pairwise Differences (select two uploaded tickers if present) ----------
    st.markdown("### Pairwise Differences (select two tickers)")
    # Use uploaded rows (those with 'Ticker') for pairwise compare
    if st.session_state.rows:
        base = pd.DataFrame(st.session_state.rows)
        if "Ticker" in base.columns and base["Ticker"].notna().any():
            tickers = base["Ticker"].dropna().astype(str).unique().tolist()
            tickers = [t for t in tickers if t != ""]
            if len(tickers) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    tA = st.selectbox("Ticker A", tickers, index=0, key="pair_A")
                with c2:
                    tB = st.selectbox("Ticker B", [t for t in tickers if t != tA], index=0, key="pair_B")

                # Pick most recent rows for each
                a = base[base["Ticker"] == tA].iloc[-1]
                b = base[base["Ticker"] == tB].iloc[-1]

                num_vars = [
                    "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                    "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"
                ]
                rows = []
                for v in num_vars:
                    va = pd.to_numeric(a.get(v), errors="coerce")
                    vb = pd.to_numeric(b.get(v), errors="coerce")
                    if pd.notna(va) and pd.notna(vb):
                        diff = float(va - vb)
                        rows.append({"Variable": v, f"{tA}": va, f"{tB}": vb, "Δ(A−B)": diff, "|Δ|": abs(diff)})
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
        else:
            st.info("No tickers available for pairwise comparison.")
    else:
        st.info("No data available for pairwise comparison.")

    # ---------- Alignment Summary (manual rows vs model medians) ----------
    st.markdown("### Alignment Summary (how many variables look like each model group)")
    def compute_alignment_counts(stock_row: dict, models_tbl: pd.DataFrame) -> dict:
        if models_tbl is None or models_tbl.empty:
            return {}
        groups_cols = list(models_tbl.columns)
        cand_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%"]
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
            med = models_tbl.loc[v].astype(float).dropna()
            if med.empty:
                continue
            diffs = (med - xv).abs()
            nearest = diffs.idxmin()
            counts[nearest] = counts.get(nearest, 0) + 1
            used += 1
        counts["N_Vars_Used"] = used
        return counts

    if st.session_state.rows and st.session_state.models:
        models_tbl: pd.DataFrame = st.session_state.models["models_tbl"]
        group_cols = list(models_tbl.columns)

        summary_rows = []
        for row in st.session_state.rows:
            counts = compute_alignment_counts(row, models_tbl)
            if not counts:
                continue
            out = {"Ticker": row.get("Ticker","—"), "N_Vars_Used": counts.pop("N_Vars_Used", 0)}
            for g in group_cols:
                out[f"Like_{g}"] = counts.get(g, 0)
            pairs = [(g, out.get(f"Like_{g}", 0)) for g in group_cols]
            if pairs:
                m = max(c for _, c in pairs)
                top = [g for g, c in pairs if c == m and m > 0]
                out["Lean"] = ", ".join(top) if top else "—"
            else:
                out["Lean"] = "—"
            summary_rows.append(out)

        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                       for c in sum_df.columns if c not in ["Ticker","Lean"]}
            col_cfg = {"Ticker": st.column_config.TextColumn("Ticker"),
                       "Lean": st.column_config.TextColumn("Lean"), **num_cfg}
            st.dataframe(sum_df, use_container_width=True, hide_index=True, column_config=col_cfg)

            st.markdown("**Markdown**")
            st.code(df_to_markdown_table(sum_df, list(sum_df.columns)), language="markdown")

            st.download_button("Download Alignment CSV",
                               sum_df.to_csv(index=False).encode("utf-8"),
                               "alignment_summary.csv", "text/csv",
                               use_container_width=True)
        else:
            st.info("Add at least one stock and build model stocks to compute alignment.")
    else:
        st.info("To compute alignment, add stocks and build model stocks first.")

    # Clear all
    st.markdown("---")
    if st.button("Clear All Rows", use_container_width=True):
        st.session_state.rows = []
        st.session_state.models = {}
        st.session_state.groups = {}
        do_rerun()
