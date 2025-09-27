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
    # ----------- (Moved here) Upload Database — no group field, fixed to 'FT' -----------
    st.markdown("### Upload Database (Excel) → Build Model Stocks (FT=1 & FT=0 medians only)")
    uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl", label_visibility="collapsed")
    build_btn = st.button("Build model stocks (medians for FT=1 and FT=0)", use_container_width=True, key="db_build_btn")

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

                group_field = "FT"  # fixed (no input)

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

    # ----------- Manual Add Form -----------
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

# ============================== 📊 Tables & Models ==============================
with tab_tables:
    # ---------- Manual Table (editable; ONLY manual rows) ----------
    st.markdown("### Manual Table (editable)")
    
    if st.session_state.rows:
        df_rows = pd.DataFrame(st.session_state.rows)
        show_cols = [
            "Ticker",
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%",
            "Catalyst","Dilution",
            "Q_GapStruct","Q_LevelStruct","Q_Monthly",
        ]
        # pad missing columns so editor stays stable
        for c in show_cols:
            if c not in df_rows.columns:
                df_rows[c] = ""
    
        # column configs
        num_cols_cfg = {
            c: st.column_config.NumberColumn(c, format="%.2f")
            for c in show_cols
            if c not in ["Ticker","Q_GapStruct","Q_LevelStruct","Q_Monthly"]
        }
        text_cols_cfg = {
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Q_GapStruct": st.column_config.TextColumn("Q_GapStruct"),
            "Q_LevelStruct": st.column_config.TextColumn("Q_LevelStruct"),
            "Q_Monthly": st.column_config.TextColumn("Q_Monthly"),
        }
    
        edited_df = st.data_editor(
            df_rows[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={**num_cols_cfg, **text_cols_cfg},
            num_rows="dynamic",  # allows adding/removing rows
            key="manual_editor",
        )
    
        c_save, c_dl = st.columns([1,1])
        with c_save:
            if st.button("Save Changes", use_container_width=True, key="save_manual_changes"):
                st.session_state.rows = edited_df.to_dict(orient="records")
                st.success("Manual table updated.")
                do_rerun()
    
        with c_dl:
            st.download_button(
                "Download Manual Table CSV",
                edited_df.to_csv(index=False).encode("utf-8"),
                "manual_table.csv",
                "text/csv",
                use_container_width=True,
            )
    
        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(edited_df, show_cols), language="markdown")
    
    else:
        st.info("No manual rows yet. Add a stock in **Manual Input**.")

    # ---------- Divergence (FT=1 vs FT=0 only) ----------
    with st.expander("Divergence (FT=1 vs FT=0 medians)", expanded=False):
        models_data = st.session_state.models
        if models_data:
            models_tbl: pd.DataFrame = models_data["models_tbl"]  # rows=variables, cols=FT=0/FT=1
            needed = {"FT=1","FT=0"}.issubset(models_tbl.columns)
            if not needed:
                st.warning("Need both FT=1 and FT=0 medians from DB to show divergence.")
            else:
                diffs = models_tbl.copy()
                diffs["Δ |FT=1 − FT=0|"] = (models_tbl["FT=1"] - models_tbl["FT=0"])
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
    with st.expander("Pairwise Differences (Manual Ticker vs FT=1 & FT=0 medians)", expanded=False):
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

# ---------- Alignment Summary (DataTables child-rows) ----------
st.markdown("### Alignment Summary (model medians + manual stocks)")

import json
import streamlit.components.v1 as components

def _compute_alignment_counts(stock_row: dict, models_tbl: pd.DataFrame) -> dict:
    if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
        return {}
    groups = ["FT=1","FT=0"]
    cand_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                 "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"]
    common = [v for v in cand_vars if (v in stock_row) and (v in models_tbl.index)]
    counts = {g: 0 for g in groups}; used = 0
    for v in common:
        xv = pd.to_numeric(stock_row.get(v), errors="coerce")
        if not np.isfinite(xv): 
            continue
        med = models_tbl.loc[v, groups].astype(float).dropna()
        if med.empty: 
            continue
        diffs = (med - xv).abs()
        nearest = diffs.idxmin()
        counts[nearest] += 1
        used += 1
    counts["N_Vars_Used"] = used
    return counts

models_data = st.session_state.get("models")
if models_data and isinstance(models_data, dict):
    models_tbl: pd.DataFrame = models_data.get("models_tbl", pd.DataFrame())
else:
    models_tbl = pd.DataFrame()

if not models_tbl.empty and {"FT=1","FT=0"}.issubset(models_tbl.columns):
    # two model “stocks”
    model_rows = []
    for g in ["FT=1","FT=0"]:
        vals = models_tbl[g].to_dict()
        vals.update({"Ticker": g})
        model_rows.append(vals)

    # manual rows (if any)
    manual_rows = st.session_state.get("rows", [])
    all_rows = model_rows + manual_rows

    # variables shown in details
    num_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"]

    summary_rows = []
    detail_map = {}

    for row in all_rows:
        stock = dict(row)
        tkr = stock.get("Ticker") or "—"
        counts = _compute_alignment_counts(stock, models_tbl)
        if not counts: 
            continue
        like1, like0 = counts.get("FT=1",0), counts.get("FT=0",0)
        n_used = counts.get("N_Vars_Used",0)

        ft1_pct = round((like1 / n_used * 100.0), 2) if n_used>0 else 0.0
        ft0_pct = round((like0 / n_used * 100.0), 2) if n_used>0 else 0.0
        lean01  = ((like1 - like0) / n_used + 1) / 2.0 if n_used>0 else 0.5
        lean_pct = round(lean01 * 100.0, 2)
        lean_lbl = "FT=1" if like1>like0 else "FT=0" if like0>like1 else "Tie"

        summary_rows.append({
            "Ticker": tkr,
            "FT1_pct": ft1_pct,
            "FT0_pct": ft0_pct,
            "Lean_pct": lean_pct,
            "Lean": lean_lbl,
        })

        # child table: per-variable values and deltas
        drows = []
        for v in num_vars:
            va = pd.to_numeric(stock.get(v), errors="coerce")
            v1 = models_tbl.loc[v, "FT=1"] if (v in models_tbl.index) else np.nan
            v0 = models_tbl.loc[v, "FT=0"] if (v in models_tbl.index) else np.nan
            if pd.notna(va) or pd.notna(v1) or pd.notna(v0):
                drows.append({
                    "Variable": v,
                    "Value": None if pd.isna(va) else float(va),
                    "FT1":   None if pd.isna(v1) else float(v1),
                    "FT0":   None if pd.isna(v0) else float(v0),
                    "d_vs_FT1": None if (pd.isna(va) or pd.isna(v1)) else float(va - v1),
                    "d_vs_FT0": None if (pd.isna(va) or pd.isna(v0)) else float(va - v0),
                })
        detail_map[tkr] = drows

    if summary_rows:
        payload = {"rows": summary_rows, "details": detail_map}
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }}
  table.dataTable tbody tr {{ cursor: pointer; }}
  .bar {{ height: 16px; border-radius: 10px; background: #eee; position: relative; overflow: hidden; }}
  .bar > span {{ position: absolute; left: 0; top: 0; bottom: 0; width: 0%; background: linear-gradient(90deg, #ef4444, #10b981); }}
  .bar-label {{ font-size: 11px; margin-left: 8px; white-space: nowrap; color:#374151; }}
  .bar-wrap {{ display:flex; align-items:center; gap:8px; }}
  .child-table {{ width: 100%; border-collapse: collapse; margin: 6px 0 4px 32px; }}
  .child-table th, .child-table td {{ font-size: 12px; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; text-align:right; }}
  .child-table th:first-child, .child-table td:first-child {{ text-align:left; }}
  .pos {{ color:#059669; }} .neg {{ color:#dc2626; }}
</style>
</head>
<body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead>
      <tr>
        <th>Ticker</th>
        <th>FT=1 %</th>
        <th>FT=0 %</th>
        <th>Lean (0=FT0 • 50=tie • 100=FT=1)</th>
        <th>Lean</th>
      </tr>
    </thead>
  </table>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = {json.dumps(payload)};

    function barCell(pct) {{
      const v = (pct==null||isNaN(pct)) ? 0 : Math.max(0, Math.min(100, pct));
      return `
        <div class="bar-wrap">
          <div class="bar" style="width:140px"><span style="width:${{v}}%"></span></div>
          <div class="bar-label">${{v.toFixed(0)}}%</div>
        </div>`;
    }}
    function leanBarCell(pct) {{
      const v = (pct==null||isNaN(pct)) ? 50 : Math.max(0, Math.min(100, pct));
      return `
        <div class="bar-wrap">
          <div class="bar" style="width:220px; background: linear-gradient(90deg,#fecaca,#e5e7eb 50%,#bbf7d0);">
            <span style="width:${{v}}%"></span>
          </div>
          <div class="bar-label">${{v.toFixed(0)}}%</div>
        </div>`;
    }}

    function childTableHTML(ticker) {{
      const rows = data.details[ticker] || [];
      if (!rows.length) return '<div style="margin-left:32px;color:#6b7280;">No variable overlaps for this stock.</div>';
      const cells = rows.map(r => {{
        const v  = (r.Value==null||isNaN(r.Value)) ? '' : r.Value.toFixed(2);
        const f1 = (r.FT1==null ||isNaN(r.FT1))  ? '' : r.FT1.toFixed(2);
        const f0 = (r.FT0==null ||isNaN(r.FT0))  ? '' : r.FT0.toFixed(2);
        const d1 = (r.d_vs_FT1==null||isNaN(r.d_vs_FT1)) ? '' : r.d_vs_FT1.toFixed(2);
        const d0 = (r.d_vs_FT0==null||isNaN(r.d_vs_FT0)) ? '' : r.d_vs_FT0.toFixed(2);
        const c1 = (!d1)? '' : (parseFloat(d1)>=0 ? 'pos' : 'neg');
        const c0 = (!d0)? '' : (parseFloat(d0)>=0 ? 'pos' : 'neg');
        return `
          <tr>
            <td>${{r.Variable}}</td>
            <td>${{v}}</td>
            <td>${{f1}}</td>
            <td>${{f0}}</td>
            <td class="${{c1}}">${{d1}}</td>
            <td class="${{c0}}">${{d0}}</td>
          </tr>`;
      }}).join('');
      return `
        <table class="child-table">
          <thead>
            <tr>
              <th>Variable</th><th>Value</th><th>FT=1 median</th><th>FT=0 median</th>
              <th>Δ vs FT=1</th><th>Δ vs FT=0</th>
            </tr>
          </thead>
          <tbody>${{cells}}</tbody>
        </table>`;
    }}

    $(function() {{
      const table = $('#align').DataTable({{
        data: data.rows,
        responsive: true,
        paging: false, info: false, searching: false,
        order: [[0,'asc']],
        columns: [
          {{ data: 'Ticker' }},
          {{ data: 'FT1_pct',  render: (d)=>barCell(d) }},
          {{ data: 'FT0_pct',  render: (d)=>barCell(d) }},
          {{ data: 'Lean_pct', render: (d)=>leanBarCell(d) }},
          {{ data: 'Lean' }},
        ]
      }});

      // whole-row toggle child
      $('#align tbody').on('click', 'tr', function () {{
        const row = table.row(this);
        if (row.child.isShown()) {{
          row.child.hide(); $(this).removeClass('shown');
        }} else {{
          const ticker = row.data().Ticker;
          row.child(childTableHTML(ticker)).show(); $(this).addClass('shown');
        }}
      }});
    }});
  </script>
</body>
</html>
        """
        components.html(html, height=600, scrolling=True)
    else:
        st.info("No eligible rows yet. Add manual stocks and/or ensure FT=1/FT=0 medians are built.")
else:
    st.info("Upload DB and click **Build model stocks** (to compute FT=1/FT=0 medians) first.")

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
