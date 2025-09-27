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

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []      # manual rows ONLY
if "last" not in st.session_state: st.session_state.last = {}      # last manual row
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl": DataFrame, "var_list": [...]}

# ============================== Upload DB → Build Medians ==============================
st.subheader("Upload Database → Build Model Stocks (FT=1 and FT=0 medians)")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
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

            # map columns
            col_group = _pick(raw, ["FT"])
            if col_group is None:
                st.error(f"Group field 'FT' not found in sheet '{sheet}'. It must have 0/1 values.")
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
                    st.error("Could not find both FT=1 and FT=0 rows in the DB. Please check the FT column.")
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

# Show medians table right after the button (when available)
models_data = st.session_state.models
if models_data and isinstance(models_data, dict) and not models_data.get("models_tbl", pd.DataFrame()).empty:
    st.markdown("#### Model Medians (FT=1 vs FT=0)")
    med_tbl: pd.DataFrame = models_data["models_tbl"]
    cfg = {
        "FT=1": st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
        "FT=0": st.column_config.NumberColumn("FT=0 (median)", format="%.2f"),
    }
    st.dataframe(med_tbl, use_container_width=True, column_config=cfg, hide_index=False)

# ============================== ➕ Manual Input (simplified) ==============================
st.markdown("---")
st.subheader("Add Stock")

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
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)

    st.form_submit_button.disabled = False
    submitted = st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
    # Derived metrics
    fr = (pm_vol / float_m) if float_m > 0 else 0.0
    pmmc = (pm_dol / mc_m * 100.0) if mc_m > 0 else 0.0

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
        # Catalyst as 1/0 plus label
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "CatalystYN": catalyst_yn,
    }
    st.session_state.rows.append(row)
    st.session_state.last = row
    st.success(f"Saved {ticker}.")
    do_rerun()

# ============================== Added Stocks (editable) ==============================
st.markdown("### Added Stocks (editable)")
if st.session_state.rows:
    df_rows = pd.DataFrame(st.session_state.rows)
    show_cols = [
        "Ticker",
        "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
        "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%",
        "CatalystYN","Catalyst",
    ]
    for c in show_cols:
        if c not in df_rows.columns:
            df_rows[c] = ""

    num_cols_cfg = {
        c: st.column_config.NumberColumn(c, format="%.2f")
        for c in show_cols
        if c not in ["Ticker","CatalystYN"]
    }
    text_cols_cfg = {
        "Ticker": st.column_config.TextColumn("Ticker"),
        "CatalystYN": st.column_config.SelectboxColumn("Catalyst?",
                                                       options=["No","Yes"])
    }

    edited_df = st.data_editor(
        df_rows[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={**num_cols_cfg, **text_cols_cfg},
        num_rows="dynamic",
        key="manual_editor",
    )

    # Convert CatalystYN back to numeric Catalyst consistently
    edited_df["Catalyst"] = edited_df["CatalystYN"].map({"Yes":1.0,"No":0.0}).fillna(0.0)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Changes", use_container_width=True, key="save_manual_changes"):
            st.session_state.rows = edited_df.to_dict(orient="records")
            st.success("Manual table updated.")
            do_rerun()
    with c2:
        st.download_button(
            "Download Added Stocks CSV",
            edited_df.to_csv(index=False).encode("utf-8"),
            "added_stocks.csv", "text/csv",
            use_container_width=True,
        )
else:
    st.info("No manual rows yet. Add a stock above.")

# ============================== Alignment (only added stocks; FT=1 / FT=0 bars) ==============================
st.markdown("### Alignment (only added stocks)")

def compute_alignment_counts_vs_binary(stock_row: dict, models_tbl: pd.DataFrame) -> dict:
    if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
        return {}
    groups_cols = ["FT=1","FT=0"]
    cand_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                 "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst"]
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

models_tbl = (st.session_state.get("models") or {}).get("models_tbl", pd.DataFrame())
if st.session_state.rows and not models_tbl.empty and {"FT=1","FT=0"}.issubset(models_tbl.columns):
    # Build alignment summary ONLY for added stocks (no model rows)
    summary_rows = []
    for row in st.session_state.rows:
        counts = compute_alignment_counts_vs_binary(row, models_tbl)
        if not counts: 
            continue
        like1, like0 = counts.get("FT=1", 0), counts.get("FT=0", 0)
        n_used = counts.get("N_Vars_Used", 0)
        ft1_val = (like1 / n_used) if n_used > 0 else 0.0
        ft0_val = (like0 / n_used) if n_used > 0 else 0.0
        summary_rows.append({
            "Ticker": row.get("Ticker","—"),
            "FT1": ft1_val,   # 0..1 for ProgressColumn
            "FT0": ft0_val,   # 0..1 for ProgressColumn
        })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        # Order columns to feel centered visually: Ticker | (spacer via narrow col?) Not possible directly,
        # but we can at least put FT=1 and FT=0 beside each other and keep ticker narrow.
        cfg = {
            "Ticker": st.column_config.TextColumn("Ticker"),
            "FT1": st.column_config.ProgressColumn("FT=1", format="%.0f%%", min_value=0.0, max_value=1.0, help="Share of variables closer to FT=1"),
            "FT0": st.column_config.ProgressColumn("FT=0", format="%.0f%%", min_value=0.0, max_value=1.0, help="Share of variables closer to FT=0"),
        }
        # Small trick to visually center: use empty text column left & right (narrow). Commented out by default.
        st.data_editor(
            sum_df[["Ticker","FT1","FT0"]],
            hide_index=True,
            use_container_width=True,
            column_config=cfg,
            disabled=True,
            key="align_only_added",
        )

        # Markdown copy if needed
        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(sum_df, list(sum_df.columns)), language="markdown")
    else:
        st.info("No overlapping variables between added stocks and model medians yet.")
elif st.session_state.rows and (models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns)):
    st.info("Upload DB and build FT=1/FT=0 medians to compute alignment.")
else:
    st.info("Add at least one stock above to compute alignment.")

# ============================== Clear ==============================
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    if st.button("Clear Added Stocks", use_container_width=True):
        st.session_state.rows = []
        do_rerun()
with c2:
    if st.button("Clear Model Medians (FT=1/FT=0)", use_container_width=True):
        st.session_state.models = {}
        do_rerun()
