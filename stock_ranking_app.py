# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Helpers ==============================
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy()
    # force two decimals for floats
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

# Robust column matcher
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("%","").replace("$","").replace("(","").replace(")","").replace("  "," ").replace("’","").replace("'","")

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

# Effect size (Cliff's delta) for direction insight (not required, but cheap)
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    # sample-lite approach to keep things fast if arrays are big
    A = a[:300]
    B = b[:300]
    # pairwise compare
    gt = 0
    lt = 0
    for x in A:
        gt += np.sum(x > B)
        lt += np.sum(x < B)
    n = len(A) * len(B)
    if n == 0:
        return np.nan
    return (gt - lt) / n

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []  # user-added rows
if "last" not in st.session_state: st.session_state.last = {}  # last added row
if "models" not in st.session_state: st.session_state.models = {}  # medians per group
if "groups" not in st.session_state: st.session_state.groups = {}  # dataframes per group

# ============================== Sidebar (optional tuning) ==============================
st.sidebar.header("Display Options")
TOP_K = st.sidebar.slider("Top-K divergent variables", 3, 12, 8, 1)
SHOW_DIRECTION = st.sidebar.checkbox("Show direction vs your last input", True)

# ============================== Upload DB & Build Model Stocks ==============================
st.markdown("### Upload Database (Excel)")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], label_visibility="collapsed")
group_field = st.text_input("Group field (e.g., 'FT' or 'FT_Flag')", value="FT")

build_btn = st.button("Build model stocks (medians per group)", use_container_width=True)

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            # pick first sheet that looks like data (skip 'Legend' if present)
            sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
            sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
            raw = pd.read_excel(xls, sheet)

            # Map columns (be lenient)
            col_mc   = _pick(raw, ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap"])
            col_float= _pick(raw, ["float m","public float (m)","float_m","float (m)","float m shares"])
            col_si   = _pick(raw, ["shortint %","short interest %","short float %","si","short interest (float) %"])
            col_gap  = _pick(raw, ["gap %","gap%","premarket gap","gap"])
            col_atr  = _pick(raw, ["atr $","atr$","atr (usd)","atr"])
            col_rvol = _pick(raw, ["rvol","relative volume","rvol @ bo"])
            col_pm   = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)"])
            col_pmd  = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
            col_cat  = _pick(raw, ["catalyst","news","pr"])
            col_group= _pick(raw, [group_field])

            if col_group is None:
                st.error(f"Group field '{group_field}' not found in sheet '{sheet}'.")
            else:
                df = pd.DataFrame()
                df["Group"] = raw[col_group]

                if col_mc:   df["MarketCap_M$"] = pd.to_numeric(raw[col_mc], errors="coerce")
                if col_float:df["Float_M"]      = pd.to_numeric(raw[col_float], errors="coerce")
                if col_si:   df["ShortInt_%"]   = pd.to_numeric(raw[col_si], errors="coerce")
                if col_gap:  df["Gap_%"]        = pd.to_numeric(raw[col_gap], errors="coerce")
                if col_atr:  df["ATR_$"]        = pd.to_numeric(raw[col_atr], errors="coerce")
                if col_rvol: df["RVOL"]         = pd.to_numeric(raw[col_rvol], errors="coerce")
                if col_pm:   df["PM_Vol_M"]     = pd.to_numeric(raw[col_pm], errors="coerce")
                if col_pmd:  df["PM_$Vol_M$"]   = pd.to_numeric(raw[col_pmd], errors="coerce")

                # Catalyst -> binary
                if col_cat:
                    cat_series = raw[col_cat].astype(str).str.strip().str.lower()
                    df["Catalyst01"] = cat_series.isin(["1","true","yes","y","t"]).astype(int)

                # Derived columns (from DB rows)
                if {"PM_Vol_M","Float_M"}.issubset(df.columns):
                    df["FR_x"] = (df["PM_Vol_M"] / df["Float_M"]).replace([np.inf,-np.inf], np.nan)
                if {"PM_$Vol_M$","MarketCap_M$"}.issubset(df.columns):
                    df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df["MarketCap_M$"] * 100.0).replace([np.inf,-np.inf], np.nan)

                # Split groups & compute medians
                groups = {}
                for gname, gdf in df.groupby("Group"):
                    groups[str(gname)] = gdf.drop(columns=["Group"]).copy()

                # Model medians table
                var_list = [c for c in df.columns if c != "Group"]
                medians = []
                for gname, gdf in groups.items():
                    meds = gdf[var_list].median(numeric_only=True)
                    meds["Group"] = gname
                    medians.append(meds)
                models_tbl = pd.DataFrame(medians).set_index("Group").T

                st.session_state.groups = groups
                st.session_state.models = {"models_tbl": models_tbl, "var_list": var_list}

                st.success(f"Built model stocks from sheet '{sheet}' with groups: {list(groups.keys())}")

        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============================== Inputs + Table (no ranking/weights) ==============================
st.markdown("### Add Stock")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    with c1:
        ticker  = st.text_input("Ticker", "").strip().upper()
        mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
        si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01, format="%.2f")
        gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")  # real percent (e.g., 45 = +45%)
    with c2:
        atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
        rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")
        pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
    with c3:
        catalyst = st.selectbox("Catalyst?", ["No","Yes"])
        dilution = st.selectbox("Dilution?", ["No","Yes"])  # kept if you want to display later (not modeled from DB)

    submitted = st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
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
    st.session_state.rows.append(row)
    st.session_state.last = row
    st.success(f"Saved {ticker}.")
    do_rerun()

# Table of added stocks
st.markdown("### Current Table")
if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows)
    # show concise qualitative (Catalyst/Dilution already short)
    show_cols = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst","Dilution"]
    for c in show_cols:
        if c not in df.columns:
            df[c] = ""
    st.dataframe(
        df[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in show_cols if c not in ["Ticker","Catalyst","Dilution"]}
    )
    st.download_button("Download CSV", df[show_cols].to_csv(index=False).encode("utf-8"),
                       "table.csv","text/csv",use_container_width=True)

    st.markdown("#### 📋 Markdown view")
    st.code(df_to_markdown_table(df, show_cols), language="markdown")
else:
    st.info("No rows yet. Add a stock above.")

# ============================== Divergence: Top-K by median differences ==============================
st.markdown("### Divergence across Model Stocks (Top-K by median differences)")
models_data = st.session_state.models
groups = st.session_state.groups

def _safe_median(s: pd.Series) -> float:
    try:
        return float(pd.to_numeric(s, errors="coerce").median())
    except Exception:
        return np.nan

if models_data and groups:
    models_tbl: pd.DataFrame = models_data["models_tbl"]
    var_list: list[str] = models_data["var_list"]

    # pick up to three canonical groups if present to stabilize view order
    group_names = list(groups.keys())
    # Keep order FT=1, FT=0, MaxPush if present
    preferred = [g for g in ["1","FT=1","True","Yes","MaxPush","0","FT=0","False","No"] if g in group_names]
    seen = set(preferred)
    rest = [g for g in group_names if g not in seen]
    ordered_groups = preferred + rest
    # Reindex models table to ordered groups (columns)
    models_tbl_disp = models_tbl.copy()
    for g in ordered_groups:
        if g not in models_tbl_disp.columns:
            models_tbl_disp[g] = np.nan
    models_tbl_disp = models_tbl_disp[ordered_groups]

    # Compute max pairwise median difference per variable
    diffs = []
    for v in models_tbl_disp.index:
        vals = models_tbl_disp.loc[v].astype(float)
        if vals.notna().sum() < 2:
            continue
        # max absolute pairwise difference
        mat = vals.dropna()
        max_diff = 0.0
        best_pair = ("","")
        arr = mat.values
        cols = list(mat.index)
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                d = abs(arr[i] - arr[j])
                if d > max_diff:
                    max_diff = d
                    best_pair = (cols[i], cols[j])
        diffs.append({"Variable": v, "MaxΔMedian": max_diff, "Pair": f"{best_pair[0]} vs {best_pair[1]}"})

    if diffs:
        dfd = pd.DataFrame(diffs).sort_values("MaxΔMedian", ascending=False).head(TOP_K)
        # Build display by merging medians
        med_view = models_tbl_disp.loc[dfd["Variable"]].reset_index().rename(columns={"index":"Variable"})
        show_df = pd.merge(dfd, med_view, on="Variable", how="left")

        # Pretty display
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f") for c in show_df.columns if c not in ["Variable","Pair"]}
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

# ============================== Manual Input vs Model Stocks (direction) ==============================
if SHOW_DIRECTION and models_data and groups and isinstance(st.session_state.last, dict) and st.session_state.last:
    st.markdown("### Manual Input vs Model Stocks")
    last = st.session_state.last
    # Compute your row in the same variable space
    your_row = {
        "MarketCap_M$": float(last.get("MarketCap_M$", np.nan)),
        "Float_M": float(last.get("Float_M", np.nan)),
        "ShortInt_%": float(last.get("ShortInt_%", np.nan)),
        "Gap_%": float(last.get("Gap_%", np.nan)),
        "ATR_$": float(last.get("ATR_$", np.nan)),
        "RVOL": float(last.get("RVOL", np.nan)),
        "PM_Vol_M": float(last.get("PM_Vol_M", np.nan)),
        "PM_$Vol_M$": float(last.get("PM_$Vol_M$", np.nan)),
        "FR_x": float(last.get("FR_x", np.nan)),
        "PM$Vol/MC_%": float(last.get("PM$Vol/MC_%", np.nan)),
        # Catalyst handled as text for display; not used in numeric direction
    }
    models_tbl: pd.DataFrame = models_data["models_tbl"]
    # Limit to vars present in models table
    common_vars = [v for v in your_row.keys() if v in models_tbl.index]
    comp_rows = []
    # Choose two anchor groups if available for direction: try FT=1 and FT=0 else first two
    gnames = list(st.session_state.groups.keys())
    gA = "FT=1" if "FT=1" in gnames else ("1" if "1" in gnames else (gnames[0] if gnames else None))
    gB = "FT=0" if "FT=0" in gnames else ("0" if "0" in gnames else (gnames[1] if len(gnames)>1 else None))
    for v in common_vars:
        your_v = your_row[v]
        medA = float(models_tbl.at[v, gA]) if (gA in models_tbl.columns and v in models_tbl.index) else np.nan
        medB = float(models_tbl.at[v, gB]) if (gB in models_tbl.columns and v in models_tbl.index) else np.nan
        # direction label
        if np.isfinite(your_v) and np.isfinite(medA) and np.isfinite(medB):
            # distance vs anchors
            dA = abs(your_v - medA)
            dB = abs(your_v - medB)
            if dA < dB:
                direction = f"→ {gA}-like"
            elif dB < dA:
                direction = f"→ {gB}-like"
            else:
                direction = "→ In-between"
        else:
            direction = "—"
        row = {"Variable": v, "Your Value": your_v, f"{gA} Median": medA, f"{gB} Median": medB, "Direction": direction}
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f") for c in comp_df.columns if c not in ["Variable","Direction"]}
    col_cfg = {"Variable": st.column_config.TextColumn("Variable"),
               "Direction": st.column_config.TextColumn("Direction"), **num_cfg}
    st.dataframe(comp_df, use_container_width=True, hide_index=True, column_config=col_cfg)

    st.markdown("**Markdown**")
    st.code(df_to_markdown_table(comp_df, list(comp_df.columns)), language="markdown")
