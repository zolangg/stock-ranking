# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json

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
    cols_lc = {c: c.strip().lower() for c in cols}

    # 1) RAW case-insensitive exact match first (preserves '$' distinction)
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc:
                return c

    # 2) Normalized exact match
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c

    # 3) Normalized 'contains' fallback
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
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl": DataFrame, "mad_tbl": DataFrame, "var_list": [...]}

# ============================== Upload DB → Build Medians ==============================
st.subheader("Upload Database")

uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

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

            # --- auto-detect group column ---
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
                # fallback: look for binary column
                for c in raw.columns:
                    vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group = c
                        break

            if col_group is None:
                st.error("Could not detect FT column (0/1). Please ensure your sheet has an FT or binary column.")
            else:
                df = pd.DataFrame()
                df["GroupRaw"] = raw[col_group]

                def add_num(df, name, src_candidates):
                    src = _pick(raw, src_candidates)
                    if src:
                        df[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                add_num(df, "MarketCap_M$", ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap"])
                add_num(df, "Float_M",      ["float m","public float (m)","float_m","float (m)","float m shares"])
                add_num(df, "ShortInt_%",   ["shortint %","short interest %","short float %","si","short interest (float) %","SI"])
                # Scale fractional short interest (e.g., 0.06 -> 6.0), keep already-in-% as-is
                if "ShortInt_%" in df.columns:
                    s_si = pd.to_numeric(df["ShortInt_%"], errors="coerce")
                    df["ShortInt_%"] = np.where(s_si.notna() & (s_si.abs() <= 2), s_si * 100.0, s_si)
                add_num(df, "Gap_%",        ["gap %","gap%","premarket gap","gap"])
                # Scale fractional gaps (e.g., 0.90 -> 90.0), keep already-in-% as-is
                if "Gap_%" in df.columns:
                    s = pd.to_numeric(df["Gap_%"], errors="coerce")
                    df["Gap_%"] = np.where(s.notna() & (s.abs() <= 2), s * 100.0, s)
                add_num(df, "ATR_$",        ["atr $","atr$","atr (usd)","atr"])
                add_num(df, "RVOL",         ["rvol","relative volume","rvol @ bo"])
                add_num(df, "PM_Vol_M",     ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)"])
                add_num(df, "PM_$Vol_M$",   ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)"])

                # --- Catalyst (binary: Yes/No/1/0/True/False) from DB, if present ---
                cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
                def _to_binary_local(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1.0
                    if sv in {"0","false","no","n","f"}: return 0.0
                    try:
                        fv = float(sv)
                        return 1.0 if fv >= 0.5 else 0.0
                    except:
                        return np.nan
                if cand_catalyst:
                    df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

                # derived
                if {"PM_Vol_M","Float_M"}.issubset(df.columns):
                    df["FR_x"] = (df["PM_Vol_M"] / df["Float_M"]).replace([np.inf,-np.inf], np.nan)
                if {"PM_$Vol_M$","MarketCap_M$"}.issubset(df.columns):
                    df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df["MarketCap_M$"] * 100.0).replace([np.inf,-np.inf], np.nan)

                # normalize to binary for FT groups
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
                    var_list = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                                "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst"]
                    gmed = df.groupby("Group")[var_list].median(numeric_only=True).T  # rows=variables, cols=FT=0/FT=1

                    # --- robust spread: MAD (median absolute deviation) per group ---
                    def _mad(series: pd.Series) -> float:
                        s = pd.to_numeric(series, errors="coerce").dropna()
                        if s.empty:
                            return np.nan
                        med = float(np.median(s))
                        return float(np.median(np.abs(s - med)))

                    gmads = df.groupby("Group")[var_list].apply(lambda g: g.apply(_mad)).T  # same shape as gmed

                    # store both
                    st.session_state.models = {"models_tbl": gmed, "mad_tbl": gmads, "var_list": var_list}
                    st.success(f"Built model stocks: columns in medians table = {list(gmed.columns)}")
                    do_rerun()

        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# Show medians table INSIDE AN EXPANDER (when available)
models_data = st.session_state.models
if models_data and isinstance(models_data, dict) and not models_data.get("models_tbl", pd.DataFrame()).empty:
    with st.expander("Model Medians (FT=1 vs FT=0)", expanded=False):
        med_tbl: pd.DataFrame = models_data["models_tbl"]
        mad_tbl: pd.DataFrame = models_data.get("mad_tbl", pd.DataFrame())

        # User control for what we call "significant"
        sig_thresh = st.slider("Significance threshold (σ)", 0.0, 5.0, 5.0, 0.1,
                               help="Highlight rows where |FT=1 − FT=0| / (MAD₁ + MAD₀) ≥ σ")
        st.session_state["sig_thresh"] = float(sig_thresh)

        if not mad_tbl.empty and {"FT=1","FT=0"}.issubset(mad_tbl.columns):
            eps = 1e-9  # avoid /0
            diff = (med_tbl["FT=1"] - med_tbl["FT=0"]).abs()
            spread = (mad_tbl["FT=1"].fillna(0.0) + mad_tbl["FT=0"].fillna(0.0))
            sig = diff / (spread.replace(0.0, np.nan) + eps)  # NaN if no spread

            sig_flag = sig >= sig_thresh  # per-variable boolean

            def _style_sig(col: pd.Series):
                # color FT cells for rows deemed significant
                return ["background-color: #fde68a; font-weight: 600;" if sig_flag.get(idx, False) else "" 
                        for idx in col.index]

            styled = (med_tbl
                      .style
                      .apply(_style_sig, subset=["FT=1"])
                      .apply(_style_sig, subset=["FT=0"])
                      .format("{:.2f}"))

            st.dataframe(styled, use_container_width=True)
            st.caption("Highlighted where |median(FT=1)−median(FT=0)| ≥ σ × (MAD₁ + MAD₀).")
        else:
            st.info("Not enough info to compute significance (MADs missing). Rebuild models with the current DB.")
            cfg = {
                "FT=1": st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
                "FT=0": st.column_config.NumberColumn("FT=0 (median)", format="%.2f"),
            }
            st.dataframe(med_tbl, use_container_width=True, column_config=cfg, hide_index=False)

# ============================== ➕ Manual Input ==============================
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

    submitted = st.form_submit_button("Add to Table", use_container_width=True)

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
        "CatalystYN": catalyst_yn,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
    }
    st.session_state.rows.append(row)
    st.session_state.last = row
    st.success(f"Saved {ticker}.")
    do_rerun()

# ============================== Toolbar: Delete / Clear + Multiselect (FULL WIDTH) ==============================
if st.session_state.rows:
    tcol1, tcol2 = st.columns([1.6, 1.6])
    with tcol1:
        delete_clicked = st.button("Delete selected", use_container_width=True, type="primary", key="btn_delete")
    with tcol2:
        clear_disabled = len(st.session_state.rows) == 0
        clear_clicked = st.button("Clear Added Stocks", use_container_width=True, disabled=clear_disabled, key="btn_clear")

    # Full-width multiselect WITHOUT title, directly under buttons
    current_tickers = [str(r.get("Ticker", "")) for r in st.session_state.rows if str(r.get("Ticker","")).strip() != ""]
    selected_to_delete = st.multiselect(
        label="",
        options=current_tickers,
        default=[],
        placeholder="Select tickers to delete…",
        label_visibility="collapsed",
        key="ms_delete_fullwidth"
    )

    if clear_clicked:
        st.session_state.rows = []
        st.success("Cleared all added stocks.")
        do_rerun()

    if delete_clicked:
        if not selected_to_delete:
            st.info("No rows selected in the list.")
        else:
            before = len(st.session_state.rows)
            selset = set(map(str.upper, selected_to_delete))
            st.session_state.rows = [
                r for r in st.session_state.rows
                if str(r.get("Ticker", "")).strip().upper() not in selset
            ]
            removed = before - len(st.session_state.rows)
            if removed > 0:
                st.success(f"Removed {removed} row(s): {', '.join(sorted(selset))}")
            else:
                st.info("No rows removed.")
            do_rerun()

# ============================== Alignment (DataTables child-rows; ONLY added stocks) ==============================
st.markdown("### Alignment")

def _compute_alignment_counts(stock_row: dict, models_tbl: pd.DataFrame) -> dict:
    if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
        return {}
    groups = ["FT=1","FT=0"]
    cand_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                 "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst"]
    common = [v for v in cand_vars if (v in stock_row) and (v in models_tbl.index)]
    counts = {g: 0 for g in groups}; used = 0
    for v in common:
        xv = pd.to_numeric(stock_row.get(v), errors="coerce")
        if not np.isfinite(xv):
            continue
        med = models_tbl.loc[v, groups].astype(float).dropna()
        if med.empty:
            continue
        nearest = (med - xv).abs().idxmin()
        counts[nearest] += 1
        used += 1
    counts["N_Vars_Used"] = used
    return counts

models_tbl = (st.session_state.get("models") or {}).get("models_tbl", pd.DataFrame())
SIG_THR = float(st.session_state.get("sig_thresh", 2.0))
mad_tbl = (st.session_state.get("models") or {}).get("mad_tbl", pd.DataFrame())

if st.session_state.rows and not models_tbl.empty and {"FT=1","FT=0"}.issubset(models_tbl.columns):
    # Build summary for ADDED stocks only (no model rows)
    summary_rows, detail_map = [], {}
    num_vars = ["MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst"]

    for row in st.session_state.rows:
        stock = dict(row)
        tkr = stock.get("Ticker") or "—"
        counts = _compute_alignment_counts(stock, models_tbl)
        if not counts:
            continue

        like1, like0 = counts.get("FT=1",0), counts.get("FT=0",0)
        n_used = counts.get("N_Vars_Used",0)
        ft1_val = round((like1 / n_used * 100.0), 0) if n_used > 0 else 0.0
        ft0_val = round((like0 / n_used * 100.0), 0) if n_used > 0 else 0.0

        summary_rows.append({"Ticker": tkr, "FT1_val": ft1_val, "FT0_val": ft0_val})

        # ---- child details WITH significance flags vs FT=1 / FT=0 ----
        drows = []
        for v in num_vars:
            va = pd.to_numeric(stock.get(v), errors="coerce")
            v1 = models_tbl.loc[v, "FT=1"] if (v in models_tbl.index) else np.nan
            v0 = models_tbl.loc[v, "FT=0"] if (v in models_tbl.index) else np.nan
            m1 = mad_tbl.loc[v, "FT=1"] if (not mad_tbl.empty and v in mad_tbl.index and "FT=1" in mad_tbl.columns) else np.nan
            m0 = mad_tbl.loc[v, "FT=0"] if (not mad_tbl.empty and v in mad_tbl.index and "FT=0" in mad_tbl.columns) else np.nan

            if pd.isna(va) and pd.isna(v1) and pd.isna(v0):
                continue

            def _sig(delta, mad):
                if pd.isna(delta): return np.nan
                if pd.isna(mad):   return np.nan
                if mad == 0:
                    return np.inf if abs(delta) > 0 else 0.0
                return abs(delta) / abs(mad)

            d1 = None if (pd.isna(va) or pd.isna(v1)) else float(va - v1)
            d0 = None if (pd.isna(va) or pd.isna(v0)) else float(va - v0)
            s1 = _sig(d1, float(m1) if pd.notna(m1) else np.nan)
            s0 = _sig(d0, float(m0) if pd.notna(m0) else np.nan)

            sig1 = (not pd.isna(s1)) and (s1 >= SIG_THR)
            sig0 = (not pd.isna(s0)) and (s0 >= SIG_THR)

            drows.append({
                "Variable": v,
                "Value": None if pd.isna(va) else float(va),
                "FT1":   None if pd.isna(v1) else float(v1),
                "FT0":   None if pd.isna(v0) else float(v0),
                "d_vs_FT1": None if d1 is None else d1,
                "d_vs_FT0": None if d0 is None else d0,
                "sig1": sig1,
                "sig0": sig0,
            })

        detail_map[tkr] = drows

    if summary_rows:
        import streamlit.components.v1 as components
        payload = {"rows": summary_rows, "details": detail_map}

        html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  table.dataTable tbody tr { cursor: pointer; }

  .bar-wrap { display:flex; justify-content:center; align-items:center; gap:6px; }
  .bar { height: 12px; width: 120px; border-radius: 8px; background: #eee; position: relative; overflow: hidden; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; width: 0%; }
  .bar-label { font-size: 11px; white-space: nowrap; color:#374151; min-width: 24px; text-align:center; }
  .blue > span { background:#3b82f6; }  /* FT=1 = blue */
  .red  > span { background:#ef4444; }  /* FT=0 = red  */

  /* Align FT columns */
  #align td:nth-child(2), #align th:nth-child(2),
  #align td:nth-child(3), #align th:nth-child(3) { text-align: center; }

  /* Child table */
  .child-table { width: 100%; border-collapse: collapse; margin: 2px 0 2px 24px; table-layout: fixed; }
  .child-table th, .child-table td {
    font-size: 11px; padding: 3px 6px; border-bottom: 1px solid #e5e7eb;
    text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .child-table th:first-child, .child-table td:first-child { text-align:left; }

  /* Significance backgrounds */
  tr.sig_up td   { background: rgba(253, 230, 138, 0.85) !important; }  /* yellow-ish */
  tr.sig_down td { background: rgba(254, 202, 202, 0.85) !important; }  /* red-ish */

  /* Column widths for child table */
  .col-var { width: 18%; }
  .col-val { width: 12%; }
  .col-ft1 { width: 18%; }
  .col-ft0 { width: 18%; }
  .col-d1  { width: 17%; }
  .col-d0  { width: 17%; }

  .pos { color:#059669; } 
  .neg { color:#dc2626; }
</style>
</head>
<body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead>
      <tr>
        <th>Ticker</th>
        <th>FT=1</th>
        <th>FT=0</th>
      </tr>
    </thead>
  </table>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCellBlue(val) {
      const v = (val==null||isNaN(val)) ? 0 : Math.max(0, Math.min(100, val));
      return `
        <div class="bar-wrap">
          <div class="bar blue"><span style="width:${v}%"></span></div>
          <div class="bar-label">${v.toFixed(0)}</div>
        </div>`;
    }
    function barCellRed(val) {
      const v = (val==null||isNaN(val)) ? 0 : Math.max(0, Math.min(100, val));
      return `
        <div class="bar-wrap">
          <div class="bar red"><span style="width:${v}%"></span></div>
          <div class="bar-label">${v.toFixed(0)}</div>
        </div>`;
    }

    // Build the DataTable (no checkbox column)
    const table = $('#align').DataTable({
      data: data.rows,
      responsive: true,
      paging: false, info: false, searching: false,
      order: [[0,'asc']],
      columns: [
        { data: 'Ticker' },
        { data: 'FT1_val', render: (d)=>barCellBlue(d) },
        { data: 'FT0_val', render: (d)=>barCellRed(d) },
      ]
    });

    // Child rows
    function childTableHTML(ticker) {
      const rows = data.details[ticker] || [];
      if (!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No variable overlaps for this stock.</div>';
      const cells = rows.map(r => {
        const v  = (r.Value==null||isNaN(r.Value)) ? '' : Number(r.Value).toFixed(2);
        const f1 = (r.FT1==null ||isNaN(r.FT1))  ? '' : Number(r.FT1).toFixed(2);
        const f0 = (r.FT0==null ||isNaN(r.FT0))  ? '' : Number(r.FT0).toFixed(2);
        const d1 = (r.d_vs_FT1==null||isNaN(r.d_vs_FT1)) ? '' : Number(r.d_vs_FT1).toFixed(2);
        const d0 = (r.d_vs_FT0==null || isNaN(r.d_vs_FT0)) ? '' : Number(r.d_vs_FT0).toFixed(2);
        const c1 = (!d1)? '' : (parseFloat(d1)>=0 ? 'pos' : 'neg');
        const c0 = (!d0)? '' : (parseFloat(d0)>=0 ? 'pos' : 'neg');

        const s1 = !!r.sig1, s0 = !!r.sig0;
        const d1num = (r.d_vs_FT1==null || isNaN(r.d_vs_FT1)) ? NaN : Number(r.d_vs_FT1);
        const d0num = (r.d_vs_FT0==null || isNaN(r.d_vs_FT0)) ? NaN : Number(r.d_vs_FT0);

        let rowClass = '';
        if (s1 || s0) {
          let delta = NaN;
          if (s1 && s0) {
            const abs1 = isNaN(d1num) ? -Infinity : Math.abs(d1num);
            const abs0 = isNaN(d0num) ? -Infinity : Math.abs(d0num);
            delta = (abs1 >= abs0) ? d1num : d0num;
          } else {
            delta = !isNaN(d1num) && s1 ? d1num : d0num;
          }
          rowClass = (delta >= 0) ? 'sig_up' : 'sig_down';
        }

        return `
          <tr class="${rowClass}">
            <td class="col-var">${r.Variable}</td>
            <td class="col-val">${v}</td>
            <td class="col-ft1">${f1}</td>
            <td class="col-ft0">${f0}</td>
            <td class="col-d1 ${c1}">${d1}</td>
            <td class="col-d0 ${c0}">${d0}</td>
          </tr>`;
      }).join('');
      return `
        <table class="child-table">
          <colgroup>
            <col class="col-var"/><col class="col-val"/><col class="col-ft1"/><col class="col-ft0"/><col class="col-d1"/><col class="col-d0"/>
          </colgroup>
          <thead>
            <tr>
              <th class="col-var">Variable</th>
              <th class="col-val">Value</th>
              <th class="col-ft1">FT=1 median</th>
              <th class="col-ft0">FT=0 median</th>
              <th class="col-d1">Δ vs FT=1</th>
              <th class="col-d0">Δ vs FT=0</th>
            </tr>
          </thead>
          <tbody>${cells}</tbody>
        </table>`;
    }

    $('#align tbody').on('click', 'tr', function () {
      const row = table.row(this);
      if (row.child.isShown()) {
        row.child.hide(); $(this).removeClass('shown');
      } else {
        const ticker = row.data().Ticker;
        row.child(childTableHTML(ticker)).show(); $(this).addClass('shown');
      }
    });
  </script>
</body>
</html>
        """
        html = html.replace("%%PAYLOAD%%", json.dumps(payload))
        import streamlit.components.v1 as components
        components.html(html, height=620, scrolling=True)
    else:
        st.info("No eligible rows yet. Add manual stocks and/or ensure FT=1/FT=0 medians are built.")
elif st.session_state.rows and (models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns)):
    st.info("Upload DB and click **Build model stocks** to compute FT=1/FT=0 medians first.")
else:
    st.info("Add at least one stock above to compute alignment.")
