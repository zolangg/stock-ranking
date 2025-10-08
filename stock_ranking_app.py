# app.py — RF proximity similarity (FT=1 blue / FT=0 red)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== RF deps ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    st.error("Missing dependency: scikit-learn. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

# ============================== Page ==============================
st.set_page_config(page_title="Premarket RF Similarity", layout="wide")
st.title("Premarket RF Similarity (FT=1 blue / FT=0 red)")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])        # added stocks
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rf_model", {})    # {scaler, rf, leaf_train, X_scaled, y, feat_names, df_meta}
ss.setdefault("face_vars", [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%","Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"
])

# ============================== Helpers ==============================
def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def SAFE_JSON_DUMPS(obj) -> str:
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)
    s = json.dumps(obj, cls=NpEncoder, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</script>", "<\\/script>")

_norm_cache = {}
def _norm(s: str) -> str:
    if s in _norm_cache: return _norm_cache[s]
    v = re.sub(r"\s+", " ", str(s).strip().lower())
    v = v.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")
    _norm_cache[s] = v
    return v

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty: return None
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc:
                return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]:
                return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        ss_ = str(s).strip().replace(" ", "")
        if "," in ss_ and "." not in ss_:
            ss_ = ss_.replace(",", ".")
        else:
            ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

# ============================== Upload / Build ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build RF model", use_container_width=True, key="db_build_btn")

@st.cache_data(show_spinner=False)
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_data(show_spinner=True)
def _load_sheet(file_bytes: bytes):
    xls = pd.ExcelFile(file_bytes)
    sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
    sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
    raw = pd.read_excel(xls, sheet)
    return raw, sheet, tuple(xls.sheet_names)

def _safe_to_binary(v):
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        fv = float(sv); return 1 if fv >= 0.5 else 0
    except: return np.nan

def _safe_to_binary_float(v):
    x = _safe_to_binary(v)
    if np.isnan(x): return np.nan
    return float(x)

def _map_numeric(dfout, raw, name, candidates):
    src = _pick(raw, candidates)
    if src is not None:
        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

def _build_rf_from_df(df: pd.DataFrame, face_vars: list[str]):
    # Need FT01 + all face vars with no NaNs
    needed = ["FT01"] + face_vars
    if not set(needed).issubset(df.columns):
        return {}
    df_face = df[needed].dropna()
    if df_face.shape[0] < 30:
        return {}
    y = df_face["FT01"].astype(int).values
    X = df_face[face_vars].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(Xs, y)
    leaf_train = rf.apply(Xs)  # (n_samples, n_trees)
    # Meta (optional)
    meta_cols = []
    if "Ticker" in df.columns: meta_cols.append("Ticker")
    if "TickerDB" in df.columns: meta_cols.append("TickerDB")
    if "Max_Push_Daily_%" in df.columns: meta_cols.append("Max_Push_Daily_%")
    df_meta = df_face.reset_index(drop=True)
    extra = [c for c in meta_cols if c in df.columns]
    meta_frame = df.loc[df_face.index, extra].reset_index(drop=True) if extra else pd.DataFrame(index=df_face.index)
    meta_frame["FT01"] = y
    return {
        "scaler": scaler,
        "rf": rf,
        "leaf_train": leaf_train,
        "X_scaled": Xs,
        "y": y,
        "feat_names": face_vars,
        "df_meta": meta_frame.reset_index(drop=True),
    }

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            file_bytes = uploaded.getvalue()
            _ = _hash_bytes(file_bytes)
            raw, sel_sheet, all_sheets = _load_sheet(file_bytes)

            # Detect FT column
            possible = [c for c in raw.columns if _norm(c) in {"ft", "ft01", "group", "label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
                for c in raw.columns:
                    vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if len(vals) and vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group = c; break
            if col_group is None:
                st.error("Could not detect FT (0/1) column."); st.stop()

            df = pd.DataFrame()
            df["GroupRaw"] = raw[col_group]

            # Map numeric fields used by RF face
            _map_numeric(df, raw, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            _map_numeric(df, raw, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            _map_numeric(df, raw, "MarketCap_M$",     ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
            _map_numeric(df, raw, "Float_M",          ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
            _map_numeric(df, raw, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
            _map_numeric(df, raw, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            _map_numeric(df, raw, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            _map_numeric(df, raw, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            _map_numeric(df, raw, "PM_Vol_%",         ["pm vol (%)","pm_vol_%","pm vol percent","pm volume (%)","pm_vol_percent"])
            _map_numeric(df, raw, "Daily_Vol_M",      ["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])
            _map_numeric(df, raw, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            _map_numeric(df, raw, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            # Catalyst
            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            if cand_catalyst:
                df["Catalyst"] = pd.Series(raw[cand_catalyst]).map(_safe_to_binary_float)

            # Derived needed for face vars
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            mcap_basis  = "MC_PM_Max_M"    if "MC_PM_Max_M"    in df.columns and df["MC_PM_Max_M"].notna().any()    else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (pd.to_numeric(df["PM_$Vol_M$"], errors="coerce") / pd.to_numeric(df[mcap_basis], errors="coerce") * 100.0).replace([np.inf,-np.inf], np.nan)

            # Scale % fields (DB may store as fractions)
            if "Gap_%" in df.columns:         df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns: df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT labels
            df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()

            # Optional Ticker passthrough
            tcol = _pick(raw, ["ticker","symbol","name"])
            if tcol is not None:
                df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

            ss.base_df = df

            # Build RF model on the 9-face vars
            ss.rf_model = _build_rf_from_df(df, ss.face_vars) or {}
            if not ss.rf_model:
                st.error("RF model could not be built (need FT01 and all 9 variables; ≥30 valid rows).")
                st.stop()

            st.success(f"Loaded “{sel_sheet}”. RF model ready with {ss.rf_model['X_scaled'].shape[0]} rows.")
            do_rerun()
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============================== Add Stock ==============================
st.markdown("---")
st.subheader("Add Stock")

with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        ticker      = st.text_input("Ticker", "").strip().upper()
        mc_pmmax    = st.number_input("Premarket Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_pm    = st.number_input("Premarket Float (M)", 0.0, step=0.01, format="%.2f")
        gap_pct     = st.number_input("Gap %", 0.0, step=0.01, format="%.2f")
        max_pull_pm = st.number_input("Premarket Max Pullback (%)", 0.0, step=0.01, format="%.2f")
    with c2:
        atr_usd     = st.number_input("Prior Day ATR ($)", 0.0, step=0.01, format="%.2f")
        pm_vol      = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol      = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
        rvol_pm_cum = st.number_input("Premarket Max RVOL", 0.0, step=0.01, format="%.2f")
    with c3:
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)
    submitted = st.form_submit_button("Add & Compare", use_container_width=True)

def _row_to_face(row: dict, face_vars: list[str]):
    vals = []
    for f in face_vars:
        v = row.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        vals.append(float(v))
    return np.array(vals, dtype=float)

def _rf_proximity(model: dict, x_vec: np.ndarray):
    scaler = model["scaler"]; rf = model["rf"]; leaf_train = model["leaf_train"]
    xs = scaler.transform(x_vec.reshape(1, -1)).ravel()
    leaf_q = rf.apply(xs.reshape(1, -1)).ravel()
    prox = (leaf_train == leaf_q).mean(axis=1)  # (n_train,)
    order = np.argsort(-prox)                   # best first
    return prox[order], order

def _elbow_kstar(prox_sorted: np.ndarray, max_rank: int = 30, k_min=3, k_max=25):
    if prox_sorted.size < k_min:
        return max(1, prox_sorted.size)
    upto = min(max_rank, prox_sorted.size - 1)
    if upto < 2:
        return min(k_max, max(k_min, prox_sorted.size))
    gaps = prox_sorted[:upto] - prox_sorted[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

if submitted and ticker:
    row = {
        "Ticker": ticker,
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Gap_%": gap_pct,
        "ATR_$": atr_usd,
        "PM_Vol_M": pm_vol,
        "PM$Vol_M$": pm_dol,
        "Max_Pull_PM_%": max_pull_pm,
        "RVOL_Max_PM_cum": rvol_pm_cum,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
    }
    ss.rows.append(row)
    st.success(f"Saved {ticker}.")
    do_rerun()

# ============================== Similarity (RF) ==============================
st.markdown("### RF Similarity — FT=1 vs FT=0 (Top-K* by elbow)")

if ss.get("rf_model", {}):
    rf_m = ss["rf_model"]; face_vars = ss["face_vars"]
    n_train = rf_m["X_scaled"].shape[0]
    st.caption(f"Model rows: {n_train} • Features: {', '.join(face_vars)}")

    # Build summary for each added stock
    summary_rows = []
    detail_map = {}

    for row in ss.rows:
        tkr = row.get("Ticker") or "—"
        x = _row_to_face(row, face_vars)
        if x is None:
            summary_rows.append({
                "Ticker": tkr, "A_val_raw": 0.0, "B_val_raw": 0.0,
                "A_val_int": 0, "B_val_int": 0,
                "A_label": "FT=1", "B_label": "FT=0",
                "A_cnt": 0, "B_cnt": 0, "Kstar": 0
            })
            detail_map[tkr] = [{"__group__": "Missing required inputs for similarity."}]
            continue

        prox_sorted, order = _rf_proximity(rf_m, x)
        if prox_sorted.size == 0:
            summary_rows.append({
                "Ticker": tkr, "A_val_raw": 0.0, "B_val_raw": 0.0,
                "A_val_int": 0, "B_val_int": 0,
                "A_label": "FT=1", "B_label": "FT=0",
                "A_cnt": 0, "B_cnt": 0, "Kstar": 0
            })
            detail_map[tkr] = [{"__group__": "No proximities computed."}]
            continue

        k_star = _elbow_kstar(prox_sorted, max_rank=30, k_min=3, k_max=min(25, prox_sorted.size))
        sel_idx = order[:k_star]
        y = rf_m["y"][sel_idx]
        cnt_A = int((y == 1).sum())
        cnt_B = int((y == 0).sum())
        tot = max(1, cnt_A + cnt_B)
        pct_A = 100.0 * cnt_A / tot
        pct_B = 100.0 - pct_A

        summary_rows.append({
            "Ticker": tkr,
            "A_val_raw": pct_A, "B_val_raw": pct_B,
            "A_val_int": int(round(pct_A)), "B_val_int": int(round(pct_B)),
            "A_label": "FT=1", "B_label": "FT=0",
            "A_cnt": cnt_A, "B_cnt": cnt_B, "Kstar": int(k_star),
        })

        # Build details table for this ticker
        meta = rf_m.get("df_meta", pd.DataFrame())
        det_rows = [{"__group__": f"Top-K* neighbors (K*={k_star}) by RF proximity"}]
        for i, idx in enumerate(sel_idx, start=1):
            rec = {"Rank": i, "FT01": int(rf_m["y"][idx]), "Proximity": float(prox_sorted[i-1])}
            if not meta.empty:
                # Attach meta if present
                for c in meta.columns:
                    rec[c] = meta.iloc[idx][c]
            det_rows.append(rec)
        detail_map[tkr] = det_rows

    # ---------------- HTML/JS render (bars: FT=1 blue, FT=0 red) ----------------
    import streamlit.components.v1 as components

    def _round_rec(o):
        if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
        if isinstance(o, list): return [_round_rec(v) for v in o]
        if isinstance(o, float): return float(np.round(o, 6))
        return o

    payload = _round_rec({"rows": summary_rows, "details": detail_map})

    html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:160px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:54px;text-align:center}
  .blue>span{background:#3b82f6}.red>span{background:#ef4444}
  #align td:nth-child(2),#align th:nth-child(2),#align td:nth-child(3),#align th:nth-child(3){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>FT=1 (blue)</th><th>FT=0 (red)</th><th>K*</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCellLabeled(valRaw,label,valInt,count){
      const cls = (label==='FT=1') ? 'blue' : 'red';
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const pct=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
      const txt = `${pct}% • n=${count}`;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${txt}</div></div>`;
    }
    function formatVal(x){return (x==null||isNaN(x))?'':Number(x).toFixed(4);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No neighbors.</div>';

      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="8">'+r.__group__+'</td></tr>';
        const ft = (r.FT01===1)?'FT=1':'FT=0';
        const prox = formatVal(r.Proximity);
        const tick = (r.TickerDB||r.Ticker||'');
        const mpd = (r.Max_Push_Daily_%!=null)?Number(r.Max_Push_Daily_%).toFixed(2)+'%':'';
        return `<tr>
          <td>${r.Rank}</td><td>${ft}</td><td>${prox}</td><td>${tick}</td><td>${mpd}</td>
        </tr>`;
      }).join('');

      return `<table class="child-table">
        <thead><tr>
          <th>#</th><th>FT</th><th>Proximity</th><th>Ticker</th><th>MaxPushDaily%</th>
        </tr></thead><tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCellLabeled(row.A_val_raw,row.A_label,row.A_val_int,row.A_cnt)},
          {data:null, render:(row)=>barCellLabeled(row.B_val_raw,row.B_label,row.B_val_int,row.B_cnt)},
          {data:'Kstar'}
        ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { const t=row.data().Ticker; row.child(childTableHTML(t)).show(); $(this).addClass('shown'); }
      });
    });
  </script>
</body></html>
"""
    html = html.replace("%%PAYLOAD%%", SAFE_JSON_DUMPS(payload))
    components.html(html, height=620, scrolling=True)

# ============================== Delete Control ==============================
tickers = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
unique_tickers, _seen = [], set()
for t in tickers:
    if t and t not in _seen:
        unique_tickers.append(t); _seen.add(t)

del_cols = st.columns([4, 1])
with del_cols[0]:
    to_delete = st.multiselect(
        "",
        options=unique_tickers,
        default=[],
        key="del_selection",
        placeholder="Select tickers…",
        label_visibility="collapsed",
    )
with del_cols[1]:
    if st.button("Delete", use_container_width=True, key="delete_btn"):
        if to_delete:
            ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(to_delete)]
            st.success(f"Deleted: {', '.join(to_delete)}"); do_rerun()
        else:
            st.info("No tickers selected.")
