import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ===== RF deps =====
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    st.error("scikit-learn not installed. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

st.set_page_config(page_title="RF Similarity (FT=1 blue / FT=0 red)", layout="wide")
st.title("RF Similarity — FT=1 (blue) vs FT=0 (red)")

ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rf_model", {})     # scaler, rf, leaf_train, X_scaled, y, feat_names, df_meta
ss.setdefault("rows", [])         # added queries

FACE_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"
]

# ---------- helpers ----------
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc: return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n: return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        s = str(s).strip().replace(" ", "")
        if "," in s and "." not in s: s = s.replace(",", ".")
        else: s = s.replace(",", "")
        return float(s)
    except: return np.nan

def _safe_to_binary(v):
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        return 1 if float(sv) >= 0.5 else 0
    except: return np.nan

def _safe_to_binary_float(v):
    x = _safe_to_binary(v)
    return float(x) if x in (0,1) else np.nan

def _map_numeric(dfout, raw, name, candidates):
    src = _pick(raw, candidates)
    if src is not None:
        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

def _build_rf(df: pd.DataFrame):
    need = ["FT01"] + FACE_VARS
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"DB missing required columns: {miss}")
        return {}
    df_face = df[need].dropna()
    if df_face.shape[0] < 30:
        st.error(f"Need ≥30 rows without NaNs in {need}. Have {df_face.shape[0]}.")
        return {}
    y = df_face["FT01"].astype(int).values
    X = df_face[FACE_VARS].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(Xs, y)
    leaf_train = rf.apply(Xs)
    meta_cols = []
    if "Ticker" in df.columns: meta_cols.append("Ticker")
    if "TickerDB" in df.columns: meta_cols.append("TickerDB")
    if "Max_Push_Daily_%" in df.columns: meta_cols.append("Max_Push_Daily_%")
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=df_face.index)
    meta["FT01"] = y
    return {"scaler":scaler,"rf":rf,"leaf_train":leaf_train,"X_scaled":Xs,"y":y,"feat_names":FACE_VARS,"df_meta":meta}

def _row_to_face(row: dict):
    vals = []
    for f in FACE_VARS:
        v = row.get(f, None)
        if v is None or (isinstance(v,float) and not np.isfinite(v)):
            return None
        vals.append(float(v))
    return np.array(vals, dtype=float)

def _rf_proximity(model, x_vec):
    xs = model["scaler"].transform(x_vec.reshape(1,-1)).ravel()
    leaf_q = model["rf"].apply(xs.reshape(1,-1)).ravel()
    prox = (model["leaf_train"] == leaf_q).mean(axis=1)  # (n_train,)
    order = np.argsort(-prox)
    return prox[order], order

def _elbow_kstar(prox_sorted, max_rank=30, k_min=3, k_max=25):
    if prox_sorted.size <= k_min: return max(1, prox_sorted.size)
    upto = min(max_rank, prox_sorted.size - 1)
    if upto < 2: return min(k_max, max(k_min, prox_sorted.size))
    gaps = prox_sorted[:upto] - prox_sorted[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

def _safe_json(obj):
    class Np(json.JSONEncoder):
        def default(self, o):
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            if isinstance(o,(np.ndarray,)): return o.tolist()
            return super().default(o)
    return json.dumps(obj, cls=Np, ensure_ascii=False, separators=(",",":")).replace("</script>","<\\/script>")

# ---------- upload / build ----------
st.subheader("Upload Database")
upl = st.file_uploader("Upload .xlsx", type=["xlsx"])
if st.button("Build RF model", use_container_width=True):
    if not upl:
        st.error("Upload an Excel first."); st.stop()
    try:
        raw = pd.read_excel(upl)
        # detect FT
        poss = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
        col_group = poss[0] if poss else None
        if col_group is None:
            for c in raw.columns:
                vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                if len(vals) and vals.isin(["0","1","true","false","yes","no"]).all():
                    col_group = c; break
        if col_group is None:
            st.error("Could not detect FT (0/1) column."); st.stop()
        df = pd.DataFrame()
        df["GroupRaw"] = raw[col_group]
        # map fields needed for face
        _map_numeric(df, raw, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
        _map_numeric(df, raw, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
        _map_numeric(df, raw, "MarketCap_M$",     ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
        _map_numeric(df, raw, "Float_M",          ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
        _map_numeric(df, raw, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
        _map_numeric(df, raw, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
        _map_numeric(df, raw, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
        _map_numeric(df, raw, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
        _map_numeric(df, raw, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
        _map_numeric(df, raw, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])
        # Catalyst
        cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
        if cand_catalyst:
            df["Catalyst"] = pd.Series(raw[cand_catalyst]).map(_safe_to_binary_float)
        # derived PM$Vol/MC_% from MarketCap or MC_PM_Max_M
        basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
        if {"PM_$Vol_M$", basis}.issubset(df.columns):
            df["PM$Vol/MC_%"] = (pd.to_numeric(df["PM_$Vol_M$"], errors="coerce") /
                                 pd.to_numeric(df[basis], errors="coerce") * 100.0).replace([np.inf,-np.inf], np.nan)
        # scale % fields (if stored as fraction)
        if "Gap_%" in df.columns:         df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
        if "Max_Pull_PM_%" in df.columns: df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0
        # FT labels
        df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
        df = df[df["FT01"].isin([0,1])].copy()
        # passthrough ticker if present
        tcol = _pick(raw, ["ticker","symbol","name"])
        if tcol is not None: df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()
        ss.base_df = df
        ss.rf_model = _build_rf(df)
        if not ss.rf_model: st.stop()
        st.success(f"RF model ready with {ss.rf_model['X_scaled'].shape[0]} rows.")
    except Exception as e:
        st.error("Loading/processing failed.")
        st.exception(e)

# ---------- Add Stock ----------
st.markdown("---")
st.subheader("Add Stock to Compare (9 variables)")
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
    submit = st.form_submit_button("Add & Compare", use_container_width=True)

if submit:
    row = {
        "Ticker": ticker or "—",
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "ATR_$": atr_usd,
        "Gap_%": gap_pct,
        "Max_Pull_PM_%": max_pull_pm,
        "PM_Vol_M": pm_vol,
        "PM$Vol/MC_%": (pm_dol / mc_pmmax * 100.0) if mc_pmmax > 0 else np.nan,
        "RVOL_Max_PM_cum": rvol_pm_cum,
    }
    ss.rows.append(row)
    st.success(f"Saved {row['Ticker']}")

# ---------- RF Similarity ----------
st.markdown("### Similarity (Top-K* by elbow) — bars show FT=1 (blue) vs FT=0 (red) share")

if not ss.rf_model:
    st.info("Build the RF model first.")
else:
    rf_m = ss.rf_model
    # build summaries
    summaries, details = [], {}
    for row in ss.rows:
        tkr = row.get("Ticker","—")
        x = _row_to_face(row)
        if x is None:
            summaries.append({"Ticker": tkr, "pA":0.0,"pB":0.0,"iA":0,"iB":0,"cntA":0,"cntB":0,"K":0})
            details[tkr] = [{"__group__":"Missing inputs for similarity (need all 9 variables)."}]
            continue
        prox, order = _rf_proximity(rf_m, x)
        if prox.size == 0:
            summaries.append({"Ticker": tkr, "pA":0.0,"pB":0.0,"iA":0,"iB":0,"cntA":0,"cntB":0,"K":0})
            details[tkr] = [{"__group__":"No proximities computed."}]
            continue
        K = _elbow_kstar(prox, max_rank=30, k_min=3, k_max=min(25, prox.size))
        sel = order[:K]; y = rf_m["y"][sel]
        cntA = int((y==1).sum()); cntB = int((y==0).sum()); tot = max(1, cntA+cntB)
        pA = 100.0*cntA/tot; pB = 100.0 - pA
        summaries.append({"Ticker": tkr, "pA":pA,"pB":pB,"iA":int(round(pA)),"iB":int(round(pB)),"cntA":cntA,"cntB":cntB,"K":int(K)})
        # detail rows
        meta = rf_m.get("df_meta", pd.DataFrame())
        rows = [{"__group__": f"Top-{K} neighbors by RF proximity"}]
        for rnk, idx in enumerate(sel, start=1):
            rec = {"#": rnk, "FT": int(rf_m["y"][idx]), "Proximity": float(prox[rnk-1])}
            if not meta.empty:
                rec.update({c: meta.iloc[idx][c] for c in meta.columns})
            rows.append(rec)
        details[tkr] = rows

    # render bars
    import streamlit.components.v1 as components
    def _round(o):
        if isinstance(o, dict): return {k:_round(v) for k,v in o.items()}
        if isinstance(o, list): return [_round(v) for v in o]
        if isinstance(o, float): return float(np.round(o, 6))
        return o
    payload = _round({"rows":[
        {"Ticker":s["Ticker"],"A_val_raw":s["pA"],"B_val_raw":s["pB"],
         "A_val_int":s["iA"],"B_val_int":s["iB"],
         "A_cnt":s["cntA"],"B_cnt":s["cntB"],"Kstar":s["K"]}
        for s in summaries
    ], "details":details})

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
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:70px;text-align:center}
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
    function barCell(valRaw,isBlue,valInt,count){
      const cls = isBlue ? 'blue' : 'red';
      const w = (!valRaw||isNaN(valRaw)) ? 0 : Math.max(0, Math.min(100, valRaw));
      const pct = (!valInt||isNaN(valInt)) ? Math.round(w) : valInt;
      const txt = `${pct}% • n=${count}`;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${txt}</div></div>`;
    }
    function fmt(x){return (x==null||isNaN(x))?'':Number(x).toFixed(4);}
    function childHTML(t){
      const rows=(data.details||{})[t]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No neighbors.</div>';
      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="8">'+r.__group__+'</td></tr>';
        const ft = (r.FT===1)?'FT=1':'FT=0';
        const tick = (r.TickerDB||r.Ticker||'');
        const mpd = (r.Max_Push_Daily_%!=null)?Number(r.Max_Push_Daily_%).toFixed(2)+'%':'';
        return `<tr><td>${r["#"]}</td><td>${ft}</td><td>${fmt(r.Proximity)}</td><td>${tick}</td><td>${mpd}</td></tr>`;
      }).join('');
      return `<table class="child-table"><thead><tr><th>#</th><th>FT</th><th>Proximity</th><th>Ticker</th><th>MaxPushDaily%</th></tr></thead><tbody>${cells}</tbody></table>`;
    }
    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(r)=>barCell(r.A_val_raw,true,r.A_val_int,r.A_cnt)},
          {data:null, render:(r)=>barCell(r.B_val_raw,false,r.B_val_int,r.B_cnt)},
          {data:'Kstar'}
        ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { row.child(childHTML(row.data().Ticker)).show(); $(this).addClass('shown'); }
      });
    });
  </script></body></html>
"""
    components.html(html.replace("%%PAYLOAD%%", _safe_json(payload)), height=620, scrolling=True)

# delete control
names = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
opts, seen = [], set()
for t in names:
    if t and t not in seen: opts.append(t); seen.add(t)
c1, c2 = st.columns([4,1])
with c1:
    sel = st.multiselect("", options=opts, default=[], placeholder="Select tickers…", label_visibility="collapsed")
with c2:
    if st.button("Delete", use_container_width=True):
        if sel:
            ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(sel)]
            st.success(f"Deleted: {', '.join(sel)}"); st.experimental_rerun()
        else:
            st.info("No tickers selected.")
