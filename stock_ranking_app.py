# app.py — Premarket Stock Ranking (NCA + kernel-kNN similarity; Alignment UI w/ child rows)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("nca_model", {})  # scaler, nca, X_learned, y, df_meta
ss.setdefault("face_vars", [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"
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
        ss_ = str(s).strip().replace(" ", "")
        if "," in ss_ and "." not in ss_:
            ss_ = ss_.replace(",", ".")
        else:
            ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

def _safe_to_binary(v):
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        fv = float(sv); return 1 if fv >= 0.5 else 0
    except: return np.nan

def _safe_to_binary_float(v):
    x = _safe_to_binary(v)
    return float(x) if x in (0,1) else np.nan

# ============================== NCA + kernel-kNN ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
except ModuleNotFoundError:
    st.error("scikit-learn not installed. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

def _row_to_face(row: dict, face_vars: list[str]) -> np.ndarray | None:
    vals = []
    for f in face_vars:
        v = row.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        try: vals.append(float(v))
        except: return None
    return np.array(vals, dtype=float)

def _build_nca(df: pd.DataFrame, face_vars: list[str]) -> dict:
    need = ["FT01"] + face_vars
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"DB missing required columns: {miss}")
        return {}

    df_face = df[need].dropna()
    if df_face.shape[0] < 30:
        st.error(f"Need ≥30 rows without NaNs in {need}. Have {df_face.shape[0]}.")
        return {}

    y = df_face["FT01"].astype(int).values
    X = df_face[face_vars].astype(float).values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    nca = NeighborhoodComponentsAnalysis(random_state=42, max_iter=1000, tol=1e-5)
    nca.fit(Xs, y)
    Xz = nca.transform(Xs)

    # Keep meta with FT + ticker + all 9 features for display
    meta_cols = ["FT01"]
    for c in ("TickerDB", "Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in face_vars:
        if f in df.columns and f not in meta_cols:
            meta_cols.append(f)
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True)

    return {"scaler": scaler, "nca": nca, "X_learned": Xz, "y": y, "df_meta": meta}

def _elbow_kstar(sim: np.ndarray, max_rank=30, k_min=3, k_max=25) -> int:
    if sim.size <= k_min: return max(1, sim.size)
    upto = min(max_rank, sim.size - 1)
    if upto < 2: return min(k_max, max(k_min, sim.size))
    gaps = sim[:upto] - sim[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

def _nca_similarity(model: dict, x_vec: np.ndarray, k_max=50):
    scaler, nca, Xz, y = model["scaler"], model["nca"], model["X_learned"], model["y"]
    xq = scaler.transform(x_vec.reshape(1, -1))
    zq = nca.transform(xq).ravel()

    d = np.linalg.norm(Xz - zq[None, :], axis=1)  # distances
    order = np.argsort(d)
    d_sorted = d[order]
    y_sorted = y[order]

    kk = min(20, len(d_sorted))
    if kk == 0:
        return 0.0, 0.0, 0, order[:0], np.array([])
    bw = np.median(d_sorted[:kk])
    if not np.isfinite(bw) or bw <= 1e-12:
        bw = np.mean(d_sorted[:kk]) + 1e-6

    w = np.exp(- (d_sorted / bw) ** 2)  # Gaussian kernel weights

    scale = d_sorted[kk-1] + 1e-9
    sim = 1.0 - (d_sorted / scale)
    sim = np.clip(sim, 0.0, 1.0)

    K = _elbow_kstar(sim, max_rank=min(k_max, len(sim)), k_min=3, k_max=min(25, len(sim)))
    sel = slice(0, K)
    wK, yK = w[sel], y_sorted[sel]
    w_sum = wK.sum() if wK.size else 1.0
    p1 = float((wK * (yK == 1)).sum() / w_sum) * 100.0
    p0 = 100.0 - p1
    return p1, p0, int(K), order[:K], w[:K]

# ============================== Variables (rest of app can reuse) ==============================
VAR_CORE = [
    "Gap_%", "FR_x", "PM$Vol/MC_%", "Catalyst", "PM_Vol_%", "Max_Pull_PM_%", "RVOL_Max_PM_cum",
]
VAR_MODERATE = [
    "MC_PM_Max_M", "Float_PM_Max_M", "PM_Vol_M", "PM_$Vol_M$", "ATR_$", "Daily_Vol_M", "MarketCap_M$", "Float_M",
]

# ============================== Upload / Build (DB) ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

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

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            file_bytes = uploaded.getvalue()
            _ = _hash_bytes(file_bytes)
            raw, sel_sheet, all_sheets = _load_sheet(file_bytes)

            # detect FT column
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
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

            def add_num(dfout, name, src_candidates):
                src = _pick(raw, src_candidates)
                if src: dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

            # map fields (incl. the 9 face vars + some derived)
            add_num(df, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            add_num(df, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            add_num(df, "MarketCap_M$",     ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
            add_num(df, "Float_M",          ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
            add_num(df, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
            add_num(df, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            add_num(df, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            add_num(df, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            add_num(df, "PM_Vol_%",         ["pm vol (%)","pm_vol_%","pm vol percent","pm volume (%)","pm_vol_percent"])
            add_num(df, "Daily_Vol_M",      ["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])
            add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            # catalyst
            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_safe_to_binary_float)

            # derived FACE vars
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # scale % fields (DB stores fractions)
            if "Gap_%" in df.columns:            df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "PM_Vol_% in df.columns":         df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns:    df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT groups
            df["FT01"] = df["GroupRaw"].map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

            # optional passthrough ticker
            tcol = _pick(raw, ["ticker","symbol","name"])
            if tcol is not None:
                df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

            # save base and build NCA on full FT=0/1 set
            ss.base_df = df
            ss.nca_model = _build_nca(df, ss.face_vars)
            if not ss.nca_model: st.stop()
            st.success(f"Loaded “{sel_sheet}”. NCA model ready with {ss.nca_model['X_learned'].shape[0]} rows.")
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
        rvol_pm_cum = st.number_input("Premarket Max RVOL (cum)", 0.0, step=0.01, format="%.2f")
    with c3:
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)
    submitted = st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
    fr   = (pm_vol / float_pm) if float_pm > 0 else 0.0
    pmmc = (pm_dol / mc_pmmax * 100.0) if mc_pmmax > 0 else 0.0
    row = {
        "Ticker": ticker,
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Gap_%": gap_pct,
        "ATR_$": atr_usd,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,
        "FR_x": fr,
        "PM$Vol/MC_%": pmmc,
        "Max_Pull_PM_%": max_pull_pm,
        "RVOL_Max_PM_cum": rvol_pm_cum,
        "CatalystYN": catalyst_yn,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
    }
    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Alignment (NCA similarity) ==============================
st.markdown("### Alignment")

# Compact top row like your app (kept for UI parity; no effect on NCA)
col_mode, col_gain = st.columns([2.8, 1.0])
with col_mode:
    st.radio(
        "", ["FT vs Fail (Gain% cutoff on FT=1 only)", "FT=1 (High vs Low cutoff)", "Gain% vs Rest"],
        horizontal=True, key="cmp_mode", label_visibility="collapsed",
    )
with col_gain:
    st.selectbox("", [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        index=2, key="gain_min_pct", help="(Display only; NCA uses FT labels on full set.)",
        label_visibility="collapsed",
    )

base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty or not ss.nca_model:
    st.info("Upload DB and click **Build model stocks** first.")
    st.stop()

# Build similarity summaries & neighbor details
gA, gB = "FT=1", "FT=0"
summary_rows, detail_map = [], {}
face_vars = ss.face_vars

for row in ss.rows:
    stock = dict(row); tkr = stock.get("Ticker") or "—"
    x = _row_to_face(stock, face_vars)
    if x is None:
        summary_rows.append({"Ticker": tkr, "A_val_raw":0.0,"B_val_raw":0.0,"A_val_int":0,"B_val_int":0,
                             "A_label":gA,"B_label":gB})
        detail_map[tkr] = [{"__group__":"Missing inputs for similarity (need all 9 variables)."}]
        continue

    p1, p0, K, idx_sel, w_sel = _nca_similarity(ss.nca_model, x, k_max=50)
    summary_rows.append({
        "Ticker": tkr,
        "A_val_raw": p1, "B_val_raw": p0,
        "A_val_int": int(round(p1)), "B_val_int": int(round(p0)),
        "A_label": gA, "B_label": gB,
    })

    # neighbor rows: top-K by weight (show top 12)
    meta = ss.nca_model.get("df_meta", pd.DataFrame())
    rows = [{"__group__": f"Top-{K} neighbors by NCA distance (kernel-weighted)"}]
    if len(idx_sel):
        top_idx = np.argsort(-w_sel)[:12]
        for rnk, pos in enumerate(top_idx, 1):
            idx = int(idx_sel[pos])
            wt  = float(w_sel[pos])
            rec = {"#": rnk, "FT": int(ss.nca_model["y"][idx]), "Weight": wt}
            if not meta.empty and 0 <= idx < len(meta):
                m = meta.iloc[idx]
                rec["Ticker"] = m.get("TickerDB", m.get("Ticker", ""))
                # attach the 9 features used for similarity
                for f in face_vars:
                    val = m.get(f, np.nan)
                    rec[f] = float(val) if (val is not None and np.isfinite(val)) else None
            rows.append(rec)
    detail_map[tkr] = rows

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": summary_rows, "details": detail_map, "gA": gA, "gB": gB, "face_vars": face_vars})

html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:120px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:28px;text-align:center}
  .blue>span{background:#3b82f6}.red>span{background:#ef4444}
  #align td:nth-child(2),#align th:nth-child(2),#align td:nth-child(3),#align th:nth-child(3){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:auto}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .w-col{width:80px}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th id="hdrA"></th><th id="hdrB"></th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;
    document.getElementById('hdrA').textContent = data.gA + " (blue)";
    document.getElementById('hdrB').textContent = data.gB + " (red)";

    function barCellLabeled(valRaw,label,valInt){
      const cls=(label==='FT=1')?'blue':'red';
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,Number(valRaw)));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):Number(valInt);
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function fmt(x){return (x==null||isNaN(x))?'':Number(x).toFixed(2);}
    function fmtW(x){return (x==null||isNaN(x))?'':Number(x).toFixed(4);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No neighbors.</div>';

      // Build dynamic header for 9 features
      const fv = data.face_vars || [];
      const headF = fv.map(v=>`<th>${v}</th>`).join('');

      const cells = rows.map(r=>{
        if(r.__group__) return `<tr class="group-row"><td colspan="${4 + fv.length}">${r.__group__}</td></tr>`;
        const ft = (r.FT===1)?'FT=1':'FT=0';
        const tick = (r.Ticker||'');
        const feats = fv.map(v=>{
          const val = r[v];
          return `<td>${fmt(val)}</td>`;
        }).join('');
        return `<tr>
          <td class="w-col">${r["#"]||""}</td>
          <td>${ft}</td>
          <td class="w-col">${fmtW(r.Weight)}</td>
          <td>${tick}</td>
          ${feats}
        </tr>`;
      }).join('');

      return `<table class="child-table">
        <thead><tr>
          <th class="w-col">#</th><th>FT</th><th class="w-col">Weight</th><th>Ticker</th>${headF}
        </tr></thead>
        <tbody>${cells}</tbody>
      </table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCellLabeled(row.A_val_raw,row.A_label,row.A_val_int)},
          {data:null, render:(row)=>barCellLabeled(row.B_val_raw,row.B_label,row.B_val_int)}
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

# ============================== Delete Control (below table; no title) ==============================
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
            st.success(f"Deleted: {', '.join(to_delete)}")
            do_rerun()
        else:
            st.info("No tickers selected.")
