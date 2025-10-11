# app.py — Premarket Stock Ranking
# (Median-only centers; Gain% filter; 3σ coloring; delete UI callback-safe; NCA bar w/ live features; distributions w/ clear labels)

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
ss.setdefault("lassoA", {})
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("var_core", [])
ss.setdefault("var_moderate", [])
ss.setdefault("nca_model", {})
ss.setdefault("cat_model", {})           # ==== CATTAG: keep CatBoost model in state
ss.setdefault("del_selection", [])       # for delete UI
ss.setdefault("__delete_msg", None)      # flash msg
ss.setdefault("__catboost_warned", False) # one-time missing-package notice

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
        if "," in ss_ and "." not in ss_: ss_ = ss_.replace(",", ".")
        else: ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))

# ---------- Winsorization ----------
def _compute_bounds(arr: np.ndarray, lo_q=0.01, hi_q=0.99):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (np.nan, np.nan)
    return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

def _apply_bounds(arr: np.ndarray, lo: float, hi: float):
    out = arr.copy()
    if np.isfinite(lo): out = np.maximum(out, lo)
    if np.isfinite(hi): out = np.minimum(out, hi)
    return out

# ---------- Isotonic Regression ----------
def _pav_isotonic(x: np.ndarray, y: np.ndarray):
    order = np.argsort(x)
    xs = x[order]; ys = y[order]
    level_y = ys.astype(float).copy(); level_n = np.ones_like(level_y)
    i = 0
    while i < len(level_y) - 1:
        if level_y[i] > level_y[i+1]:
            new_y = (level_y[i]*level_n[i] + level_y[i+1]*level_n[i+1]) / (level_n[i] + level_n[i+1])
            new_n = level_n[i] + level_n[i+1]
            level_y[i] = new_y; level_n[i] = new_n
            level_y = np.delete(level_y, i+1)
            level_n = np.delete(level_n, i+1)
            xs = np.delete(xs, i+1)
            if i > 0: i -= 1
        else:
            i += 1
    return xs, level_y

def _iso_predict(break_x: np.ndarray, break_y: np.ndarray, x_new: np.ndarray):
    if break_x.size == 0: return np.full_like(x_new, np.nan, dtype=float)
    idx = np.argsort(break_x)
    bx = break_x[idx]; by = break_y[idx]
    if bx.size == 1: return np.full_like(x_new, by[0], dtype=float)
    return np.interp(x_new, bx, by, left=by[0], right=by[-1])

# ============================== Variables ==============================
VAR_CORE = [
    "Gap_%",
    "FR_x",
    "PM$Vol/MC_%",
    "Catalyst",
    "PM_Vol_%",
    "Max_Pull_PM_%",
    "RVOL_Max_PM_cum",
]
VAR_MODERATE = [
    "MC_PM_Max_M",
    "Float_PM_Max_M",
    "PM_Vol_M",
    "PM_$Vol_M$",
    "ATR_$",
    "Daily_Vol_M",
    "MarketCap_M$",
    "Float_M",
]
VAR_ALL = VAR_CORE + VAR_MODERATE

# NCA: only “live” features from Add Stock (exclude PredVol_M / PM_Vol_% / Daily_Vol_M)
ALLOWED_LIVE_FEATURES = [
    "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
    "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
]
EXCLUDE_FOR_NCA = {"PredVol_M","PM_Vol_%","Daily_Vol_M"}

# ============================== LASSO (unchanged core) ==============================
def _kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return np.array_split(idx, k)

def _lasso_cd_std(Xs, y, lam, max_iter=900, tol=1e-6):
    n, p = Xs.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        w_old = w.copy()
        y_hat = Xs @ w
        for j in range(p):
            r_j = y - y_hat + Xs[:, j] * w[j]
            rho = (Xs[:, j] @ r_j) / n
            if   rho < -lam/2: w[j] = rho + lam/2
            elif rho >  lam/2: w[j] = rho - lam/2
            else:              w[j] = 0.0
            y_hat = Xs @ w
        if np.linalg.norm(w - w_old) < tol: break
    return w

# ... (unchanged: train_ratio_winsor_iso, predict_daily_calibrated, Upload/Build, Add Stock, Alignment UI, group building, med_tbl/mad_tbl, etc.)

# ============================== NCA training (live features only) ==============================
def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    # (unchanged)
    # ...
    return {
        "ok": True, "kind": used, "feats": feats,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "w_vec": (None if used=="nca" else (w_vec.tolist() if w_vec is not None else None)),
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
        "platt": (platt_params if platt_params is not None else None),
        "gA": gA_label, "gB": gB_label,
    }

def _nca_predict_proba(row: dict, model: dict) -> float:
    # (unchanged)
    # ...
    return float(np.clip(pA, 0.0, 1.0))

# ============================== CatBoost training (same live features) ==============================  # ==== CATTAG
def _train_catboost(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    """
    Train CatBoost on the same live features as NCA. Output calibrated P(A).
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception:
        if not ss.get("__catboost_warned", False):
            st.info("CatBoost is not installed. Run `pip install catboost` to enable the purple CatBoost column/series.")
            ss["__catboost_warned"] = True
        return {}

    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats:
        return {}

    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)

    mask = np.isfinite(Xdf.values).all(axis=1)
    X = Xdf.values[mask]
    yy = y[mask]
    if X.shape[0] < 40 or np.unique(yy).size < 2:
        return {}

    # simple train/val split
    n = X.shape[0]
    split = max(10, int(n * 0.8))
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = yy[:split], yy[split:]

    # CatBoost (silent, small trees for speed)
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    model.fit(Xtr, ytr, eval_set=(Xva, yva))

    # Get raw probs on validation to calibrate
    if Xva.shape[0] >= 8:
        p_raw = model.predict_proba(Xva)[:, 1]
        z = p_raw.astype(float)
        yf = yva.astype(int)

        iso_bx, iso_by = np.array([]), np.array([])
        platt = None
        if np.unique(z).size >= 3:
            bx, by = _pav_isotonic(z, yf.astype(float))
            if len(bx) >= 2:
                iso_bx, iso_by = np.array(bx), np.array(by)
        if iso_bx.size < 2:
            # simple logistic mapping around mid with slope by pooled std
            z0 = z[yf==0]; z1 = z[yf==1]
            if z0.size and z1.size:
                m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                m = 0.5*(m0+m1)
                k = 2.0 / (0.5*(s0+s1) + 1e-6)
                platt = (m, k)
    else:
        iso_bx, iso_by, platt = np.array([]), np.array([]), None

    return {
        "ok": True,
        "feats": feats,
        "gA": gA_label, "gB": gB_label,
        "cb": model,
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
        "platt": platt
    }

def _cat_predict_proba(row: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    x = []
    for f in feats:
        v = pd.to_numeric(row.get(f), errors="coerce")
        if not np.isfinite(v): return np.nan
        x.append(float(v))
    x = np.array(x, dtype=float).reshape(1, -1)

    try:
        cb = model.get("cb")
        if cb is None: return np.nan
        z = float(cb.predict_proba(x)[0, 1])  # raw prob of class A
    except Exception:
        return np.nan

    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        if not pl: 
            pA = z
        else:
            m, k = pl
            pA = 1.0 / (1.0 + np.exp(-k*(z - m)))
    return float(np.clip(pA, 0.0, 1.0))

# Train NCA + CatBoost on current split
features_for_nca = VAR_ALL[:]  # filtered in trainer to live features
ss.nca_model = _train_nca_or_lda(df_cmp, gA, gB, features_for_nca) or {}
ss.cat_model = _train_catboost(df_cmp, gA, gB, features_for_nca) or {}   # ==== CATTAG

# ---------- alignment computation for entered rows ----------
# (unchanged up to summary_rows loop, then we add CatBoost columns)

summary_rows, detail_map = [], {}
detail_order = [("Core variables", var_core),
                ("Moderate variables", var_mod + (["PredVol_M"] if "PredVol_M" not in var_mod else []))]

for row in ss.rows:
    stock = dict(row); tkr = stock.get("Ticker") or "—"
    counts = _compute_alignment_counts_weighted(
        stock_row=stock, centers_tbl=centers_tbl, var_core=var_core, var_mod=var_mod,
        w_core=1.0, w_mod=0.5, tie_mode="split",
    )
    if not counts: continue

    # NCA probability (A)
    pA = _nca_predict_proba(stock, ss.get("nca_model", {}))
    nca_raw = float(pA)*100.0 if np.isfinite(pA) else np.nan
    nca_int = int(round(nca_raw)) if np.isfinite(nca_raw) else None

    # CatBoost probability (A)                                    # ==== CATTAG
    pC = _cat_predict_proba(stock, ss.get("cat_model", {}))
    cat_raw = float(pC)*100.0 if np.isfinite(pC) else np.nan
    cat_int = int(round(cat_raw)) if np.isfinite(cat_raw) else None

    summary_rows.append({
        "Ticker": tkr,
        "A_val_raw": counts.get("A_pct_raw", 0.0),
        "B_val_raw": counts.get("B_pct_raw", 0.0),
        "A_val_int": counts.get("A_pct_int", 0),
        "B_val_int": counts.get("B_pct_int", 0),
        "A_label": counts.get("A_label", gA),
        "B_label": counts.get("B_label", gB),
        "A_pts": counts.get("A_pts", 0.0),
        "B_pts": counts.get("B_pts", 0.0),
        "A_core": counts.get("A_core", 0.0),
        "B_core": counts.get("B_core", 0.0),
        "A_mod": counts.get("A_mod", 0.0),
        "B_mod": counts.get("B_mod", 0.0),
        "NCA_raw": nca_raw,
        "NCA_int": nca_int,
        "CAT_raw": cat_raw,           # ==== CATTAG
        "CAT_int": cat_int,           # ==== CATTAG
    })

    # details (unchanged)
    # ...

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": summary_rows, "details": detail_map, "gA": gA, "gB": gB})

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
  .blue>span{background:#3b82f6}.red>span{background:#ef4444}.green>span{background:#10b981}
  .purple>span{background:#8b5cf6} /* ==== CATTAG: CatBoost purple */
  #align td:nth-child(2),#align th:nth-child(2),
  #align td:nth-child(3),#align th:nth-child(3),
  #align td:nth-child(4),#align th:nth-child(4),
  #align td:nth-child(5),#align th:nth-child(5){text-align:center} /* ==== CATTAG: 4th metric col */
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .col-var{width:18%}.col-val{width:12%}.col-a{width:18%}.col-b{width:18%}.col-da{width:17%}.col-db{width:17%}
  .pos{color:#059669}.neg{color:#dc2626}
  .sig-hi{background:rgba(250,204,21,0.18)!important}
  .sig-lo{background:rgba(239,68,68,0.18)!important}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th id="hdrA"></th><th id="hdrB"></th><th id="hdrN"></th><th id="hdrC"></th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;
    document.getElementById('hdrA').textContent = data.gA;
    document.getElementById('hdrB').textContent = data.gB;
    document.getElementById('hdrN').textContent = 'NCA: P(' + data.gA + ')';
    document.getElementById('hdrC').textContent = 'CatBoost: P(' + data.gA + ')'; /* ==== CATTAG */

    function barCellLabeled(valRaw,label,valInt,clsOverride){
      const cls=clsOverride || 'blue';
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function formatVal(x){return (x==null||isNaN(x))?'':Number(x).toFixed(2);}

    function childTableHTML(ticker){
      // (unchanged)
      // ...
      return `<table class="child-table">
        <colgroup><col class="col-var"/><col class="col-val"/><col class="col-a"/><col class="col-b"/><col class="col-da"/><col class="col-db"/></colgroup>
        <thead><tr>
          <th class="col-var">Variable</th>
          <th class="col-val">Value</th>
          <th class="col-a">${data.gA} center</th>
          <th class="col-b">${data.gB} center</th>
          <th class="col-da">Δ vs ${data.gA}</th>
          <th class="col-db">Δ vs ${data.gB}</th>
        </tr></thead>
        <tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCellLabeled(row.A_val_raw,row.A_label,row.A_val_int,'blue')},
          {data:null, render:(row)=>barCellLabeled(row.B_val_raw,row.B_label,row.B_val_int,'red')},
          {data:null, render:(row)=>barCellLabeled(row.NCA_raw,'NCA',row.NCA_int,'green')},
          {data:null, render:(row)=>barCellLabeled(row.CAT_raw,'CatBoost',row.CAT_int,'purple')} /* ==== CATTAG */
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

# ============================== Alignment exports (CSV full + Markdown compact) ==============================
import math

if summary_rows:
    # ---------- Markdown (compact summary) ----------
    df_align_md = pd.DataFrame(summary_rows)[
        ["Ticker", "A_label", "A_val_int", "B_label", "B_val_int", "NCA_int", "CAT_int"]  # ==== CATTAG
    ].rename(
        columns={
            "A_label": "A group",
            "A_val_int": "A (%) — Median centers",
            "B_label": "B group",
            "B_val_int": "B (%) — Median centers",
            "NCA_int": "NCA (%)",
            "CAT_int": "CatBoost (%)",  # ==== CATTAG
        }
    )

    # (unchanged: _df_to_markdown_simple)

    # ---------- CSV (full with child rows) ----------
    full_rows = []
    sum_by_ticker = {s["Ticker"]: s for s in summary_rows}

    for tkr, rows in detail_map.items():
        s = sum_by_ticker.get(tkr, {})
        section = ""
        for r in rows:
            if r.get("__group__"):
                full_rows.append({
                    "Ticker": tkr,
                    "Section": r["__group__"],
                    "Variable": "",
                    "Value": "",
                    "A center": "",
                    "B center": "",
                    "Δ vs A": "",
                    "Δ vs B": "",
                    "σ(A)": "",
                    "σ(B)": "",
                    "Is core": "",
                    "A group": s.get("A_label", ""),
                    "B group": s.get("B_label", ""),
                    "A (%) — Median centers": s.get("A_val_int", ""),
                    "B (%) — Median centers": s.get("B_val_int", ""),
                    "NCA (%)": s.get("NCA_int", ""),
                    "CatBoost (%)": s.get("CAT_int", ""),   # ==== CATTAG
                })
                continue

            full_rows.append({
                "Ticker": tkr,
                "Section": section,
                "Variable": r.get("Variable", ""),
                "Value": ("" if pd.isna(r.get("Value")) else r.get("Value")),
                "A center": ("" if pd.isna(r.get("A")) else r.get("A")),
                "B center": ("" if pd.isna(r.get("B")) else r.get("B")),
                "Δ vs A": ("" if (r.get("d_vs_A") is None or pd.isna(r.get("d_vs_A"))) else r.get("d_vs_A")),
                "Δ vs B": ("" if (r.get("d_vs_B") is None or pd.isna(r.get("d_vs_B"))) else r.get("d_vs_B")),
                "σ(A)": ("" if (r.get("sA") is None or pd.isna(r.get("sA"))) else r.get("sA")),
                "σ(B)": ("" if (r.get("sB") is None or pd.isna(r.get("sB"))) else r.get("sB")),
                "Is core": bool(r.get("is_core", False)),
                "A group": s.get("A_label", ""),
                "B group": s.get("B_label", ""),
                "A (%) — Median centers": s.get("A_val_int", ""),
                "B (%) — Median centers": s.get("B_val_int", ""),
                "NCA (%)": s.get("NCA_int", ""),
                "CatBoost (%)": s.get("CAT_int", ""),     # ==== CATTAG
            })

    df_align_csv_full = pd.DataFrame(full_rows)

    # (unchanged: pretty formatting helpers)

    col_order = [
        "Ticker", "Section", "Variable",
        "Value", "A center", "B center", "Δ vs A", "Δ vs B", "σ(A)", "σ(B)", "Is core",
        "A group", "B group",
        "A (%) — Median centers", "B (%) — Median centers", "NCA (%)", "CatBoost (%)",  # ==== CATTAG
    ]
    df_align_csv_pretty = df_align_csv_full[[c for c in col_order if c in df_align_csv_full.columns]]

    # (unchanged: two download buttons)

# ============================== Distributions across Gain% cutoffs ==============================
import altair as alt

st.markdown("---")
st.subheader("Distributions across Gain% cutoffs")

# (unchanged down to the loop producing As, Bs, Ns)
# We add CatBoost medians as Cs.

# inside the thresholds for-loop (unchanged code above):
#   ...
#   Ns.append(n)
#   >>> add:
#   pC = _cat_predict_proba(row, nca_model2 if False else _train_catboost(df_split, gA2, gB2, var_all))  # we will NOT retrain every row
# We won't retrain per-row; instead train once per cutoff like NCA:

# ==== CATTAG (replace the body that builds As/Bs/Ns with CatBoost too)
        for thr_val in gain_choices:
            df_split, gA2, gB2 = _make_split(base_df, float(thr_val), mode)
            med_tbl2, mad_tbl2 = _summaries(df_split, var_all, "__Group__")
            if med_tbl2.empty or med_tbl2.shape[1] < 2:
                continue

            cols2 = list(med_tbl2.columns)
            if (gA2 in cols2) and (gB2 in cols2):
                med_tbl2 = med_tbl2[[gA2, gB2]]
                mad_tbl2 = mad_tbl2.reindex(index=med_tbl2.index)[[gA2, gB2]]
            else:
                top2 = df_split["__Group__"].value_counts().index[:2].tolist()
                if len(top2) < 2:
                    continue
                gA2, gB2 = top2[0], top2[1]
                med_tbl2 = med_tbl2[[gA2, gB2]]
                mad_tbl2 = mad_tbl2.reindex(index=med_tbl2.index)[[gA2, gB2]]

            nca_model2 = _train_nca_or_lda(df_split, gA2, gB2, var_all) or {}
            cat_model2 = _train_catboost(df_split, gA2, gB2, var_all) or {}

            As, Bs, Ns, Cs = [], [], [], []
            for row in rows_for_dist:
                counts2 = _compute_alignment_counts_weighted(
                    stock_row=row,
                    centers_tbl=med_tbl2,
                    var_core=ss.var_core,
                    var_mod=ss.var_moderate,
                    w_core=1.0, w_mod=0.5, tie_mode="split",
                )
                a = counts2.get("A_pct_raw", np.nan) if counts2 else np.nan
                b = counts2.get("B_pct_raw", np.nan) if counts2 else np.nan

                pA = _nca_predict_proba(row, nca_model2)
                pC = _cat_predict_proba(row, cat_model2)

                Ns.append((float(pA)*100.0) if np.isfinite(pA) else np.nan)
                Cs.append((float(pC)*100.0) if np.isfinite(pC) else np.nan)
                As.append(a); Bs.append(b)

            thr_labels.append(int(thr_val))
            series_A_med.append(float(np.nanmedian(As)) if len(As) else np.nan)
            series_B_med.append(float(np.nanmedian(Bs)) if len(Bs) else np.nan)
            series_N_med.append(float(np.nanmedian(Ns)) if len(Ns) else np.nan)
            # new CatBoost median series
            # we define it here:
            # (Initialize series_C_med=[] before loop)
            # (Append below)
# ==== END CATTAG

# remember to define series_C_med before the for loop:
# series_C_med = []   # ==== CATTAG (put this alongside series_A_med/B/N)

# after computing within the for-loop, append:
# series_C_med.append(float(np.nanmedian(Cs)) if len(Cs) else np.nan)   # ==== CATTAG

# and when building the dataframe:
# ==== CATTAG add purple CatBoost series
            labA = f"{gA} (Median centers)"
            labB = f"{gB} (Median centers)"
            labN = f"NCA: P({gA})"
            labC = f"CatBoost: P({gA})"

            dist_df = pd.DataFrame({
                "GainCutoff_%": thr_labels,
                labA: series_A_med,
                labB: series_B_med,
                labN: series_N_med,
                labC: series_C_med,   # new series
            })
            df_long = dist_df.melt(id_vars="GainCutoff_%", var_name="Series", value_name="Value")

            color_domain = [labA, labB, labN, labC]
            color_range  = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]  # blue, red, green, purple

            chart = (
                alt.Chart(df_long)
                .mark_bar()
                .encode(
                    x=alt.X("GainCutoff_%:O", title="Gain% cutoff"),
                    y=alt.Y("Value:Q", title="Median across selected stocks (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
                    xOffset="Series:N",
                    tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

# ============================== Distribution chart export (PNG via Matplotlib fallback) ==============================
# (only extend color_map with purple CatBoost)                                  # ==== CATTAG
color_map = {
    f"{gA} (Median centers)": "#3b82f6",   # blue
    f"{gB} (Median centers)": "#ef4444",   # red
    f"NCA: P({gA})": "#10b981",            # green
    f"CatBoost: P({gA})": "#8b5cf6",       # purple
}

# ============================== Distribution chart export (HTML, no extra deps) ==============================
# (unchanged; it uses chart’s spec which already carries the purple series)
