# app.py — RF Similarity (offline-safe renderer)
import streamlit as st
import pandas as pd
import numpy as np
import re, json
from typing import Any, Dict, List, Optional

# ===== RF deps =====
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    st.error("scikit-learn not installed. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

st.set_page_config(page_title="RF Similarity (FT=1 blue / FT=0 red)", layout="wide")
st.title("RF Similarity — FT=1 (blue) vs FT=0 (red)")

# --------------- Session ----------------
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rf_model", {})     # scaler, rf, leaf_train, X_scaled, y, feat_names, df_meta
ss.setdefault("rows", [])         # added queries

# --------------- Config -----------------
FACE_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"
]

# --------------- Helpers ----------------
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return (
        s.replace("%","").replace("$","")
         .replace("(","").replace(")","")
         .replace("’","").replace("'","")
    )

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
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

def _to_float(s: Any) -> float:
    if pd.isna(s): return np.nan
    try:
        s = str(s).strip().replace(" ", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        return float(s)
    except Exception:
        return np.nan

def _safe_to_binary(v: Any) -> Any:
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        return 1 if float(sv) >= 0.5 else 0
    except Exception:
        return np.nan

def _safe_to_binary_float(v: Any) -> float:
    x = _safe_to_binary(v)
    return float(x) if x in (0, 1) else np.nan

def _map_numeric(dfout: pd.DataFrame, raw: pd.DataFrame, name: str, candidates: List[str]) -> None:
    src = _pick(raw, candidates)
    if src is not None:
        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

def _build_rf(df: pd.DataFrame) -> Dict[str, Any]:
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
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )
    rf.fit(Xs, y)
    leaf_train = rf.apply(Xs)  # (n_samples, n_estimators)
    meta_cols = []
    if "Ticker" in df.columns: meta_cols.append("Ticker")
    if "TickerDB" in df.columns: meta_cols.append("TickerDB")
    if "Max_Push_Daily_%" in df.columns: meta_cols.append("Max_Push_Daily_%")
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=range(len(df_face)))
    meta["FT01"] = y
    return {"scaler":scaler,"rf":rf,"leaf_train":leaf_train,"X_scaled":Xs,"y":y,"feat_names":FACE_VARS,"df_meta":meta}

def _row_to_face(row: Dict[str, Any]) -> Optional[np.ndarray]:
    vals = []
    for f in FACE_VARS:
        v = row.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        try:
            vals.append(float(v))
        except Exception:
            return None
    return np.array(vals, dtype=float)

def _rf_proximity(model: Dict[str, Any], x_vec: np.ndarray):
    xs = model["scaler"].transform(x_vec.reshape(1, -1)).ravel()
    leaf_q = model["rf"].apply(xs.reshape(1, -1)).ravel()  # (n_estimators,)
    prox = (model["leaf_train"] == leaf_q).mean(axis=1)    # (n_train,)
    order = np.argsort(-prox)
    return prox[order], order

def _elbow_kstar(prox_sorted: np.ndarray, max_rank=30, k_min=3, k_max=25) -> int:
    if prox_sorted.size <= k_min: return max(1, prox_sorted.size)
    upto = min(max_rank, prox_sorted.size - 1)
    if upto < 2: return min(k_max, max(k_min, prox_sorted.size))
    gaps = prox_sorted[:upto] - prox_sorted[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

# --------------- Upload / Build ----------------
st.subheader("Upload Database")
upl = st.file_uploader("Upload .xlsx", type=["xlsx"])

if st.button("Build RF model", use_container_width=True):
    if not upl:
        st.error("Upload an Excel first.")
        st.stop()
    try:
        try:
            raw = pd.read_excel(upl, engine="openpyxl")
        except Exception:
            raw = pd.read_excel(upl)
        if raw is None or raw.empty:
            st.error("The Excel seems empty or unreadable.")
            st.stop()

        # detect FT
        poss = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
        col_group = poss[0] if poss else None
        if col_group is None:
            for c in raw.columns:
                vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                if len(vals) and vals.isin(["0","1","true","false","yes","no"]).all():
                    col_group = c; break
        if col_group is None:
            st.error("Could not detect FT (0/1) column.")
            st.stop()

        df = pd.DataFrame()
        df["GroupRaw"] = raw[col_group]
        # map needed fields
        _map_numeric(df, raw, "MC_PM_Max_M",    ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
        _map_numeric(df, raw, "Float_PM_Max_M", ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
        _map_numeric(df, raw, "MarketCap_M$",   ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
        _map_numeric(df, raw, "Float_M",        ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
        _map_numeric(df, raw, "Gap_%",          ["gap %","gap%","premarket gap","gap","gap_percent"])
        _map_numeric(df, raw, "ATR_$",          ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
        _map_numeric(df, raw, "PM_Vol_M",       ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
        _map_numeric(df, raw, "PM_$Vol_M$",     ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
        _map_numeric(df, raw, "Max_Pull_PM_%",  ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
        _map_numeric(df, raw, "RVOL_Max_PM_cum",["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

        # Catalyst
        cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
        if cand_catalyst:
            df["Catalyst"] = pd.Series(raw[cand_catalyst]).map(_safe_to_binary_float)

        # derive PM$Vol/MC_% from MarketCap or MC_PM_Max_M
        basis = "MC_PM_Max_M" if ("MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any()) else "MarketCap_M$"
        if {"PM_$Vol_M$", basis}.issubset(df.columns):
            num = pd.to_numeric(df["PM_$Vol_M$"], errors="coerce")
            den = pd.to_numeric(df[basis], errors="coerce").replace(0, np.nan)
            df["PM$Vol/MC_%"] = (num / den * 100.0).replace([np.inf,-np.inf], np.nan)

        # convert fractioned % to %
        for c in ["Gap_%", "Max_Pull_PM_%"]:
            if c in df.columns:
                ser = pd.to_numeric(df[c], errors="coerce")
                df[c] = ser * 100.0 if ser.dropna().median() <= 2 else ser

        # FT labels
        df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
        df = df[df["FT01"].isin([0,1])].copy()

        # passthrough ticker
        tcol = _pick(raw, ["ticker","symbol","name"])
        if tcol is not None:
            df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

        ss.base_df = df
        ss.rf_model = _build_rf(df)
        if not ss.rf_model:
            st.stop()
        st.success(f"RF model ready with {ss.rf_model['X_scaled'].shape[0]} rows.")
    except Exception as e:
        st.error("Loading/processing failed.")
        st.exception(e)

# --------------- Add Stock ----------------
st.markdown("---")
st.subheader("Add Stock to Compare (9 variables)")
with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        ticker      = st.text_input("Ticker", "").strip().upper()
        mc_pmmax    = st.number_input("Premarket Market Cap (M$)", value=0.0, step=0.01, format="%.2f")
        float_pm    = st.number_input("Premarket Float (M)", value=0.0, step=0.01, format="%.2f")
        gap_pct     = st.number_input("Gap %", value=0.0, step=0.01, format="%.2f")
        max_pull_pm = st.number_input("Premarket Max Pullback (%)", value=0.0, step=0.01, format="%.2f")
    with c2:
        atr_usd     = st.number_input("Prior Day ATR ($)", value=0.0, step=0.01, format="%.2f")
        pm_vol      = st.number_input("Premarket Volume (M)", value=0.0, step=0.01, format="%.2f")
        pm_dol      = st.number_input("Premarket Dollar Vol (M$)", value=0.0, step=0.01, format="%.2f")
        rvol_pm_cum = st.number_input("Premarket Max RVOL (cum)", value=0.0, step=0.01, format="%.2f")
    with c3:
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)
    submit = st.form_submit_button("Add & Compare", use_container_width=True)

if submit:
    row = {
        "Ticker": (ticker or "—"),
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "ATR_$": atr_usd,
        "Gap_%": gap_pct,
        "Max_Pull_PM_%": max_pull_pm,
        "PM_Vol_M": pm_vol,
        "PM$Vol/MC_%": ((pm_dol / mc_pmmax) * 100.0) if mc_pmmax > 0 else np.nan,
        "RVOL_Max_PM_cum": rvol_pm_cum,
    }
    ss.rows.append(row)
    st.success(f"Saved {row['Ticker']}")

# --------------- RF Similarity ----------------
st.markdown("### Similarity (Top-K* by elbow) — bars show FT=1 (blue) vs FT=0 (red) share)")

def _summarize_and_neighbors():
    if not ss.rf_model:
        return [], {}
    rf_m = ss.rf_model
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
        K = int(_elbow_kstar(prox, max_rank=30, k_min=3, k_max=min(25, prox.size)))
        sel = order[:K]
        y = rf_m["y"][sel]
        cntA = int((y==1).sum()); cntB = int((y==0).sum()); tot = max(1, cntA+cntB)
        pA = 100.0*cntA/tot; pB = 100.0 - pA
        summaries.append({"Ticker": tkr, "pA":pA,"pB":pB,"iA":int(round(pA)),"iB":int(round(pB)),"cntA":cntA,"cntB":cntB,"K":K})
        meta = rf_m.get("df_meta", pd.DataFrame())
        rows = [{"__group__": f"Top-{K} neighbors by RF proximity"}]
        # Use sorted proximities in rank order
        prox_sorted = np.sort(prox)[::-1]
        for rnk, idx in enumerate(sel, start=1):
            rec = {"#": rnk, "FT": int(rf_m["y"][idx]), "Proximity": float(prox_sorted[rnk-1])}
            if not meta.empty:
                mrow = meta.iloc[idx] if 0 <= idx < len(meta) else {}
                if isinstance(mrow, pd.Series):
                    rec.update({c: mrow.get(c, None) for c in meta.columns})
            rows.append(rec)
        details[tkr] = rows
    return summaries, details

summaries, details = _summarize_and_neighbors()

# --------- OFFLINE renderer (no external JS/CSS) ----------
def _render_bars_offline(summaries, details):
    if not summaries:
        st.info("Add at least one stock (above) to compute similarity.")
        return
    # Show a compact table with inline bars
    def bar_html(pct, is_blue, n):
        pct = 0.0 if pct is None or not np.isfinite(pct) else max(0.0, min(100.0, float(pct)))
        color = "#3b82f6" if is_blue else "#ef4444"
        return f'''
        <div style="display:flex;align-items:center;gap:6px;">
          <div style="width:160px;height:12px;border-radius:8px;background:#eee;overflow:hidden;position:relative;">
            <span style="position:absolute;left:0;top:0;bottom:0;width:{pct:.2f}%;background:{color};"></span>
          </div>
          <div style="font-size:11px;color:#374151;min-width:70px;text-align:center;">{round(pct)}% • n={int(n)}</div>
        </div>'''
    rows_html = ""
    for s in summaries:
        rows_html += f"""
        <tr>
          <td style="padding:8px 6px;">{s['Ticker']}</td>
          <td style="text-align:center;padding:8px 6px;">{bar_html(s['pA'], True, s['cntA'])}</td>
          <td style="text-align:center;padding:8px 6px;">{bar_html(s['pB'], False, s['cntB'])}</td>
          <td style="text-align:right;padding:8px 6px;">{s['K']}</td>
        </tr>"""
        # details rows via <details>
        det = details.get(s["Ticker"], [])
        if det:
            inner_rows = ""
            for r in det:
                if "__group__" in r:
                    inner_rows += f'<tr><td colspan="5" style="background:#f3f4f6;font-weight:600;padding:6px 8px;">{r["__group__"]}</td></tr>'
                else:
                    ft = "FT=1" if r.get("FT",0)==1 else "FT=0"
                    tick = r.get("TickerDB") or r.get("Ticker") or ""
                    mpd = r.get("Max_Push_Daily_%")
                    mpd_txt = f"{mpd:.2f}%" if isinstance(mpd,(int,float)) and np.isfinite(mpd) else ""
                    prox = r.get("Proximity")
                    prox_txt = f"{prox:.4f}" if isinstance(prox,(int,float)) and np.isfinite(prox) else ""
                    inner_rows += f"<tr><td>{r.get('#','')}</td><td>{ft}</td><td style='text-align:right'>{prox_txt}</td><td>{tick}</td><td style='text-align:right'>{mpd_txt}</td></tr>"
            rows_html += f"""
            <tr><td colspan="4" style="padding:0 6px 12px 6px;">
              <details>
                <summary style="cursor:pointer;color:#2563eb;">Show neighbors</summary>
                <div style="margin:8px 0 0 8px;">
                  <table style="width:100%;border-collapse:collapse;">
                    <thead>
                      <tr><th style="text-align:left;padding:4px 6px;">#</th>
                          <th style="text-align:left;padding:4px 6px;">FT</th>
                          <th style="text-align:right;padding:4px 6px;">Proximity</th>
                          <th style="text-align:left;padding:4px 6px;">Ticker</th>
                          <th style="text-align:right;padding:4px 6px;">MaxPushDaily%</th></tr>
                    </thead>
                    <tbody>{inner_rows}</tbody>
                  </table>
                </div>
              </details>
            </td></tr>"""

    html = f"""
    <div style="font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,'Helvetica Neue',sans-serif;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#f9fafb;">
            <th style="text-align:left;padding:8px 6px;">Ticker</th>
            <th style="text-align:center;padding:8px 6px;">FT=1 (blue)</th>
            <th style="text-align:center;padding:8px 6px;">FT=0 (red)</th>
            <th style="text-align:right;padding:8px 6px;">K*</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

_render_bars_offline(summaries, details)

# --------------- Debug (always visible) ----------------
with st.expander("Debug • parsed columns & sample"):
    st.write("Mapped FACE_VARS present:", [c for c in FACE_VARS if c in (ss.base_df.columns if not ss.base_df.empty else [])])
    if not ss.base_df.empty:
        st.write("Base DF shape:", ss.base_df.shape)
        st.dataframe(ss.base_df.head(10))
    if ss.rf_model:
        st.write("Model rows:", ss.rf_model["X_scaled"].shape[0])
        st.write("Added comparisons:", len(ss.rows))

# --------------- Delete control ----------------
names = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
opts, seen = [], set()
for t in names:
    if t and t not in seen:
        opts.append(t)
        seen.add(t)

c1, c2 = st.columns([4, 1])
with c1:
    sel = st.multiselect("", options=opts, default=[], placeholder="Select tickers…", label_visibility="collapsed")
with c2:
    if st.button("Delete", use_container_width=True):
        if sel:
            keep = set(sel)
            ss.rows = [r for r in ss.rows if r.get("Ticker") not in keep]
            st.success(f"Deleted: {', '.join(sel)}")
            st.rerun()
        else:
            st.info("No tickers selected.")
