# app.py — RF→NCA similarity with simple bars (Streamlit-native)
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Optional

# ===== SciKit-Learn deps =====
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
except ModuleNotFoundError:
    st.error("scikit-learn not installed. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

st.set_page_config(page_title="Similarity — NCA+kNN (FT=1 blue / FT=0 red)", layout="wide")
st.title("Similarity — NCA + kernel-kNN (FT=1 vs FT=0)")

ss = st.session_state
ss.setdefault("model", {})   # scaler, nca, X_learned, y, meta
ss.setdefault("rows", [])    # added query rows

# =============== Config ===============
FACE_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"
]

# =============== Helpers ===============
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return (s.replace("%","").replace("$","")
              .replace("(","").replace(")","")
              .replace("’","").replace("'",""))

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}
    nm = {c: _norm(c) for c in cols}
    # exact (case-insensitive)
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc: return c
    # normalized exact
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n: return c
    # normalized contains
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None

def _to_float(s: Any) -> float:
    if pd.isna(s): return np.nan
    try:
        s = str(s).strip().replace(" ", "")
        if "," in s and "." not in s: s = s.replace(",", ".")
        else: s = s.replace(",", "")
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
    return float(x) if x in (0,1) else np.nan

def _map_numeric(dfout: pd.DataFrame, raw: pd.DataFrame, name: str, candidates: List[str]) -> None:
    src = _pick(raw, candidates)
    if src is not None:
        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

def _guess_ft_column(raw: pd.DataFrame) -> Optional[str]:
    poss = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
    if poss: return poss[0]
    for c in raw.columns:
        ser = pd.Series(raw[c]).dropna().astype(str).str.lower()
        if len(ser) and ser.isin(["0","1","true","false","yes","no"]).all():
            return c
        try:
            vals = pd.to_numeric(ser, errors="coerce").dropna()
            if not vals.empty and ((vals == 0) | (vals == 1)).mean() > 0.95:
                return c
        except Exception:
            pass
    return None

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

# =============== Build Model (NCA) ===============
def _build_nca(df: pd.DataFrame):
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

    nca = NeighborhoodComponentsAnalysis(
        random_state=42,
        max_iter=1000,
        tol=1e-5
    ).fit(Xs, y)
    Xz = nca.transform(Xs)

    # optional fast kNN reference (not required for weighting, but useful for quick neighbors)
    knn = KNeighborsClassifier(n_neighbors=50, weights="distance", metric="euclidean").fit(Xz, y)

    # Meta info for debugging
    meta_cols = []
    if "Ticker" in df.columns: meta_cols.append("Ticker")
    if "TickerDB" in df.columns: meta_cols.append("TickerDB")
    if "Max_Push_Daily_%" in df.columns: meta_cols.append("Max_Push_Daily_%")
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=range(len(df_face)))
    meta["FT01"] = y

    return {"scaler": scaler, "nca": nca, "X_learned": Xz, "y": y, "knn": knn, "df_meta": meta}

# =============== Similarity via NCA space ===============
def _elbow_kstar(sim: np.ndarray, max_rank=30, k_min=3, k_max=25) -> int:
    if sim.size <= k_min:
        return max(1, sim.size)
    upto = min(max_rank, sim.size - 1)
    if upto < 2:
        return min(k_max, max(k_min, sim.size))
    gaps = sim[:upto] - sim[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

def _nca_similarity(model: Dict[str, Any], x_vec: np.ndarray, k_max=50):
    scaler, nca, Xz, y = model["scaler"], model["nca"], model["X_learned"], model["y"]
    xq = scaler.transform(x_vec.reshape(1, -1))
    zq = nca.transform(xq).ravel()

    d = np.linalg.norm(Xz - zq[None, :], axis=1)          # distances
    order = np.argsort(d)
    d_sorted = d[order]
    y_sorted = y[order]

    # adaptive bandwidth from top-20 neighbors
    kk = min(20, len(d_sorted))
    if kk == 0:
        return 0.0, 0.0, 0, order[:0], np.array([])
    bw = np.median(d_sorted[:kk])
    if not np.isfinite(bw) or bw <= 1e-12:
        bw = np.mean(d_sorted[:kk]) + 1e-6

    # Gaussian kernel weights
    w = np.exp(- (d_sorted / bw) ** 2)

    # convert distance to similarity in [0,1] for elbow (scale by top-kk max)
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

# =============== Upload / Build ===============
st.subheader("Upload Database")
upl = st.file_uploader("Upload .xlsx", type=["xlsx"])

if st.button("Build NCA model", use_container_width=True):
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

        col_group = _guess_ft_column(raw)
        if col_group is None:
            st.error("Could not detect FT (0/1) column. Name it like FT / FT01 / Group / Label, or boolean-ish 0/1.")
            st.stop()

        df = pd.DataFrame()
        df["GroupRaw"] = raw[col_group]

        # Map required fields
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

        # Catalyst -> binary float
        cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
        if cand_catalyst:
            df["Catalyst"] = pd.Series(raw[cand_catalyst]).map(_safe_to_binary_float)

        # derive PM$Vol/MC_% from MarketCap or MC_PM_Max_M
        basis = "MC_PM_Max_M" if ("MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any()) else "MarketCap_M$"
        if {"PM_$Vol_M$", basis}.issubset(df.columns):
            num = pd.to_numeric(df["PM_$Vol_M$"], errors="coerce")
            den = pd.to_numeric(df[basis], errors="coerce").replace(0, np.nan)
            df["PM$Vol/MC_%"] = (num / den * 100.0).replace([np.inf, -np.inf], np.nan)

        # convert fractioned % (<=2 median) to %
        for c in ["Gap_%", "Max_Pull_PM_%"]:
            if c in df.columns:
                ser = pd.to_numeric(df[c], errors="coerce")
                df[c] = ser * 100.0 if ser.dropna().median() <= 2 else ser

        # FT labels
        df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
        df = df[df["FT01"].isin([0,1])].copy()

        # Optional passthrough ticker
        tcol = _pick(raw, ["ticker","symbol","name"])
        if tcol is not None:
            df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

        ss.model = _build_nca(df)
        if not ss.model:
            st.stop()
        st.success(f"NCA model ready with {ss.model['X_learned'].shape[0]} rows.")
    except Exception as e:
        st.error("Loading/processing failed.")
        st.exception(e)

# =============== Add Stock ===============
st.markdown("---")
st.subheader("Add Stock (9 vars)")
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
    submit = st.form_submit_button("Add", use_container_width=True)

if submit:
    ss.rows.append({
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
    })
    st.success(f"Saved {ticker or '—'}")

# =============== Summaries (simple bars) ===============
st.subheader("Similarity — FT shares (kernel-weighted, learned metric)")

def compute_summaries_and_neighbors():
    if not ss.model or not ss.rows:
        return pd.DataFrame(columns=["Ticker","FT1_%","FT0_%","K*","n_show"]), {}
    model = ss.model
    out = []
    neigh = {}
    for row in ss.rows:
        tkr = row.get("Ticker","—")
        x = _row_to_face(row)
        if x is None:
            out.append([tkr, 0, 0, 0, 0]); neigh[tkr] = []
            continue
        p1, p0, K, idx_sel, w_sel = _nca_similarity(model, x, k_max=50)
        out.append([tkr, int(round(p1)), int(round(p0)), K, len(idx_sel)])
        # build a tiny neighbor table (top 10 by weight)
        meta = model.get("df_meta", pd.DataFrame())
        rows = []
        if len(idx_sel):
            weights = w_sel
            top_idx = np.argsort(-weights)[:10]
            for rnk, pos in enumerate(top_idx, 1):
                idx = int(idx_sel[pos])
                wt = float(weights[pos])
                rec = {"#": rnk, "Weight": wt}
                if not meta.empty and 0 <= idx < len(meta):
                    m = meta.iloc[idx]
                    rec["FT"] = int(m.get("FT01", np.nan)) if pd.notna(m.get("FT01", np.nan)) else None
                    rec["Ticker"] = m.get("TickerDB", m.get("Ticker", ""))
                    mpd = m.get("Max_Push_Daily_%", None)
                    rec["MaxPushDaily_%"] = float(mpd) if pd.notna(mpd) else None
                rows.append(rec)
        neigh[tkr] = rows
    return pd.DataFrame(out, columns=["Ticker","FT1_%","FT0_%","K*","n_show"]), neigh

df_sum, neighbors = compute_summaries_and_neighbors()

if df_sum.empty:
    st.info("Build the NCA model and add at least one stock above.")
else:
    st.data_editor(
        df_sum,
        hide_index=True,
        column_config={
            "FT1_%": st.column_config.ProgressColumn("FT=1 share", min_value=0, max_value=100, format="%d%%"),
            "FT0_%": st.column_config.ProgressColumn("FT=0 share", min_value=0, max_value=100, format="%d%%"),
            "K*": st.column_config.NumberColumn("K*", format="%d"),
            "n_show": st.column_config.NumberColumn("Shown Neigh.", format="%d"),
        },
        use_container_width=True,
        disabled=True,
        height=min(420, 80 + 35*max(1, len(df_sum)))
    )

    # optional neighbor peek
    with st.expander("Neighbors (top-10 by kernel weight)"):
        for tkr, rows in neighbors.items():
            st.markdown(f"**{tkr}**")
            if not rows:
                st.write("—")
            else:
                df_n = pd.DataFrame(rows)
                st.dataframe(df_n, use_container_width=True)

# =============== Delete control ===============
names = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
opts, seen = [], set()
for t in names:
    if t and t not in seen:
        opts.append(t); seen.add(t)
c1, c2 = st.columns([4,1])
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
