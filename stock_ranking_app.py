# stock_ranking_app.py — Add-Stock + Alignment (Distributions of Added Stocks Only)
# - Streamlined UI, High-Sensitivity CatBoost Model (smoothed).
# - Absolute probability chart with optional 68% bootstrap CI ribbons.
# - Robust "Centers" alignment via MAD-weighted distances.
# - Percent normalizer, light winsorization, feature-audit warnings, CSV export.

import streamlit as st
import pandas as pd
import numpy as np
import re, json, io, hashlib
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from catboost import CatBoostClassifier

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rows", [])

# ============================== Unified variables ==============================
UNIFIED_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
    "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
]
ALLOWED_LIVE_FEATURES = UNIFIED_VARS[:]
EXCLUDE_FOR_NCA = []

# ============================== Helpers ==============================
def _norm(s: str) -> str:
    v = re.sub(r"\s+", " ", str(s).strip().lower())
    v = v.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")
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

def _to_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        s = str(x).strip().replace("%","").replace("$","").replace(",","")
        if not s: return np.nan
        return float(s)
    except Exception:
        return np.nan

def _as_percent(s: pd.Series) -> pd.Series:
    """Convert fractional percentages to % once; leave already-% columns as-is."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0: return s
    # Heuristic: if 80% of finite values ≤ 1.5, we treat it as fraction.
    finite = s[np.isfinite(s)]
    cutoff = np.nanquantile((finite <= 1.5).astype(float), 0.8) if len(finite) else 0.0
    is_fraction_like = (finite <= 1.5).mean() >= 0.8 if len(finite) else False
    return s * 100.0 if is_fraction_like else s

def _winsorize_col(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 5: return s
    ql, qh = np.nanquantile(s, [lo, hi])
    return s.clip(ql, qh)

# ============================== Upload / Build ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

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
            raw, sel_sheet, all_sheets = _load_sheet(file_bytes)
            df = pd.DataFrame()

            def add_num(dfout, name, src_candidates):
                src = _pick(raw, src_candidates)
                if src: dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

            add_num(df, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            add_num(df, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            add_num(df, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
            add_num(df, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            add_num(df, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            add_num(df, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            def _to_binary_local(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1.0
                if sv in {"0","false","no","n","f"}: return 0.0
                try:
                    fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                except: return np.nan
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

            # Deriveds
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # Percent columns normalize once
            for pct_col in ("Gap_%","PM_Vol_%","Max_Pull_PM_%"):
                if pct_col in df.columns:
                    df[pct_col] = _as_percent(df[pct_col])

            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            df["Max_Push_Daily_%"] = (
                _as_percent(pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce"))
                if pmh_col is not None else np.nan
            )

            keep_cols = set(UNIFIED_VARS + ["Max_Push_Daily_%"])
            df = df[[c for c in df.columns if c in keep_cols]].copy()

            ss.base_df = df
            st.success(f"Loaded “{sel_sheet}”. Base ready.")
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
    submitted = st.form_submit_button("Add", use_container_width=True)

if submitted and ticker:
    row = {
        "Ticker": ticker,
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Gap_%": gap_pct,
        "ATR_$": atr_usd,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,
        "FR_x": (pm_vol / float_pm) if float_pm > 0 else np.nan,
        "PM$Vol/MC_%": (pm_dol / mc_pmmax * 100.0) if mc_pmmax > 0 else np.nan,
        "Max_Pull_PM_%": max_pull_pm,
        "RVOL_Max_PM_cum": rvol_pm_cum,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "CatalystYN": catalyst_yn,
    }
    ss.rows.append(row)
    st.success(f"Saved {ticker}.")

# ============================== Isotonic helpers & Model Training ==============================
def _pav_isotonic(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0: return [], []
    order = np.argsort(x); x = x[order]; y = y[order]
    blocks = [[y[0], 1.0]]
    for i in range(1, len(y)):
        blocks.append([y[i], 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            s1, n1 = blocks.pop()
            s0, n0 = blocks.pop()
            blocks.append([(s0*n0 + s1*n1)/(n0+n1), n0+n1])
    bx = []; by = []; i = 0
    for mean, n in blocks:
        by.extend([mean]*int(n))
        bx.extend([x[i + j] for j in range(int(n))]); i += int(n)
    return bx, by

def _iso_predict(bx, by, xq):
    bx = np.asarray(bx, dtype=float); by = np.asarray(by, dtype=float); xq = np.asarray(xq, dtype=float)
    if bx.size < 2: return np.full_like(xq, np.nan, dtype=float)
    order = np.argsort(bx); bx = bx[order]; by = by[order]
    out = np.empty_like(xq, dtype=float)
    for i, xv in enumerate(xq):
        k = np.searchsorted(bx, xv, side="right") - 1
        k = np.clip(k, 0, len(bx)-1)
        out[i] = by[k]
    return out

def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}
    # Light winsorization
    for f in feats:
        df2[f] = _winsorize_col(df2[f])
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    good_cols = []
    for c in feats:
        col = Xdf[c].values
        col = col[np.isfinite(col)]
        if col.size == 0: continue
        if np.nanstd(col) < 1e-9: continue
        good_cols.append(c)
    feats = good_cols
    if not feats: return {}
    X = df2[feats].apply(pd.to_numeric, errors="coerce").values
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]; y = y[mask]
    if X.shape[0] < 20 or np.unique(y).size < 2: return {}
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs = (X - mu) / sd
    used = "lda"; w_vec = None; components = None
    try:
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42, max_iter=400)
        z = nca.fit_transform(Xs, y).ravel()
        used = "nca"; components = nca.components_
    except Exception:
        X0 = Xs[y==0]; X1 = Xs[y==1]
        if X0.shape[0] < 2 or X1.shape[0] < 2: return {}
        m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
        S0 = np.cov(X0, rowvar=False); S1 = np.cov(X1, rowvar=False)
        Sw = S0 + S1 + 1e-3*np.eye(Xs.shape[1])
        w_vec = np.linalg.solve(Sw, (m1 - m0))
        w_vec = w_vec / (np.linalg.norm(w_vec) + 1e-12)
        z = (Xs @ w_vec)
    if np.nanmean(z[y==1]) < np.nanmean(z[y==0]):
        z = -z
        if w_vec is not None: w_vec = -w_vec
        if components is not None: components = -components
    zf = z[np.isfinite(z)]; yf = y[np.isfinite(z)]
    iso_bx, iso_by = np.array([]), np.array([]); platt_params = None
    if zf.size >= 8 and np.unique(zf).size >= 3:
        bx, by = _pav_isotonic(zf, yf.astype(float))
        if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
    if iso_bx.size < 2:
        z0 = zf[yf==0]; z1 = zf[yf==1]
        if z0.size and z1.size:
            m0, m1 = float(np.mean(z0)), float(np.mean(z1))
            s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
            m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
            platt_params = (m, k)
    return {"ok": True, "kind": used, "feats": feats, "mu": mu.tolist(), "sd": sd.tolist(), "w_vec": (w_vec.tolist() if w_vec is not None else None), "components": (components.tolist() if components is not None else None), "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": (platt_params if platt_params is not None else None), "gA": gA_label, "gB": gB_label}

def _nca_predict_proba_row(xrow: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = []
    for f in feats:
        v = pd.to_numeric(xrow.get(f), errors="coerce")
        v = float(v) if pd.notna(v) else np.nan
        if not np.isfinite(v): return np.nan
        vals.append(v)
    x = np.array(vals, dtype=float)
    mu = np.array(model["mu"], dtype=float)
    sd = np.array(model["sd"], dtype=float); sd[sd==0] = 1.0
    xs = (x - mu) / sd
    if model["kind"] == "lda":
        w = np.array(model.get("w_vec"), dtype=float)
        if w is None or not np.isfinite(w).all(): return np.nan
        z = float(xs @ w)
    else:
        comp = model.get("components")
        if comp is None: return np.nan
        w = np.array(comp, dtype=float).ravel()
        if w.size != xs.size: return np.nan
        z = float(xs @ w)
    if not np.isfinite(z): return np.nan
    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        if not pl: return np.nan
        m, k = pl
        pA = 1.0 / (1.0 + np.exp(-k*(z - m)))
    return float(np.clip(pA, 0.0, 1.0))

def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}
    # Light winsorization
    for f in feats:
        df2[f] = _winsorize_col(df2[f])
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask_finite = np.isfinite(Xdf.values).all(axis=1)
    Xdf = Xdf.loc[mask_finite]; y = y[mask_finite]
    n = len(y)
    if n < 40 or np.unique(y).size < 2: return {}
    X_all = Xdf.values.astype(np.float32, copy=False)
    y_all = y.astype(np.int32, copy=False)
    from sklearn.model_selection import StratifiedShuffleSplit
    test_size = 0.2 if n >= 100 else max(0.15, min(0.25, 20 / n))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y_all))
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx], y_all[va_idx]
    def _has_both_classes(arr): return np.unique(arr).size == 2
    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)

    # --- Smoothed CatBoost ---
    params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=400,
        learning_rate=0.04,
        depth=3,
        l2_leaf_reg=12,
        border_count=254,
        bootstrap_type="Bayesian",
        bagging_temperature=0.6,
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    params.update(dict(od_type=("Iter" if eval_ok else "None"), od_wait=60 if eval_ok else None))
    model = CatBoostClassifier(**{k:v for k,v in params.items() if v is not None})
    try:
        if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else:       model.fit(X_all, y_all)
    except Exception:
        try:
            model = CatBoostClassifier(**{**params, "od_type":"None"})
            model.fit(X_all, y_all)
            eval_ok = False
        except Exception:
            return {}

    iso_bx = np.array([]); iso_by = np.array([]); platt = None
    try:
        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            if np.unique(p_raw).size >= 3 and _has_both_classes(yva):
                bx, by = _pav_isotonic(p_raw, yva.astype(float))
                if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
            if iso_bx.size < 2:
                z0 = p_raw[yva==0]; z1 = p_raw[yva==1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                    s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                    m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                    platt = (m, k)
    except Exception: pass

    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label, "cb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

def _cat_predict_proba_row(xrow: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = []
    for f in feats:
        v = pd.to_numeric(xrow.get(f), errors="coerce")
        v = float(v) if pd.notna(v) else np.nan
        if not np.isfinite(v): return np.nan
        vals.append(v)
    X = np.array(vals, dtype=float).reshape(1, -1)
    try:
        cb = model.get("cb")
        if cb is None: return np.nan
        z = float(cb.predict_proba(X)[0, 1])
    except Exception: return np.nan
    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        pA = z if not pl else 1.0 / (1.0 + np.exp(-pl[1]*(z - pl[0])))
    return float(np.clip(pA, 0.0, 1.0))

# ---------- Robust Centers (MAD-weighted) ----------
def _compute_alignment_median_centers(stock_row: dict, centers_tbl: pd.DataFrame, df_split: pd.DataFrame) -> dict:
    # centers_tbl: features x [gA, gB] (medians)
    if centers_tbl is None or centers_tbl.empty or len(centers_tbl.columns) != 2:
        return {}
    gA_, gB_ = list(centers_tbl.columns)
    feats = centers_tbl.index.tolist()

    X = []
    for f in feats:
        xv = pd.to_numeric(stock_row.get(f), errors="coerce")
        if not np.isfinite(xv): continue
        vA = centers_tbl.at[f, gA_]; vB = centers_tbl.at[f, gB_]
        if pd.isna(vA) or pd.isna(vB): continue

        sA = df_split.loc[df_split["__Group__"]==gA_, f].astype(float)
        sB = df_split.loc[df_split["__Group__"]==gB_, f].astype(float)
        madA_f = float(np.nanmedian(np.abs(sA - np.nanmedian(sA)))) if sA.notna().any() else np.nan
        madB_f = float(np.nanmedian(np.abs(sB - np.nanmedian(sB)))) if sB.notna().any() else np.nan

        X.append((f, float(xv), float(vA), float(vB), madA_f, madB_f))

    if not X:
        return {}

    eps = 1e-9
    wA = 0.0; wB = 0.0
    for f, xv, vA, vB, madA_f, madB_f in X:
        zA = abs(xv - vA) / (1.4826*(madA_f if np.isfinite(madA_f) and madA_f>0 else 1.0) + eps)
        zB = abs(xv - vB) / (1.4826*(madB_f if np.isfinite(madB_f) and madB_f>0 else 1.0) + eps)
        wA += 1.0/(1.0 + zA)
        wB += 1.0/(1.0 + zB)

    total = wA + wB
    return {"A_pct": (100.0*wA/total) if total>0 else 0.0,
            "B_pct": (100.0*wB/total) if total>0 else 0.0}

# ============================== Alignment (Distributions for SELECTED added stocks) ==============================
st.markdown("---")
st.subheader("Alignment")

base_df = ss.get("base_df", pd.DataFrame())
if base_df.empty:
    st.warning("Upload your Excel and click **Build model stocks**. Alignment distributions are disabled until then.")
    st.stop()

if not ss.rows:
    st.info("Add at least one stock to compute distributions across cutoffs.")
    st.stop()

all_added_tickers = pd.Series([r.get("Ticker") for r in ss.rows]).dropna().unique().tolist()

if "align_sel_tickers" not in st.session_state:
    st.session_state["align_sel_tickers"] = all_added_tickers[:]

def _clear_selection():
    st.session_state["align_sel_tickers"] = []

def _delete_selected():
    tickers_to_delete = st.session_state.get("align_sel_tickers", [])
    if tickers_to_delete:
        ss.rows[:] = [r for r in ss.rows if r.get("Ticker") not in set(tickers_to_delete)]
        st.session_state["align_sel_tickers"] = []

col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    selected_tickers = st.multiselect(
        "Select stocks to include in the charts below. The delete button will act on this selection.",
        options=all_added_tickers,
        default=st.session_state.get("align_sel_tickers", []),
        key="align_sel_tickers",
        label_visibility="collapsed",
    )
with col2:
    st.button("Clear", use_container_width=True, on_click=_clear_selection, help="Clears the current selection in the box.")
with col3:
    st.button("Delete", use_container_width=True, on_click=_delete_selected, help="Deletes the stocks currently selected in the box.", disabled=not selected_tickers)

if not selected_tickers:
    st.info("No stocks selected. Pick at least one added ticker to display the chart.")
    st.stop()

if "Max_Push_Daily_%" not in base_df.columns:
    st.error("Your DB is missing column: Max_Push_Daily_% (Max Push Daily (%) as %).")
    st.stop()

var_all = [v for v in UNIFIED_VARS if v in base_df.columns]
if not var_all:
    st.error("No usable numeric features found after loading. Ensure your Excel has mapped numeric columns.")
    st.stop()

# Controls
st.caption("Tip: thresholds with too few observations are skipped to avoid noisy models.")
gain_cutoffs = list(range(25, 301, 25))
show_ci = st.checkbox("Show 68% bootstrap CI (medians across selected tickers)", value=True)
n_boot = st.slider("Bootstrap resamples", min_value=100, max_value=800, value=200, step=50, help="Used only if CI is enabled.")

def _make_split(df_base: pd.DataFrame, thr_val: float):
    df_tmp = df_base.copy()
    # Light winsorization per feature used (stabilizes centers & models)
    for f in [c for c in df_tmp.columns if c in ALLOWED_LIVE_FEATURES]:
        df_tmp[f] = _winsorize_col(df_tmp[f])
    df_tmp["__Group__"] = np.where(
        pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val,
        f"≥{int(thr_val)}%", "Rest"
    )
    gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
    return df_tmp, gA_, gB_

added_df = pd.DataFrame([r for r in ss.rows if r.get("Ticker") in set(selected_tickers)])

thr_labels = []
series_A_med, series_B_med, series_N_med, series_C_med = [], [], [], []
ci_A_lo, ci_A_hi, ci_C_lo, ci_C_hi, ci_N_lo, ci_N_hi = [], [], [], [], [], []
effective_cutoffs = []

with st.spinner("Calculating distributions across all cutoffs..."):
    for thr_val in gain_cutoffs:
        df_split, gA, gB = _make_split(base_df, float(thr_val))
        vc = df_split["__Group__"].value_counts()
        if (vc.get(gA, 0) < 10) or (vc.get(gB, 0) < 10):
            continue

        # Train models
        nca_model = _train_nca_or_lda(df_split, gA, gB, var_all) or {}
        cat_model = _train_catboost_once(df_split, gA, gB, var_all) or {}

        # Centers
        features_for_centers = [f for f in ALLOWED_LIVE_FEATURES if f in df_split.columns]
        centers_tbl = df_split.groupby("__Group__")[features_for_centers].median().T
        if gA not in centers_tbl.columns or gB not in centers_tbl.columns:
            continue
        centers_tbl = centers_tbl[[gA, gB]]

        # Feature audit (what dropped)
        req_feats = sorted(set(nca_model.get("feats", []) + cat_model.get("feats", []) + features_for_centers))
        if not req_feats: continue
        intended = set([f for f in UNIFIED_VARS if f in df_split.columns])
        actually_used = set(req_feats)
        missing = sorted(list(intended - actually_used))
        if missing:
            st.warning(f"[{gA} split] Excluded (data/variance): {', '.join(missing)}")

        if added_df.empty: continue
        Xadd = added_df[req_feats].apply(pd.to_numeric, errors="coerce")
        mask = np.isfinite(Xadd.values).all(axis=1)
        pred_rows = added_df.loc[mask].to_dict(orient="records")
        if len(pred_rows) == 0: continue

        pN, pC, pA_centers, pB_centers = [], [], [], []
        for r in pred_rows:
            if nca_model:
                p = _nca_predict_proba_row(r, nca_model)
                if np.isfinite(p): pN.append(p * 100.0)
            if cat_model:
                p = _cat_predict_proba_row(r, cat_model)
                if np.isfinite(p): pC.append(p * 100.0)
            center_scores = _compute_alignment_median_centers(r, centers_tbl, df_split)
            if center_scores:
                pA_centers.append(center_scores['A_pct'])
                pB_centers.append(center_scores['B_pct'])

        if not any([len(pN), len(pC), len(pA_centers)]):
            continue

        thr_labels.append(int(thr_val))
        effective_cutoffs.append(int(thr_val))
        series_N_med.append(float(np.nanmedian(pN)) if pN else np.nan)
        series_C_med.append(float(np.nanmedian(pC)) if pC else np.nan)
        series_A_med.append(float(np.nanmedian(pA_centers)) if pA_centers else np.nan)
        series_B_med.append(float(np.nanmedian(pB_centers)) if pB_centers else np.nan)

        # Bootstrap CIs (medians across selected stocks)
        def _boot_ci(vals, qlo=0.16, qhi=0.84):
            if not vals: return (np.nan, np.nan)
            arr = np.asarray(vals, dtype=float)
            if arr.size < 2: return (np.nan, np.nan)
            rng = np.random.default_rng(42)
            meds = []
            for _ in range(n_boot):
                idx = rng.integers(0, arr.size, size=arr.size)
                meds.append(np.nanmedian(arr[idx]))
            return (float(np.nanquantile(meds, qlo)), float(np.nanquantile(meds, qhi)))

        if show_ci:
            lo, hi = _boot_ci(pA_centers); ci_A_lo.append(lo); ci_A_hi.append(hi)
            lo, hi = _boot_ci(pC);         ci_C_lo.append(lo); ci_C_hi.append(hi)
            lo, hi = _boot_ci(pN);         ci_N_lo.append(lo); ci_N_hi.append(hi)
        else:
            ci_A_lo.append(np.nan); ci_A_hi.append(np.nan)
            ci_C_lo.append(np.nan); ci_C_hi.append(np.nan)
            ci_N_lo.append(np.nan); ci_N_hi.append(np.nan)

# --- FINAL, ROBUST VERSION: Function to generate export buttons for any chart ---
def create_export_buttons(df, chart_obj, file_prefix):
    """Generates PNG and HTML download buttons for a given dataframe and Altair chart."""
    png_bytes = b""
    try:
        x_axis_col = df.columns[0]
        pivot = df.pivot(index=x_axis_col, columns="Series", values="Value").sort_index()
        series_names = list(pivot.columns)
        chart_dict = chart_obj.to_dict()
        unique_colors = {}
        try:
            domain = chart_dict['encoding']['color']['scale']['domain']
            range_ = chart_dict['encoding']['color']['scale']['range']
            unique_colors = dict(zip(domain, range_))
        except KeyError:
            pass
        colors = [unique_colors.get(s, "#999999") for s in series_names]
        x_labels = [str(label) for label in pivot.index.tolist()]
        n_groups, n_series = len(x_labels), len(series_names)
        x_pos = np.arange(n_groups)
        bar_width = 0.8 / max(n_series, 1)
        fig, ax = plt.subplots(figsize=(max(7, n_groups * 0.6), 5))
        for i, series_name in enumerate(series_names):
            vals = pivot[series_name].values.astype(float)
            offset = i * bar_width - (n_series - 1) * bar_width / 2
            ax.bar(x_pos + offset, vals, width=bar_width, label=series_name, color=colors[i])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlabel(x_axis_col)
        ax.set_ylabel("Value (%)")
        ax.legend(loc="upper left", frameon=False)
        ax.set_title(chart_obj.title)
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buf.getvalue()
    except Exception:
        pass

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label=f"Download PNG ({file_prefix})",
            data=png_bytes,
            file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
            key=f"dl_png_{file_prefix}",
            disabled=not png_bytes
        )
    with c2:
        spec = chart_obj.to_dict()
        html_template = f'<!doctype html><html><head><meta charset="utf-8"><title>{file_prefix}</title><script src="https://cdn.jsdelivr.net/npm/vega@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script></head><body><div id="vis"></div><script>const spec = {json.dumps(spec)}; vegaEmbed("#vis", spec, {{"actions": True}});</script></body></html>'
        st.download_button(
            label=f"Download HTML ({file_prefix})",
            data=html_template.encode("utf-8"),
            file_name=f"{file_prefix}.html",
            mime="text/html",
            use_container_width=True,
            key=f"dl_html_{file_prefix}"
        )
    with c3:
        # CSV export of the plotted data
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download CSV ({file_prefix})",
            data=csv_bytes,
            file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_csv_{file_prefix}"
        )

if not thr_labels:
    st.info("Not enough data across cutoffs to train models. Try using a larger database.")
else:
    # --- Build dataset for Absolute Probability Chart ---
    data = []
    gA_label = f"≥...% (Median Centers)"
    gB_label = f"Rest (Median Centers)"
    nca_label = f"NCA: P(≥...%)"
    cat_label = f"CatBoost: P(≥...%)"

    # Core series
    for i, thr in enumerate(thr_labels):
        data.append({"GainCutoff_%": thr, "Series": gA_label, "Value": series_A_med[i]})
        data.append({"GainCutoff_%": thr, "Series": gB_label, "Value": series_B_med[i]})
        data.append({"GainCutoff_%": thr, "Series": nca_label, "Value": series_N_med[i]})
        data.append({"GainCutoff_%": thr, "Series": cat_label, "Value": series_C_med[i]})

    df_long = pd.DataFrame(data).dropna(subset=['Value'])

    # CI ribbons (optional): create separate dataframes
    if any(np.isfinite(ci) for ci in ci_C_lo+ci_C_hi+ci_N_lo+ci_N_hi+ci_A_lo+ci_A_hi) and show_ci:
        ci_rows = []
        for i, thr in enumerate(thr_labels):
            if np.isfinite(ci_C_lo[i]) and np.isfinite(ci_C_hi[i]):
                ci_rows.append({"GainCutoff_%": thr, "Series": cat_label, "lo": ci_C_lo[i], "hi": ci_C_hi[i]})
            if np.isfinite(ci_N_lo[i]) and np.isfinite(ci_N_hi[i]):
                ci_rows.append({"GainCutoff_%": thr, "Series": nca_label, "lo": ci_N_lo[i], "hi": ci_N_hi[i]})
            if np.isfinite(ci_A_lo[i]) and np.isfinite(ci_A_hi[i]):
                ci_rows.append({"GainCutoff_%": thr, "Series": gA_label, "lo": ci_A_lo[i], "hi": ci_A_hi[i]})
        df_ci = pd.DataFrame(ci_rows)
    else:
        df_ci = pd.DataFrame(columns=["GainCutoff_%","Series","lo","hi"])

    color_domain = [gA_label, gB_label, nca_label, cat_label]
    color_range  = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]

    base = alt.Chart(df_long).encode(
        x=alt.X("GainCutoff_%:O",
                title=f"Gain% cutoff (step {thr_labels[1]-thr_labels[0] if len(thr_labels)>1 else 25})",
                sort=[str(t) for t in thr_labels]),
        y=alt.Y("Value:Q", title="Median Alignment / P(A) (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="Analysis Series")),
        tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
    )

    bars = base.mark_bar().encode(xOffset="Series:N")

    if not df_ci.empty and show_ci:
        area = alt.Chart(df_ci).mark_area(opacity=0.15).encode(
            x=alt.X("GainCutoff_%:O", sort=[str(t) for t in thr_labels]),
            y="lo:Q",
            y2="hi:Q",
            color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            tooltip=["GainCutoff_%:O","Series:N", alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")]
        )
        chart = (area + bars).properties(height=420, title="Absolute Probability of Reaching Gain Cutoff")
    else:
        chart = bars.properties(height=400, title="Absolute Probability of Reaching Gain Cutoff")

    st.altair_chart(chart, use_container_width=True)

    # --- Chart Export Section for Absolute Probability Chart ---
    create_export_buttons(df_long, chart, "absolute_probability")
