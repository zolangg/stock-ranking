# stock_ranking_app.py — Add-Stock + Alignment (Distributions of Added Stocks Only)
# - Add Stock form (no table)
# - Alignment shows median P(A) over SELECTED added stocks across Gain% cutoffs (0..600 step 25)
# - Models (NCA, CatBoost & LightGBM) train per cutoff on the UPLOADED DB; predictions are for ADDED stocks only
# - CatBoost and LightGBM are included
# - No daily-volume prediction anywhere
# - Simplified: one unified variables list for all models

import streamlit as st
import pandas as pd
import numpy as np
import re, json
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from catboost import CatBoostClassifier
import lightgbm as lgb

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rows", [])  # user-added stocks (list of dicts)

# ============================== Unified variables ==============================
UNIFIED_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
    "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
]
ALLOWED_LIVE_FEATURES = UNIFIED_VARS[:]
EXCLUDE_FOR_NCA = []

# ============================== Helpers ==============================
# <<< CHANGE: Added a robust name sanitization function
def sanitize_name(name: str) -> str:
    """Replaces special characters with underscores for LightGBM compatibility."""
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)

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

# ============================== Upload / Build ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

@st.cache_data(show_spinner="Loading and processing Excel file...")
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
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
                for c in raw.columns:
                    vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if len(vals) and vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group = c; break
            if col_group is None: st.error("Could not detect FT (0/1) column."); st.stop()

            df = pd.DataFrame()
            df["GroupRaw"] = raw[col_group]
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
                try: fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                except: return np.nan
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)
            for pct_col in ("Gap_%","PM_Vol_%","Max_Pull_PM_%"):
                if pct_col in df.columns: df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce") * 100.0
            def _to_binary(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1
                if sv in {"0","false","no","n","f"}: return 0
                try: return 1 if float(sv) >= 0.5 else 0
                except: return np.nan
            df["FT01"] = df["GroupRaw"].map(_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            df["Max_Push_Daily_%"] = (pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce") * 100.0 if pmh_col is not None else np.nan)
            keep_cols = set(UNIFIED_VARS + ["GroupRaw","FT01","GroupFT","Max_Push_Daily_%"])
            df = df[[c for c in df.columns if c in keep_cols]].copy()

            # <<< CHANGE: Sanitize all column names for compatibility
            sanitized_cols = {col: sanitize_name(col) for col in df.columns}
            df = df.rename(columns=sanitized_cols)
            
            # <<< CHANGE: Update the global variable lists with sanitized names
            global SANITIZED_UNIFIED_VARS, SANITIZED_ALLOWED_LIVE_FEATURES, SANITIZED_EXCLUDE_FOR_NCA
            SANITIZED_UNIFIED_VARS = [sanitize_name(v) for v in UNIFIED_VARS]
            SANITIZED_ALLOWED_LIVE_FEATURES = [sanitize_name(v) for v in ALLOWED_LIVE_FEATURES]
            SANITIZED_EXCLUDE_FOR_NCA = [sanitize_name(v) for v in EXCLUDE_FOR_NCA]

            ss.base_df = df
            st.success(f"Loaded “{sel_sheet}”. Base ready.")
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============================== Add Stock (no table) ==============================
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
    # <<< CHANGE: Sanitize the keys of the newly added row
    sanitized_row = {sanitize_name(key): val for key, val in row.items()}
    ss.rows.append(sanitized_row)
    st.success(f"Saved {ticker}.")

# ============================== Isotonic helpers ==============================
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

# ============================== NCA / LDA ==============================
def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in SANITIZED_ALLOWED_LIVE_FEATURES and f not in SANITIZED_EXCLUDE_FOR_NCA]
    if not feats: return {}
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    good_cols = [c for c in feats if Xdf[c].notna().any() and np.nanstd(Xdf[c].values) > 1e-9]
    if not good_cols: return {}
    feats = good_cols
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
        w_vec /= (np.linalg.norm(w_vec) + 1e-12)
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
    vals = [float(pd.to_numeric(xrow.get(f), errors="coerce")) for f in feats]
    if any(not np.isfinite(v) for v in vals): return np.nan
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
        pA = 1.0 / (1.0 + np.exp(-pl[1]*(z - pl[0])))
    return float(np.clip(pA, 0.0, 1.0))

# ============================== CatBoost ==============================
def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in SANITIZED_ALLOWED_LIVE_FEATURES]
    if not feats: return {}
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask_finite = Xdf.notna().all(axis=1)
    Xdf = Xdf[mask_finite]; y = y[mask_finite]
    n = len(y)
    if n < 40 or np.unique(y).size < 2: return {}
    X_all = Xdf.values.astype(np.float32, copy=False)
    y_all = y.astype(np.int32, copy=False)
    from sklearn.model_selection import StratifiedShuffleSplit
    test_size = 0.2 if n >= 100 else max(0.15, min(0.25, 20 / n))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y_all))
    Xtr, Xva, ytr, yva = X_all[tr_idx], X_all[va_idx], y_all[tr_idx], y_all[va_idx]
    def _has_both_classes(arr): return np.unique(arr).size == 2
    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)
    params = dict(loss_function="Logloss", eval_metric="Logloss", iterations=200, learning_rate=0.04, depth=3, l2_leaf_reg=6, bootstrap_type="Bayesian", bagging_temperature=0.5, auto_class_weights="Balanced", random_seed=42, allow_writing_files=False, verbose=False)
    if eval_ok: params.update(dict(od_type="Iter", od_wait=40))
    else:       params.update(dict(od_type="None"))
    model = CatBoostClassifier(**params)
    try:
        if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else:       model.fit(X_all, y_all)
    except Exception:
        try:
            model = CatBoostClassifier(**{**params, "od_type": "None"}); model.fit(X_all, y_all)
            eval_ok = False
        except Exception: return {}
    iso_bx, iso_by, platt = np.array([]), np.array([]), None
    try:
        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            if np.unique(p_raw).size >= 3 and _has_both_classes(yva):
                bx, by = _pav_isotonic(p_raw, yva.astype(float))
                if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
            if iso_bx.size < 2:
                z0, z1 = p_raw[yva==0], p_raw[yva==1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1)); s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                    m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                    platt = (m, k)
    except Exception: pass
    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label, "cb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

def _cat_predict_proba_row(xrow: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = [float(pd.to_numeric(xrow.get(f), errors="coerce")) for f in feats]
    if any(not np.isfinite(v) for v in vals): return np.nan
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

# ============================== LightGBM ==============================
def _train_lightgbm_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in SANITIZED_ALLOWED_LIVE_FEATURES]
    if not feats: return {}
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask_finite = Xdf.notna().all(axis=1)
    Xdf = Xdf[mask_finite]; y = y[mask_finite]
    n = len(y)
    if n < 40 or np.unique(y).size < 2: return {}
    X_all = Xdf.values.astype(np.float32, copy=False)
    y_all = y.astype(np.int32, copy=False)
    from sklearn.model_selection import StratifiedShuffleSplit
    test_size = 0.2 if n >= 100 else max(0.15, min(0.25, 20 / n))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y_all))
    Xtr, Xva, ytr, yva = X_all[tr_idx], X_all[va_idx], y_all[tr_idx], y_all[va_idx]
    def _has_both_classes(arr): return np.unique(arr).size == 2
    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)

    # <<< CHANGE: Replaced 'is_unbalance' with 'class_weight'
    params = {
        'objective': 'binary', 'metric': 'logloss', 'n_estimators': 200, 'learning_rate': 0.04,
        'num_leaves': 8, 'max_depth': 3, 'reg_lambda': 6, 'colsample_bytree': 0.7, 'subsample': 0.7,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    model = lgb.LGBMClassifier(**params)
    try:
        if eval_ok:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
        else:
            model.fit(X_all, y_all)
    except Exception:
        try: model.fit(X_all, y_all); eval_ok = False
        except Exception: return {}
    iso_bx, iso_by, platt = np.array([]), np.array([]), None
    try:
        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            if np.unique(p_raw).size >= 3 and _has_both_classes(yva):
                bx, by = _pav_isotonic(p_raw, yva.astype(float))
                if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
            if iso_bx.size < 2:
                z0, z1 = p_raw[yva == 0], p_raw[yva == 1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1)); s0, s1 = float(np.std(z0) + 1e-9), float(np.std(z1) + 1e-9)
                    m = 0.5 * (m0 + m1); k = 2.0 / (0.5 * (s0 + s1) + 1e-6)
                    platt = (m, k)
    except Exception: pass
    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label, "lgb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

def _lgb_predict_proba_row(xrow: dict, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = [float(pd.to_numeric(xrow.get(f), errors="coerce")) for f in feats]
    if any(not np.isfinite(v) for v in vals): return np.nan
    X = np.array(vals, dtype=float).reshape(1, -1)
    try:
        lgbm = model.get("lgb")
        if lgbm is None: return np.nan
        z = float(lgbm.predict_proba(X)[0, 1])
    except Exception: return np.nan
    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        pA = z if not pl else 1.0 / (1.0 + np.exp(-pl[1] * (z - pl[0])))
    return float(np.clip(pA, 0.0, 1.0))


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

all_added_tickers = [r.get(sanitize_name("Ticker")) for r in ss.rows if r.get(sanitize_name("Ticker"))]
_seen = set(); all_added_tickers = [t for t in all_added_tickers if not (t in _seen or _seen.add(t))]
if "align_sel_tickers" not in ss:
    ss["align_sel_tickers"] = all_added_tickers[:]
csel1, csel2 = st.columns([4, 1])
with csel1:
    selected_tickers = st.multiselect("Select added stocks to analyze:", options=all_added_tickers, default=ss["align_sel_tickers"], key="align_sel_tickers")
with csel2:
    st.write(""); st.write("")
    def _clear_sel(): ss["align_sel_tickers"] = []
    st.button("Clear Selection", use_container_width=True, on_click=_clear_sel)
if not selected_tickers:
    st.info("No stocks selected. Pick at least one added ticker above to analyze.")
    st.stop()

sanitized_mpd_col = sanitize_name("Max_Push_Daily_%")
if sanitized_mpd_col not in base_df.columns:
    st.error("Your DB is missing column: Max_Push_Daily_% (Max Push Daily (%) as %).")
    st.stop()

# Use the globally defined sanitized variable list
var_all = [v for v in SANITIZED_UNIFIED_VARS if v in base_df.columns]
if not var_all:
    st.error("No usable numeric features found after loading. Ensure your Excel has mapped numeric columns.")
    st.stop()

added_df_selected = pd.DataFrame([r for r in ss.rows if r.get(sanitize_name("Ticker")) in set(selected_tickers)])

@st.cache_data(show_spinner="Training models and computing alignment. Please wait...")
def compute_alignment_data(_base_df, _added_df):
    gain_cutoffs = list(range(0, 601, 25))
    def _make_split(df_base: pd.DataFrame, thr_val: float):
        df_tmp = df_base.copy()
        df_tmp["__Group__"] = np.where(pd.to_numeric(df_tmp[sanitized_mpd_col], errors="coerce") >= thr_val, f"≥{int(thr_val)}%", "Rest")
        gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
        return df_tmp, gA_, gB_
    thr_labels, series_N_med, series_C_med, series_L_med = [], [], [], []
    for thr_val in gain_cutoffs:
        df_split, gA, gB = _make_split(_base_df, float(thr_val))
        vc = df_split["__Group__"].value_counts()
        if (vc.get(gA, 0) < 10) or (vc.get(gB, 0) < 10): continue
        nca_model = _train_nca_or_lda(df_split, gA, gB, var_all) or {}
        cat_model = _train_catboost_once(df_split, gA, gB, var_all) or {}
        lgb_model = _train_lightgbm_once(df_split, gA, gB, var_all) or {}
        if not nca_model and not cat_model and not lgb_model: continue
        req_feats = sorted(set(nca_model.get("feats", []) + cat_model.get("feats", []) + lgb_model.get("feats", [])))
        if not req_feats or _added_df.empty: continue
        Xadd = _added_df[req_feats].apply(pd.to_numeric, errors="coerce")
        mask = Xadd.notna().all(axis=1)
        pred_rows = _added_df.loc[mask].to_dict(orient="records")
        if not pred_rows: continue
        pN, pC, pL = [], [], []
        if nca_model: pN = [p for r in pred_rows if np.isfinite(p := _nca_predict_proba_row(r, nca_model))]
        if cat_model: pC = [p for r in pred_rows if np.isfinite(p := _cat_predict_proba_row(r, cat_model))]
        if lgb_model: pL = [p for r in pred_rows if np.isfinite(p := _lgb_predict_proba_row(r, lgb_model))]
        if not pN and not pC and not pL: continue
        thr_labels.append(int(thr_val))
        series_N_med.append(float(np.nanmedian(pN) * 100.0) if pN else np.nan)
        series_C_med.append(float(np.nanmedian(pC) * 100.0) if pC else np.nan)
        series_L_med.append(float(np.nanmedian(pL) * 100.0) if pL else np.nan)
    if not thr_labels: return pd.DataFrame()
    data = []
    for i, thr in enumerate(thr_labels):
        data.append({"GainCutoff_%": thr, "Series": "NCA: P(A)", "Value": series_N_med[i]})
        data.append({"GainCutoff_%": thr, "Series": "CatBoost: P(A)", "Value": series_C_med[i]})
        data.append({"GainCutoff_%": thr, "Series": "LightGBM: P(A)", "Value": series_L_med[i]})
    return pd.DataFrame(data)

compute_btn = st.button("Compute & Display Alignment Chart", use_container_width=True, type="primary")

if compute_btn:
    df_long = compute_alignment_data(base_df, added_df_selected)
    ss.alignment_df = df_long
else:
    df_long = ss.get("alignment_df")

if df_long is not None and not df_long.empty:
    color_domain = ["NCA: P(A)", "CatBoost: P(A)", "LightGBM: P(A)"]
    color_range  = ["#10b981", "#8b5cf6", "#f59e0b"]
    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X("GainCutoff_%:O", title="Gain% cutoff (0 → 600, step 25)"),
        y=alt.Y("Value:Q", title="Median P(A) (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
        xOffset="Series:N",
        tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    try:
        pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
        series_names = list(pivot.columns)
        color_map = {"NCA: P(A)": "#10b981", "CatBoost: P(A)": "#8b5cf6", "LightGBM: P(A)": "#f59e0b"}
        colors = [color_map.get(s, "#999999") for s in series_names]
        thresholds = pivot.index.tolist()
        n_groups, n_series = len(thresholds), len(series_names)
        x, width = np.arange(n_groups), 0.8 / max(n_series, 1)
        import io as _io
        fig, ax = plt.subplots(figsize=(max(6, n_groups*0.7), 4.5))
        for i, s in enumerate(series_names):
            ax.bar(x + i*width - (n_series-1)*width/2, pivot[s].values.astype(float), width, label=s, color=colors[i])
        ax.set_xticks(x); ax.set_xticklabels([str(t) for t in thresholds], rotation=0)
        ax.set_ylim(0, 100); ax.set_xlabel("Gain% cutoff"); ax.set_ylabel("Median P(A) for selected added stocks (%)")
        ax.legend(loc="upper left", frameon=False)
        buf = _io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig); png_bytes = buf.getvalue()
    except Exception: png_bytes = None
    col1, col2 = st.columns(2)
    if png_bytes:
        with col1: st.download_button("Download PNG", png_bytes, f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "image/png", use_container_width=True)
    spec = chart.to_dict()
    html_tpl = f'<!doctype html><html><head><meta charset="utf-8"><title>Alignment Distribution</title><script src="https://cdn.jsdelivr.net/npm/vega@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script></head><body><div id="vis"></div><script>vegaEmbed("#vis", {json.dumps(spec)}, {{"actions": true}});</script></body></html>'
    with col2: st.download_button("Download HTML", html_tpl.encode("utf-8"), "alignment.html", "text/html", use_container_width=True)
elif df_long is not None and df_long.empty:
     st.warning("Could not generate an alignment chart. There may not have been enough data across the gain cutoffs for the selected stocks.")
