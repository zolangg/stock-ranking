# stock_ranking_app.py — Add-Stock + Alignment (Distributions of Added Stocks Only)
# - Add Stock form (no table)
# - Alignment shows median P(A) over SELECTED added stocks across Gain% cutoffs (0..600 step 25)
# - Models (NCA [required] & CatBoost [required] & LightGBM [required]) train per cutoff on the UPLOADED DB; predictions are for ADDED stocks only
# - Sidebar tuning controls + diagnostics table
# - Simplified: unified variables only

import streamlit as st
import pandas as pd
import numpy as np
import re, json
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# ===== Required model libs =====
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ---- TUNING CONTROLS ----
with st.sidebar:
    st.header("Training & Cutoff Controls")
    MIN_CLASS = st.number_input(
        "Min samples per class",
        min_value=2, max_value=200, value=6, step=1,
        help="Minimum rows needed in EACH class (≥cutoff / Rest). Lower to be more permissive."
    )
    MIN_PRED_ROWS = st.number_input(
        "Min added rows with all features",
        min_value=1, max_value=1000, value=1, step=1,
        help="Minimum number of your SELECTED added stocks that must have all features present."
    )
    REQUIRE_ALL_MODELS = st.checkbox(
        "Require all 3 models per cutoff",
        value=True,
        help="If off, a cutoff is kept when at least one model trained & predicted."
    )

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rows", [])  # user-added stocks (list of dicts)

# ============================== Unified variables ==============================
UNIFIED_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
    "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
]
ALLOWED_LIVE_FEATURES = UNIFIED_VARS[:]     # actually used features
EXCLUDE_FOR_NCA = []                        # hook to drop any for NCA if needed

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

            # detect FT column (not used for split, but mapped for completeness)
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

            # Map numeric fields
            add_num(df, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            add_num(df, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            add_num(df, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
            add_num(df, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            add_num(df, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            add_num(df, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            # catalyst → binary
            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            def _to_binary_local(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1.0
                if sv in {"0","false","no","n","f"}: return 0.0
                try:
                    fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                except: return np.nan
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

            # derived fields
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # scale % fields (DB stores fractions)
            for pct_col in ("Gap_%","PM_Vol_%","Max_Pull_PM_%"):
                if pct_col in df.columns:
                    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce") * 100.0

            # FT groups (not used for split here)
            def _to_binary(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1
                if sv in {"0","false","no","n","f"}: return 0
                try: return 1 if float(sv) >= 0.5 else 0
                except: return np.nan
            df["FT01"] = df["GroupRaw"].map(_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

            # Max Push Daily (%) fraction -> %
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            df["Max_Push_Daily_%"] = (
                pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce") * 100.0
                if pmh_col is not None else np.nan
            )

            # Keep only columns we may use (+ necessary label columns)
            keep_cols = set(UNIFIED_VARS + ["GroupRaw","FT01","GroupFT","Max_Push_Daily_%"])
            df = df[[c for c in df.columns if c in keep_cols]].copy()

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
    ss.rows.append(row)
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

# ============================== NCA (required; LDA fallback) ==============================
def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()

    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}

    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    # drop degenerate columns
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

    # standardize
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs = (X - mu) / sd

    used = "lda"; w_vec = None; components = None
    try:
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

    # Calibration
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

    return {
        "ok": True, "kind": used, "feats": feats,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "w_vec": (w_vec.tolist() if w_vec is not None else None),
        "components": (components.tolist() if components is not None else None),
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
        "platt": (platt_params if platt_params is not None else None),
        "gA": gA_label, "gB": gB_label,
    }

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

# ============================== CatBoost (required) ==============================
def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}

    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}

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

    def _has_both(arr):
        return np.unique(arr).size == 2

    eval_ok = (len(yva) >= 8) and _has_both(yva) and _has_both(ytr)

    params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=200,
        learning_rate=0.04,
        depth=3,
        l2_leaf_reg=6,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    if eval_ok: params.update(dict(od_type="Iter", od_wait=40))
    else:       params.update(dict(od_type="None"))

    model = CatBoostClassifier(**params)
    try:
        if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else:       model.fit(X_all, y_all)
    except Exception:
        return {}

    # Calibration (optional-lite)
    iso_bx = np.array([]); iso_by = np.array([]); platt = None
    try:
        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            if np.unique(p_raw).size >= 3 and (np.unique(yva).size == 2):
                bx, by = _pav_isotonic(p_raw, yva.astype(float))
                if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
            if iso_bx.size < 2:
                z0 = p_raw[yva==0]; z1 = p_raw[yva==1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                    s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9) # FIXED
                    m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                    platt = (m, k)
    except Exception:
        pass

    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label,
            "cb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

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
    except Exception:
        return np.nan

    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        pA = z if not pl else 1.0 / (1.0 + np.exp(-pl[1]*(z - pl[0])))
    return float(np.clip(pA, 0.0, 1.0))

# ============================== LightGBM (required) ==============================
def _train_lgbm_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}

    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES]
    if not feats: return {}

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

    def _has_both(arr): return np.unique(arr).size == 2
    eval_ok = (len(yva) >= 8) and _has_both(yva) and _has_both(ytr)

    params = dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.04,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model = LGBMClassifier(**params)
    try:
        if eval_ok:
            callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)] # FIXED
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=callbacks) # FIXED
        else:
            model.fit(X_all, y_all) # FIXED
    except Exception:
        return {}

    # Calibration (optional-lite)
    iso_bx = np.array([]); iso_by = np.array([]); platt = None
    try:
        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            if np.unique(p_raw).size >= 3 and (np.unique(yva).size == 2):
                bx, by = _pav_isotonic(p_raw, yva.astype(float))
                if len(bx) >= 2: iso_bx, iso_by = np.array(bx), np.array(by)
            if iso_bx.size < 2:
                z0 = p_raw[yva==0]; z1 = p_raw[yva==1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                    s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9) # FIXED
                    m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                    platt = (m, k)
    except Exception:
        pass

    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label,
            "lgb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

def _lgbm_predict_proba_row(xrow: dict, model: dict) -> float:
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
        m = model.get("lgb")
        if m is None: return np.nan
        z = float(m.predict_proba(X)[0, 1])
    except Exception:
        return np.nan

    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    else:
        pl = model.get("platt")
        pA = z if not pl else 1.0 / (1.0 + np.exp(-pl[1]*(z - pl[0])))
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

# --- choose which added stocks to include (safe pattern: sanitize BEFORE, don't assign AFTER) ---
sel_key = "align_sel_tickers"

# Build clean, deduped options list (UPPERCASE strings)
all_added_tickers = [
    str(r.get("Ticker")).strip().upper()
    for r in ss.rows
    if r.get("Ticker") is not None and str(r.get("Ticker")).strip() != ""
]
_seen = set()
all_added_tickers = [t for t in all_added_tickers if t not in _seen and not _seen.add(t)]
valid_set = set(all_added_tickers)

# FIXED: Sanitize session state and handle initialization to preserve user's empty selection
if sel_key not in st.session_state:
    # First run, default to all tickers
    st.session_state[sel_key] = all_added_tickers[:]
else:
    # On subsequent runs, filter the existing selection against the current valid tickers
    current_selection = st.session_state.get(sel_key, [])
    if not isinstance(current_selection, list):
        current_selection = []
    # Filter out any tickers that are no longer in the main list
    st.session_state[sel_key] = [s for s in current_selection if s in valid_set]


csel1, csel2 = st.columns([4, 1])
with csel1:
    # Render widget, which is now correctly bound to the session state
    selected_tickers = st.multiselect(
        label="Select stocks to analyze in the chart",
        options=all_added_tickers,
        key=sel_key,
        placeholder="Select added tickers…",
    )
with csel2:
    # Callback to clear the selection
    def _clear_sel():
        st.session_state[sel_key] = []
    st.button("Clear Selection", use_container_width=True, on_click=_clear_sel, key="clear_btn")

if not selected_tickers:
    st.info("No stocks selected. Pick at least one added ticker above to see the alignment chart.")
    st.stop()

# Required DB columns
if "Max_Push_Daily_%" not in base_df.columns:
    st.error("Your DB is missing column: Max_Push_Daily_% (Max Push Daily (%) as %).")
    st.stop()

# Features available in DB (live) — intersection with unified set
var_all = [v for v in UNIFIED_VARS if v in base_df.columns]
if not var_all:
    st.error("No usable numeric features found after loading. Ensure your Excel has mapped numeric columns.")
    st.stop()

# Gain% cutoffs 0 → 600 step 25
gain_cutoffs = list(range(0, 601, 25))

# Split helper: Gain% ≥ cutoff vs Rest
def _make_split(df_base: pd.DataFrame, thr_val: float):
    df_tmp = df_base.copy()
    df_tmp["__Group__"] = np.where(
        pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val,
        f"≥{int(thr_val)}%", "Rest"
    )
    gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
    return df_tmp, gA_, gB_

# Build a DataFrame of SELECTED added stocks for prediction
added_df = pd.DataFrame([r for r in ss.rows if str(r.get("Ticker")).strip().upper() in set(selected_tickers)])

# ============================== Cutoff sweep with diagnostics & relaxed rules ==============================
diagnostics = []  # per-cutoff notes
thr_labels = []
series_N_med, series_C_med, series_L_med = [], [], []

for thr_val in gain_cutoffs:
    df_split, gA, gB = _make_split(base_df, float(thr_val))

    # class balance check
    vc = df_split["__Group__"].value_counts()
    nA, nB = int(vc.get(gA, 0)), int(vc.get(gB, 0))
    if (nA < MIN_CLASS) or (nB < MIN_CLASS):
        diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": f"Too few samples (A={nA}, B={nB})"})
        continue

    # Train models on DB for this cutoff
    nca_model = _train_nca_or_lda(df_split, gA, gB, var_all) or {}
    cat_model = _train_catboost_once(df_split, gA, gB, var_all) or {}
    lgb_model = _train_lgbm_once(df_split, gA, gB, var_all) or {}

    trained = {"NCA": bool(nca_model), "CatBoost": bool(cat_model), "LightGBM": bool(lgb_model)}
    if REQUIRE_ALL_MODELS:
        if not all(trained.values()):
            diagnostics.append({"Cutoff": int(thr_val), "Kept": False,
                                "Reason": f"Model(s) failed: {', '.join([k for k,v in trained.items() if not v])}"})
            continue
    else:
        if not any(trained.values()):
            diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": "No model trained"})
            continue

    # Determine required features for predicting on SELECTED ADDED stocks
    req_feats = sorted(set(
        (nca_model.get("feats", []) if nca_model else []) +
        (cat_model.get("feats", []) if cat_model else []) +
        (lgb_model.get("feats", []) if lgb_model else [])
    ))
    if not req_feats:
        diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": "No usable features after cleaning"})
        continue

    if added_df.empty:
        diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": "No added stocks to evaluate"})
        continue

    Xadd = added_df[req_feats].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(Xadd.values).all(axis=1)
    pred_rows = added_df.loc[mask].to_dict(orient="records")

    if len(pred_rows) < MIN_PRED_ROWS:
        diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": f"Too few added rows with all features (have {len(pred_rows)}, need ≥{MIN_PRED_ROWS})"})
        continue

    # Predict on SELECTED ADDED stocks
    pN, pC, pL = [], [], []

    if nca_model:
        for r in pred_rows:
            p = _nca_predict_proba_row(r, nca_model)
            if np.isfinite(p): pN.append(p)
    if cat_model:
        for r in pred_rows:
            p = _cat_predict_proba_row(r, nca_model)
            if np.isfinite(p): pC.append(p)
    if lgb_model:
        for r in pred_rows:
            p = _lgbm_predict_proba_row(r, lgb_model)
            if np.isfinite(p): pL.append(p)

    if REQUIRE_ALL_MODELS:
        if not (len(pN) and len(pC) and len(pL)):
            diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": "Predictions missing for one or more models"})
            continue
    else:
        if not (len(pN) or len(pC) or len(pL)):
            diagnostics.append({"Cutoff": int(thr_val), "Kept": False, "Reason": "No predictions produced"})
            continue

    thr_labels.append(int(thr_val))
    series_N_med.append(float(np.nanmedian(pN)*100.0) if len(pN) else np.nan)
    series_C_med.append(float(np.nanmedian(pC)*100.0) if len(pC) else np.nan)
    series_L_med.append(float(np.nanmedian(pL)*100.0) if len(pL) else np.nan)
    diagnostics.append({"Cutoff": int(thr_val), "Kept": True, "Reason": "OK"})

# ============================== Visualization & Downloads ==============================
if not thr_labels:
    st.warning("No cutoffs made it through the filters. See diagnostics below to understand why and adjust the sidebar controls.")
else:
    # Build tidy frame; include series depending on REQUIRE_ALL_MODELS & actual data
    data = []
    for i, thr in enumerate(thr_labels):
        if not np.isnan(series_N_med[i]): data.append({"GainCutoff_%": thr, "Series": "NCA: P(A)", "Value": series_N_med[i]})
        if not np.isnan(series_C_med[i]): data.append({"GainCutoff_%": thr, "Series": "CatBoost: P(A)", "Value": series_C_med[i]})
        if not np.isnan(series_L_med[i]): data.append({"GainCutoff_%": thr, "Series": "LightGBM: P(A)", "Value": series_L_med[i]})
    df_long = pd.DataFrame(data)

    if df_long.empty:
        st.warning("Models ran, but no valid predictions to plot. Check diagnostics below.")
    else:
        color_map = {
            "NCA: P(A)": "#10b981",        # green
            "CatBoost: P(A)": "#8b5cf6",   # purple
            "LightGBM: P(A)": "#f59e0b",   # amber
        }
        series_in_plot = [s for s in ["NCA: P(A)", "CatBoost: P(A)", "LightGBM: P(A)"] if s in df_long["Series"].unique()]
        color_domain = series_in_plot
        color_range  = [color_map[s] for s in series_in_plot]

        chart = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("GainCutoff_%:O", title="Gain% cutoff (0 → 600, step 25)"),
                y=alt.Y("Value:Q", title="Median P(A) (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
                xOffset="Series:N",
                tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

        # PNG export via Matplotlib (dynamic colors)
        png_bytes = None
        try:
            matplotlib.use("Agg") # FIXED: Ensure Matplotlib runs in headless environments
            pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
            series_names = list(pivot.columns)
            thresholds = pivot.index.tolist()
            colors = [color_map.get(s, "#999999") for s in series_names]
            n_groups = len(thresholds); n_series = len(series_names)
            x = np.arange(n_groups); width = 0.8 / max(n_series, 1)

            import io as _io
            fig, ax = plt.subplots(figsize=(max(6, n_groups*0.6), 4.5))
            for i, s in enumerate(series_names):
                vals = pivot[s].values.astype(float)
                ax.bar(x + i*width - (n_series-1)*width/2, vals, width=width, label=s, color=colors[i])

            ax.set_xticks(x); ax.set_xticklabels([str(t) for t in thresholds], rotation=0)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Gain% cutoff")
            ax.set_ylabel("Median P(A) for selected added stocks (%)")
            ax.legend(loc="upper left", frameon=False)

            buf = _io.BytesIO(); fig.tight_layout()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            png_bytes = buf.getvalue()
        except Exception:
            png_bytes = None

        col1, col2 = st.columns(2)
        with col1:
            if png_bytes:
                st.download_button(
                    "Download PNG (Alignment distribution)",
                    data=png_bytes,
                    file_name=f"alignment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_png",
                )
            else:
                st.caption("PNG export unavailable.")
        with col2:
            spec = chart.to_dict()
            html_tpl = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Alignment Distribution</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head><body>
<div id="vis"></div>
<script>
const spec = {json.dumps(spec)};
vegaEmbed("#vis", spec, {{"actions": true}});
</script>
</body></html>"""
            st.download_button(
                "Download HTML (interactive Alignment chart)",
                data=html_tpl.encode("utf-8"),
                file_name="alignment_distribution.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html",
            )

# ============================== Diagnostics ==============================
st.markdown("---")
st.subheader("Diagnostics")
if len(diagnostics):
    diag_df = pd.DataFrame(diagnostics).sort_values(["Cutoff", "Kept"], ascending=[True, False])
    st.dataframe(diag_df, use_container_width=True, hide_index=True)
else:
    st.caption("No diagnostics yet — load a DB and add stocks to evaluate.")
