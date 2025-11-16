import streamlit as st
import pandas as pd
import numpy as np
import re, json, io, math
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime

# CatBoost import (graceful)
try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except Exception:
    _CATBOOST_OK = False

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Alignment", layout="wide")
st.title("Premarket Stock Alignment")

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
            raw, sel_sheet, _all = _load_sheet(file_bytes)

            df = pd.DataFrame()

            def add_num(dfout, name, src_candidates):
                src = _pick(raw, src_candidates)
                if src:
                    dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

            add_num(df, "MC_PM_Max_M",      ["mc pm max (m)","premarket market cap (m)","mc_pm_max_m","mc pm max (m$)","market cap pm max (m)","market cap pm max m","premarket market cap (m$)"])
            add_num(df, "Float_PM_Max_M",   ["float pm max (m)","premarket float (m)","float_pm_max_m","float pm max (m shares)"])
            add_num(df, "Gap_%",            ["gap %","gap%","premarket gap","gap","gap_percent"])
            add_num(df, "ATR_$",            ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
            add_num(df, "PM_Vol_M",         ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
            add_num(df, "PM_$Vol_M$",       ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
            add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])

            # Load FT if present
            cand_ft = _pick(raw, ["FT","Follow Through","FT_flag","FT=1","FollowThrough","ft"])
            if cand_ft:
                def _to_ft_flag(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1
                    if sv in {"0","false","no","n","f"}: return 0
                    try:
                        return 1 if float(sv) >= 0.5 else 0
                    except:
                        return np.nan
                df["FT"] = raw[cand_ft].map(_to_ft_flag)

            # Deriveds
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            for pct_col in ("Gap_%","PM_Vol_%","Max_Pull_PM_%"):
                if pct_col in df.columns:
                    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce") * 100.0

            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            df["Max_Push_Daily_%"] = (
                pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce") * 100.0
                if pmh_col is not None else np.nan
            )
            
            # --- NEW: load Date (if present) ---
            date_col = _pick(raw, ["Date", "Session Date", "Trade Date", "Day", "SessionDate"])
            if date_col:
                df["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
            
            # --- NEW: load Max Push (biggest leg) if present ---
            max_push_leg_col = _pick(raw, ["Max Push (%)", "Max Push %", "Max_Push_%", "Max Push (leg %)"])
            if max_push_leg_col:
                df["Max_Push_%"] = (
                    pd.to_numeric(raw[max_push_leg_col].map(_to_float), errors="coerce") * 100.0
                )
            
            # --- NEW: load η (%/min) if present (no ×100 here, we keep raw %/min) ---
            eta_col = _pick(raw, ["η (%/min)", "eta (%/min)", "Eta (%/min)", "Eta", "eta"])
            if eta_col:
                df["Eta_%_per_min"] = pd.to_numeric(raw[eta_col].map(_to_float), errors="coerce")
            
            # --- NEW: load MpB% Day (RTH) if present ---
            mpb_col = _pick(raw, ["MpB% Day", "MpB% Day (RTH)", "MpB_Day_%", "Max push breakout day %"])
            if mpb_col:
                df["MpB_Day_%"] = (
                    pd.to_numeric(raw[mpb_col].map(_to_float), errors="coerce") * 100.0
                )
            
            keep_cols = set(
                UNIFIED_VARS
                + ["Max_Push_Daily_%", "Max_Push_%", "Eta_%_per_min", "MpB_Day_%", "FT", "Date"]
            )
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
    c1, c2, c3 = st.columns([1.2, 1.2, 0.9])

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
        catalyst_level = st.selectbox(
            "Catalyst impact",
            ["None", "Very negative", "Negative", "Neutral", "Positive", "Very positive"],
            index=0,
        )
        dilution_level = st.selectbox(
            "Dilution risk",
            ["Unknown", "Low", "Medium", "High"],
            index=0,
        )

    submitted = st.form_submit_button("Add", use_container_width=True)

if submitted and ticker:
    # For models: none = 0, all other catalyst levels = 1
    catalyst_bin = 0.0 if catalyst_level == "None" else 1.0

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

        # Models see simple 0/1
        "Catalyst": catalyst_bin,

        # EV / tilts see the detailed labels
        "CatalystLevel": catalyst_level,
        "DilutionLevel": dilution_level,
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

def _ece_score(y_true, p_pred, bins=10):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]; p = p[mask]
    if y.size < 10: return np.nan
    edges = np.linspace(0, 1, bins+1)
    ece = 0.0; n = y.size
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        idx = (p >= lo) & ((p < hi) if i < bins-1 else (p <= hi))
        if idx.sum() == 0: continue
        avg_p = float(np.mean(p[idx]))
        freq = float(np.mean(y[idx]))
        ece += (idx.sum() / n) * abs(avg_p - freq)
    return float(ece)

def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats: return {}
    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)
    mask = np.isfinite(Xdf.values).all(axis=1)
    Xdf = Xdf.loc[mask]; y = y[mask]
    if Xdf.shape[0] < 40 or np.unique(y).size < 2: return {}
    X = Xdf.values
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=max(0.15, min(0.25, 40/len(y))), random_state=42)
    tr_idx, va_idx = next(sss.split(X, y))
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xtr_s = (Xtr - mu) / sd; Xva_s = (Xva - mu) / sd

    used = "lda"; w_vec = None; components = None
    try:
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42, max_iter=400)
        ztr = nca.fit_transform(Xtr_s, ytr).ravel()
        zva = (Xva_s @ nca.components_.ravel())
        used = "nca"; components = nca.components_
    except Exception:
        X0, X1 = Xtr_s[ytr==0], Xtr_s[ytr==1]
        if X0.shape[0] < 2 or X1.shape[0] < 2: return {}
        m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
        S0, S1 = np.cov(X0, rowvar=False), np.cov(X1, rowvar=False)
        Sw = S0 + S1 + 1e-3*np.eye(Xtr_s.shape[1])
        w_vec = np.linalg.solve(Sw, (m1 - m0))
        w_vec = w_vec / (np.linalg.norm(w_vec) + 1e-12)
        ztr = (Xtr_s @ w_vec)
        zva = (Xva_s @ w_vec)

    if np.nanmean(ztr[ytr==1]) < np.nanmean(ztr[ytr==0]):
        ztr = -ztr; zva = -zva
        if w_vec is not None: w_vec = -w_vec
        if components is not None: components = -components

    iso_bx, iso_by = np.array([]), np.array([])
    platt_params = None
    ece = np.nan
    if ztr.size >= 8 and np.unique(ztr).size >= 3:
        bx, by = _pav_isotonic(ztr, ytr.astype(float))
        if len(bx) >= 2:
            iso_bx, iso_by = np.array(bx), np.array(by)
            pva = _iso_predict(iso_bx, iso_by, zva)
            ece = _ece_score(yva, pva)
    if iso_bx.size < 2:
        z0, z1 = ztr[ytr==0], ztr[ytr==1]
        if z0.size and z1.size:
            m0, m1 = float(np.mean(z0)), float(np.mean(z1))
            s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
            m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
            platt_params = (m, k)
            pva = 1.0 / (1.0 + np.exp(-k*(zva - m)))
            ece = _ece_score(yva, pva)
    return {
        "ok": True, "kind": used, "feats": feats,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "w_vec": (w_vec.tolist() if w_vec is not None else None),
        "components": (components.tolist() if components is not None else None),
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": (platt_params if platt_params is not None else None),
        "ece": float(ece) if np.isfinite(ece) else np.nan,
        "gA": gA_label, "gB": gB_label
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

def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    """
    Train a (possibly) small CatBoost ensemble for the given Gain% split.
    - Uses dynamic complexity based on sample size.
    - Averages predictions from several seeds.
    - Calibrates the averaged probability with isotonic or Platt.
    """
    if not _CATBOOST_OK:
        return {}

    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present):
        return {}

    df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
    if not feats:
        return {}

    Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
    y = (df2["__Group__"].values == gA_label).astype(int)

    mask_finite = np.isfinite(Xdf.values).all(axis=1)
    Xdf = Xdf.loc[mask_finite]
    y   = y[mask_finite]

    n = len(y)
    if n < 40 or np.unique(y).size < 2:
        return {}

    X_all = Xdf.values.astype(np.float32, copy=False)
    y_all = y.astype(np.int32, copy=False)

    # ---------- Dynamic complexity based on n ----------
    if n >= 400:
        depth = 4
        iterations = 400
        l2_leaf_reg = 8
    elif n >= 200:
        depth = 3
        iterations = 300
        l2_leaf_reg = 10
    else:
        depth = 2
        iterations = 200
        l2_leaf_reg = 12

    from sklearn.model_selection import StratifiedShuffleSplit

    test_size = 0.2 if n >= 100 else max(0.15, min(0.25, 20 / n))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y_all))

    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx], y_all[va_idx]

    def _has_both_classes(arr):
        return np.unique(arr).size == 2

    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)

    base_params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        learning_rate=0.05,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=False,
    )

    # ---------- Small ensemble of seeds ----------
    # For small n, we use only one seed; for larger n, a few seeds.
    if n >= 250:
        seed_list = [42, 1337, 7]
    elif n >= 150:
        seed_list = [42, 1337]
    else:
        seed_list = [42]

    models = []
    p_raw_list = []

    for sd in seed_list:
        params = dict(base_params)
        params["random_seed"] = sd

        # Only use early stopping if eval set is valid
        if eval_ok:
            params.update(dict(od_type="Iter", od_wait=40))
        else:
            params.update(dict(od_type="None"))

        model = CatBoostClassifier(**params)

        try:
            if eval_ok:
                model.fit(Xtr, ytr, eval_set=(Xva, yva))
            else:
                model.fit(X_all, y_all)
        except Exception:
            # As a fallback, try without early stopping at all
            try:
                params["od_type"] = "None"
                model = CatBoostClassifier(**params)
                model.fit(X_all, y_all)
            except Exception:
                continue  # skip this seed if it totally fails

        models.append(model)

        if eval_ok:
            p_raw = model.predict_proba(Xva)[:, 1].astype(float)
            p_raw_list.append(p_raw)

    if not models:
        return {}

    # ---------- Calibration: isotonic or Platt, on ensemble mean ----------
    iso_bx = np.array([])
    iso_by = np.array([])
    platt = None
    ece = np.nan

    if eval_ok and p_raw_list:
        # Ensemble mean probability on validation set
        p_raw_stack = np.vstack(p_raw_list)
        p_raw_mean = np.mean(p_raw_stack, axis=0)

        try:
            if np.unique(p_raw_mean).size >= 3 and _has_both_classes(yva):
                bx, by = _pav_isotonic(p_raw_mean, yva.astype(float))
                if len(bx) >= 2:
                    iso_bx, iso_by = np.array(bx), np.array(by)
                    p_cal = _iso_predict(iso_bx, iso_by, p_raw_mean)
                    ece = _ece_score(yva, p_cal)

            if iso_bx.size < 2:
                # Platt fallback on ensemble mean
                z0 = p_raw_mean[yva == 0]
                z1 = p_raw_mean[yva == 1]
                if z0.size and z1.size:
                    m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                    s0, s1 = float(np.std(z0) + 1e-9), float(np.std(z1) + 1e-9)
                    m = 0.5 * (m0 + m1)
                    k = 2.0 / (0.5 * (s0 + s1) + 1e-6)
                    platt = (m, k)
                    p_cal = 1.0 / (1.0 + np.exp(-platt[1] * (p_raw_mean - platt[0])))
                    ece = _ece_score(yva, p_cal)
        except Exception:
            pass

    return {
        "ok": True,
        "feats": feats,
        "gA": gA_label,
        "gB": gB_label,
        # Keep the first model in cb for backward compatibility
        "cb": models[0],
        # New: full ensemble
        "cb_list": models,
        "iso_bx": iso_bx.tolist(),
        "iso_by": iso_by.tolist(),
        "platt": platt,
        "ece": float(ece) if np.isfinite(ece) else np.nan,
    }

def _cat_predict_proba_row(xrow: dict, model: dict) -> float:
    """
    Predict probability using:
    - mean of CatBoost ensemble (if available),
    - then isotonic or Platt calibration.
    """
    if not model or not model.get("ok"):
        return np.nan

    feats = model["feats"]
    vals = []
    for f in feats:
        v = pd.to_numeric(xrow.get(f), errors="coerce")
        v = float(v) if pd.notna(v) else np.nan
        if not np.isfinite(v):
            return np.nan
        vals.append(v)

    X = np.array(vals, dtype=float).reshape(1, -1)

    cb_list = model.get("cb_list")
    if not cb_list:
        # Backward compatibility: single model
        cb = model.get("cb")
        if cb is None:
            return np.nan
        try:
            z = float(cb.predict_proba(X)[0, 1])
        except Exception:
            return np.nan
    else:
        # Ensemble mean
        z_vals = []
        for cb in cb_list:
            try:
                z_vals.append(float(cb.predict_proba(X)[0, 1]))
            except Exception:
                continue
        if not z_vals:
            return np.nan
        z = float(np.mean(z_vals))

    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    pl = model.get("platt")

    if iso_bx.size >= 2 and iso_by.size >= 2:
        pA = float(_iso_predict(iso_bx, iso_by, np.array([z]))[0])
    elif pl:
        m, k = pl
        pA = 1.0 / (1.0 + np.exp(-k * (z - m)))
    else:
        pA = z

    return float(np.clip(pA, 0.0, 1.0))

def _compute_alignment_median_centers(stock_row: dict, centers_tbl: pd.DataFrame) -> dict:
    if centers_tbl is None or centers_tbl.empty or len(centers_tbl.columns) != 2:
        return {}
    gA_, gB_ = list(centers_tbl.columns)
    counts = {gA_: 0.0, gB_: 0.0}
    for var in centers_tbl.index:
        xv = pd.to_numeric(stock_row.get(var), errors="coerce")
        if not np.isfinite(xv): continue
        vA = centers_tbl.at[var, gA_]
        vB = centers_tbl.at[var, gB_]
        if pd.isna(vA) or pd.isna(vB): continue
        if abs(xv - vA) < abs(xv - vB):
            counts[gA_] += 1.0
        elif abs(xv - vB) < abs(xv - vA):
            counts[gB_] += 1.0
        else:
            counts[gA_] += 0.5
            counts[gB_] += 0.5
    total = counts[gA_] + counts[gB_]
    return {
        "A_pct": 100.0 * counts[gA_] / total if total > 0 else 0.0,
        "B_pct": 100.0 * counts[gB_] / total if total > 0 else 0.0,
    }

# ============================== Regime (PMH Environment) ==============================
st.markdown("---")
st.subheader("Regime")

def vspace(px: int = 16):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

base_df = ss.get("base_df", pd.DataFrame())
if base_df.empty:
    st.warning("Upload your Excel and click **Build model stocks**. Regime plot and alignment distributions are disabled until then.")
    st.stop()

needed_cols = {"Max_Push_Daily_%", "FT"}
missing = needed_cols - set(base_df.columns)
if missing:
    st.info("Regime view needs at least Max_Push_Daily_% and FT in your DB.")
else:
    df_reg = base_df.copy()

    # Use Date where possible, else use a simple trade index
    if "Date" in df_reg.columns and df_reg["Date"].notna().any():
        df_reg = df_reg.sort_values("Date")
        x_col = "Date"
    else:
        df_reg = df_reg.reset_index().rename(columns={"index": "TradeIdx"})
        x_col = "TradeIdx"

    # Only FT=1 winners with valid Max_Push_Daily_%
    m_gain = pd.to_numeric(df_reg["Max_Push_Daily_%"], errors="coerce")
    m_ft   = pd.to_numeric(df_reg["FT"], errors="coerce")
    df_ft = df_reg[(m_ft == 1) & np.isfinite(m_gain)].copy()

    if df_ft.shape[0] < 20:
        st.info("Not enough FT=1 trades in the DB to compute a regime yet.")
    else:
        window = 40       # rolling window for Max Push stats (FT trades)
        smooth_window = 7 # smoothing window for RegimeScore

        # Rolling stats over winners
        roll_gain = df_ft["Max_Push_Daily_%"].rolling(window=window, min_periods=5)
        df_ft["reg_median_maxpush"] = roll_gain.median()
        df_ft["reg_q90_maxpush"]    = roll_gain.quantile(0.9)

        if "Max_Push_%" in df_ft.columns:
            df_ft["reg_median_leg"] = df_ft["Max_Push_%"].rolling(window=window, min_periods=5).median()
        if "Eta_%_per_min" in df_ft.columns:
            df_ft["reg_median_eta"] = df_ft["Eta_%_per_min"].rolling(window=window, min_periods=5).median()
        if "MpB_Day_%" in df_ft.columns:
            df_ft["reg_median_mpb"] = df_ft["MpB_Day_%"].rolling(window=window, min_periods=5).median()

        # Keep only rows where we have rolling q90
        df_ft_valid = df_ft.dropna(subset=["reg_q90_maxpush"])
        if df_ft_valid.empty:
            st.info("Not enough rolling data to show a regime curve yet.")
        else:
            # --- Global baselines over all FT winners (static) ---
            glob_q90 = float(df_ft["Max_Push_Daily_%"].quantile(0.9))
            glob_med = float(df_ft["Max_Push_Daily_%"].median())
            if "Eta_%_per_min" in df_ft.columns:
                glob_eta = float(df_ft["Eta_%_per_min"].median())
            else:
                glob_eta = np.nan

            # --- Compute raw RegimeScore for each row ---
            def _row_regime_score(row, gq=glob_q90, gm=glob_med, ge=glob_eta):
                ratios = []
                q = row.get("reg_q90_maxpush", np.nan)
                m = row.get("reg_median_maxpush", np.nan)
                e = row.get("reg_median_eta", np.nan)
                if np.isfinite(q) and np.isfinite(gq) and gq > 0:
                    ratios.append(q / gq)
                if np.isfinite(m) and np.isfinite(gm) and gm > 0:
                    ratios.append(m / gm)
                if np.isfinite(e) and np.isfinite(ge) and ge > 0:
                    ratios.append(e / ge)
                return float(np.mean(ratios)) if ratios else np.nan

            df_ft_valid["RegimeScore_raw"] = df_ft_valid.apply(_row_regime_score, axis=1)

            # Smooth the regime score (rolling mean)
            df_ft_valid["RegimeScore"] = (
                df_ft_valid["RegimeScore_raw"]
                .rolling(window=smooth_window, min_periods=max(3, smooth_window // 2))
                .mean()
            )

            # Drop rows without a smoothed score
            df_ft_valid = df_ft_valid[np.isfinite(df_ft_valid["RegimeScore"])]

            if df_ft_valid.empty:
                st.info("Not enough data to compute a composite smoothed regime score yet.")
            else:
                # Latest regime snapshot
                last_row = df_ft_valid.iloc[-1]
                q90_now      = float(last_row["reg_q90_maxpush"])
                med_now      = float(last_row["reg_median_maxpush"])
                eta_now      = float(last_row["reg_median_eta"]) if "reg_median_eta" in df_ft_valid.columns else np.nan
                regime_score = float(last_row["RegimeScore"])

                # Make current regime score available to other parts of the app (e.g. EV)
                ss["RegimeScore_current"] = regime_score

                # --- Regime classification (Cold / Normal / Hot) from smoothed score ---
                if regime_score < 0.8:
                    regime = "Cold"
                    color  = "#faa1a4"
                elif regime_score > 1.2:
                    regime = "Hot"
                    color  = "#ff2501"
                else:
                    regime = "Normal"
                    color  = "#015e06"

                # Current snapshot metrics (for context)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    # Badge
                    st.markdown(
                        f"""
                        <div style="display:inline-block;padding:6px 12px;border-radius:999px;
                                   background-color:{color};color:white;font-weight:600;">
                            Regime: {regime}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Smoothed regime score: {regime_score:.2f}")
                with c2:
                    st.metric("q90 Max Push Daily (PMH→HOD)", f"{q90_now:.0f}%")
                with c3:
                    st.metric("Median Max Push Daily", f"{med_now:.0f}%")
                with c4:
                    if np.isfinite(eta_now):
                        st.metric("Median η (%/min)", f"{eta_now:.2f}")
                    else:
                        st.write("")

                # === Single historical graph: smoothed RegimeScore over time (with regime labels) ===
                
                # Build dataframe with extra diagnostic columns for tooltip
                cols_for_hist = [x_col, "RegimeScore"]
                if "reg_q90_maxpush" in df_ft_valid.columns:
                    cols_for_hist.append("reg_q90_maxpush")
                if "reg_median_maxpush" in df_ft_valid.columns:
                    cols_for_hist.append("reg_median_maxpush")
                if "reg_median_eta" in df_ft_valid.columns:
                    cols_for_hist.append("reg_median_eta")
                
                df_reg_hist = df_ft_valid[cols_for_hist].copy()
                
                # Nice readable names for tooltip
                df_reg_hist = df_reg_hist.rename(columns={
                    "RegimeScore":        "RegimeScore",
                    "reg_q90_maxpush":    "Q90_MaxPushDaily",
                    "reg_median_maxpush": "Median_MaxPushDaily",
                    "reg_median_eta":     "Median_Eta"
                })

                # Add per-point regime label (Cold / Normal / Hot)
                def _label_regime(score: float) -> str:
                    if not np.isfinite(score):
                        return "Unknown"
                    if score < 0.8:
                        return "Cold"
                    elif score > 1.2:
                        return "Hot"
                    else:
                        return "Normal"
                
                df_reg_hist["RegimeLabel"] = df_reg_hist["RegimeScore"].apply(_label_regime)
                
                # X encoding: Date or FT index
                if x_col == "Date":
                    x_enc = alt.X("Date:T", title="Date")
                    x_field_tooltip = "Date:T"
                else:
                    # ensure a stable index for plotting when no Date
                    df_reg_hist = df_reg_hist.reset_index(drop=True)
                    df_reg_hist["TradeIdx"] = df_reg_hist.index + 1
                    x_enc = alt.X("TradeIdx:Q", title="FT Trade #")
                    x_field_tooltip = "TradeIdx:Q"
                
                # Base line (always same color)
                line = (
                    alt.Chart(df_reg_hist)
                    .mark_line(color="#ff2501")
                    .encode(
                        x=x_enc,
                        y=alt.Y("RegimeScore:Q", title="Smoothed Regime Score"),
                        tooltip=[
                            x_field_tooltip,
                            alt.Tooltip("RegimeScore:Q",        title="Regime Score",           format=".2f"),
                            alt.Tooltip("Q90_MaxPushDaily:Q",   title="q90 Max Push Daily%",    format=".0f"),
                            alt.Tooltip("Median_MaxPushDaily:Q",title="Median Max Push Daily%", format=".0f"),
                            alt.Tooltip("Median_Eta:Q",         title="Median η (%/min)",       format=".2f"),
                            alt.Tooltip("RegimeLabel:N",        title="Regime"),
                        ],
                    )
                )
                
                # Horizontal guides at 0.8 and 1.2 (Cold/Hot thresholds)
                guide_df = pd.DataFrame({"y": [0.8, 1.2]})
                guides = (
                    alt.Chart(guide_df)
                    .mark_rule(strokeDash=[4, 4], color="#888888")
                    .encode(y="y:Q")
                )
                
                score_chart = (line + guides).resolve_scale(color="independent")
                st.altair_chart(score_chart, use_container_width=True)

# ============================== Alignment (Distributions) ==============================
st.markdown("---")
st.subheader("Alignment")

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
        ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(tickers_to_delete)]
        st.session_state["align_sel_tickers"] = []

col1, col2, col3, col4 = st.columns([2, 5, 1.2, 1.2])
with col1:
    split_mode = st.selectbox("", options=["Gain%", "FT Gain%"], index=0, label_visibility="collapsed")
with col2:
    selected_tickers = st.multiselect(
        "Select stocks to include in the charts below. The delete button will act on this selection.",
        options=all_added_tickers,
        default=st.session_state.get("align_sel_tickers", []),
        key="align_sel_tickers",
        label_visibility="collapsed",
    )
with col3:
    st.button("Clear", use_container_width=True, on_click=_clear_selection)
with col4:
    st.button("Delete", use_container_width=True, on_click=_delete_selected, disabled=not selected_tickers)

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

gain_cutoffs = list(range(25, 301, 25))

def _make_split(df_base: pd.DataFrame, thr_val: float, mode: str):
    df_tmp = df_base.copy()
    if mode == "FT Gain%" and "FT" in df_tmp.columns:
        gA_, gB_ = f"FT=1 ≥{int(thr_val)}%", "FT=0"
        m_ft = pd.to_numeric(df_tmp.get("FT"), errors="coerce")
        m_gain = pd.to_numeric(df_tmp.get("Max_Push_Daily_%"), errors="coerce")
        df_tmp["__Group__"] = np.where((m_ft == 1) & (m_gain >= thr_val), gA_, "FT=0")
    else:
        gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
        m_gain = pd.to_numeric(df_tmp.get("Max_Push_Daily_%"), errors="coerce")
        df_tmp["__Group__"] = np.where(m_gain >= thr_val, gA_, "Rest")
    return df_tmp, gA_, gB_

added_df = pd.DataFrame([r for r in ss.rows if r.get("Ticker") in set(selected_tickers)])

thr_labels = []
series_A_med, series_B_med, series_N_med, series_C_med = [], [], [], []
diag_conf, diag_nA, diag_nB, diag_cov, diag_ece_nca, diag_ece_cat = [], [], [], [], [], []

with st.spinner("Calculating distributions across all cutoffs..."):
    for thr_val in gain_cutoffs:
        df_split, gA, gB = _make_split(base_df, float(thr_val), split_mode)

        vc = df_split["__Group__"].value_counts()
        nA, nB = int(vc.get(gA, 0)), int(vc.get(gB, 0))
        if (nA < 10) or (nB < 10):
            continue

        nca_model = _train_nca_or_lda(df_split, gA, gB, var_all) or {}
        cat_model = _train_catboost_once(df_split, gA, gB, var_all) or {}

        features_for_centers = [f for f in ALLOWED_LIVE_FEATURES if f in df_split.columns]
        centers_tbl = df_split.groupby("__Group__")[features_for_centers].median().T
        if gA not in centers_tbl.columns or gB not in centers_tbl.columns:
            continue
        centers_tbl = centers_tbl[[gA, gB]]

        req_feats = sorted(set(nca_model.get("feats", []) + cat_model.get("feats", []) + features_for_centers))
        if not req_feats: continue

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
            center_scores = _compute_alignment_median_centers(r, centers_tbl)
            if center_scores:
                pA_centers.append(center_scores['A_pct'])
                pB_centers.append(center_scores['B_pct'])

        if not any([pN, pC, pA_centers]):
            continue

        # Confidence (for tooltip only)
        if nA == 0 or nB == 0:
            coverage = 0.0
        else:
            n_eff = 2.0 / (1.0/nA + 1.0/nB)
            coverage = min(1.0, n_eff / 200.0)
        ece_nca = nca_model.get("ece", np.nan) if nca_model else np.nan
        ece_cat = cat_model.get("ece", np.nan) if cat_model else np.nan
        cal_scores = []
        for e in [ece_nca, ece_cat]:
            if np.isfinite(e):
                cal_scores.append(max(0.0, min(1.0, 1.0 - (e / 0.2))))
        calib_score = float(np.mean(cal_scores)) if cal_scores else 0.6
        confidence = max(0.25, min(1.0, math.sqrt(coverage) * calib_score))

        thr_labels.append(int(thr_val))
        series_N_med.append(float(np.nanmedian(pN)) if pN else np.nan)
        series_C_med.append(float(np.nanmedian(pC)) if pC else np.nan)
        series_A_med.append(float(np.nanmedian(pA_centers)) if pA_centers else np.nan)
        series_B_med.append(float(np.nanmedian(pB_centers)) if pB_centers else np.nan)
        diag_conf.append(confidence); diag_cov.append(coverage)
        diag_nA.append(nA); diag_nB.append(nB)
        diag_ece_nca.append(float(ece_nca) if np.isfinite(ece_nca) else np.nan)
        diag_ece_cat.append(float(ece_cat) if np.isfinite(ece_cat) else np.nan)

if not thr_labels:
    st.info("Not enough data across cutoffs to train models. Try using a larger database.")
else:
    # --- SATURATED BARS: no opacity channel, hover still shows diagnostics ---
    data = []

    if 'split_mode' in locals() and split_mode == "FT Gain%":
        gA_label = "FT=1 ≥...% (Median Centers)"
        gB_label = "FT=0 (Median Centers)"
        nca_label = "NCA: P(FT=1 ≥...%)"
        cat_label = "CatBoost: P(FT=1 ≥...%)"
    else:
        gA_label = "≥...% (Median Centers)"
        gB_label = "Rest (Median Centers)"
        nca_label = "NCA: P(≥...%)"
        cat_label = "CatBoost: P(≥...%)"

    for i, thr in enumerate(thr_labels):
        common = {
            "GainCutoff_%": thr,
            "Confidence": float(diag_conf[i]),
            "Coverage": float(diag_cov[i]),
            "nA": int(diag_nA[i]),
            "nB": int(diag_nB[i]),
            "ECE_NCA": float(diag_ece_nca[i]) if np.isfinite(diag_ece_nca[i]) else np.nan,
            "ECE_Cat": float(diag_ece_cat[i]) if np.isfinite(diag_ece_cat[i]) else np.nan,
        }
        data.append({**common, "Series": gA_label, "Value": series_A_med[i]})
        data.append({**common, "Series": gB_label, "Value": series_B_med[i]})
        data.append({**common, "Series": nca_label, "Value": series_N_med[i]})
        data.append({**common, "Series": cat_label, "Value": series_C_med[i]})

    df_long = pd.DataFrame(data).dropna(subset=['Value'])

    color_domain = [gA_label, gB_label, nca_label, cat_label]
    color_range  = ["#015e06", "#b30100", "#faa1a4", "#ff2501"]

    tooltip_cols = [
        "GainCutoff_%:O","Series:N",
        alt.Tooltip("Value:Q", format=".1f"),
        alt.Tooltip("Confidence:Q", format=".2f"),
        alt.Tooltip("Coverage:Q", format=".2f"),
        alt.Tooltip("nA:Q", title="n(A)"),
        alt.Tooltip("nB:Q", title="n(B)"),
        alt.Tooltip("ECE_NCA:Q", format=".3f"),
        alt.Tooltip("ECE_Cat:Q", format=".3f"),
    ]

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("GainCutoff_%:O", title=f"Gain% cutoff"),
            y=alt.Y("Value:Q", title="Median Alignment / P(A) (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title="Analysis Series")),
            xOffset="Series:N",
            tooltip=tooltip_cols,
        )
    )
    vspace(24)
    st.altair_chart(chart, use_container_width=True)

# ============================== EV (Single Chart: Liquidity + Catalyst Adjusted) ==============================
st.markdown("---")
st.subheader("Expected Value")

if not thr_labels:
    st.info("EV needs the computed probability series. Upload DB → Build model → Add stocks.")
else:
    # ---- Controls you keep ----
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        prob_source = st.selectbox(
            "Probability source",
            ["NCA & CatBoost Avg", "NCA", "CatBoost", "Median Centers"],
            index=0,
            key="prob_source"
        )
    with c2:
        rr_assumed = st.number_input(
            "Assumed R:R", min_value=0.1, value=1.80, step=0.10, format="%.2f",
            key="rr_assumed"
        )

    # ---- Helper: convert % to [0,1] ----
    def _to_prob_list(series_pct):
        return [(s/100.0) if (s is not None and not np.isnan(s)) else np.nan for s in series_pct]

    # ---- Build probability list from selection ----
    if prob_source.startswith("NCA & CatBoost Avg"):
        p_n = _to_prob_list(series_N_med)
        p_c = _to_prob_list(series_C_med)
        p_list = []
        for i in range(len(thr_labels)):
            vals = []
            if i < len(p_n) and np.isfinite(p_n[i]): vals.append(p_n[i])
            if i < len(p_c) and np.isfinite(p_c[i]): vals.append(p_c[i])
            p_list.append(float(np.mean(vals)) if vals else np.nan)
    elif prob_source == "NCA":
        p_list = _to_prob_list(series_N_med)
    elif prob_source == "CatBoost":
        p_list = _to_prob_list(series_C_med)
    else:
        p_list = _to_prob_list(series_A_med)  # Median Centers as fallback

    # ---- Always-on tilts (robust; percentile-based) ----
    lam_liq = 0.10  # liquidity tilt strength on probability
    gam_cat = 0.05  # catalyst tilt strength on probability

    def _valid_arr(series):
        arr = pd.to_numeric(series, errors="coerce").astype(float).values if series is not None else np.array([])
        return arr[np.isfinite(arr)]

    def _pct_rank(x, arr: np.ndarray) -> float:
        """Percentile rank in [0,1]; 0.5 if x or arr invalid."""
        if not np.isfinite(x) or arr.size == 0:
            return 0.5
        less = np.sum(arr < x)
        equal = np.sum(arr == x)
        return float((less + 0.5 * equal) / len(arr))

    # Historical distributions (from your DB)
    arr_pm_mc = _valid_arr(base_df.get("PM$Vol/MC_%", pd.Series(dtype=float)))
    arr_fr    = _valid_arr(base_df.get("FR_x", pd.Series(dtype=float)))
    arr_rvol  = _valid_arr(base_df.get("RVOL_Max_PM_cum", pd.Series(dtype=float)))

    # Selected tickers → median cross-sectional values
    selected_names = st.session_state.get("align_sel_tickers", [])  # fallback to alignment selection
    selected_set = set(selected_names) if isinstance(selected_names, list) else set()
    added_df_full = pd.DataFrame([r for r in ss.rows if r.get("Ticker") in selected_set]) if selected_set else pd.DataFrame()

    pm_mc_val = float(np.nanmedian(pd.to_numeric(added_df_full.get("PM$Vol/MC_%"), errors="coerce"))) if not added_df_full.empty else np.nan
    frx_val   = float(np.nanmedian(pd.to_numeric(added_df_full.get("FR_x"), errors="coerce")))       if not added_df_full.empty else np.nan
    rvol_val  = float(np.nanmedian(pd.to_numeric(added_df_full.get("RVOL_Max_PM_cum"), errors="coerce"))) if not added_df_full.empty else np.nan

    # Percentile ranks (0..1) → center to [-0.5, 0.5] → gentle tanh mapping
    pr1 = _pct_rank(pm_mc_val, arr_pm_mc)  # PM$Vol/MC_%
    pr2 = _pct_rank(frx_val,   arr_fr)     # FR_x
    pr3 = _pct_rank(rvol_val,  arr_rvol)   # RVOL_Max_PM_cum
    score = float(np.nanmean([pr1, pr2, pr3]) - 0.5)  # [-0.5, 0.5]
    L_prob = 1.0 + lam_liq * float(np.tanh(score * 3.0))
    L_prob = float(np.clip(L_prob, 0.85, 1.20))  # modest bounds

    # ---- Catalyst tilt (multi-level, from CatalystLevel) ----
    cat_level_current = None
    K_prob = 1.0  # default neutral
    
    if not added_df_full.empty and "CatalystLevel" in added_df_full.columns:
        # Take the most common catalyst level among selected tickers
        mode_series = added_df_full["CatalystLevel"].dropna().astype(str)
        if not mode_series.empty:
            cat_level_current = mode_series.mode().iloc[0]
    
    # Map level -> tilt for EV (probability side)
    if cat_level_current == "Very negative":
        K_prob = 0.75    # heavy punishment
    elif cat_level_current == "Negative":
        K_prob = 0.90    # mild punishment
    elif cat_level_current == "Neutral":
        K_prob = 1.00    # fully neutral
    elif cat_level_current == "Positive":
        K_prob = 1.10    # moderate boost
    elif cat_level_current == "Very positive":
        K_prob = 1.25    # strong boost
    elif cat_level_current == "None":
        # no catalyst at all: slightly worse than neutral
        K_prob = 0.90
    else:
        K_prob = 1.00    # unknown = neutral
    
    K_prob = float(np.clip(K_prob, 0.70, 1.30))  # safety clamp

    # ---- Dilution tilt (from DilutionLevel) ----
    dil_level_current = None
    D_prob = 1.0  # default neutral
    
    if not added_df_full.empty and "DilutionLevel" in added_df_full.columns:
        mode_series = added_df_full["DilutionLevel"].dropna().astype(str)
        if not mode_series.empty:
            dil_level_current = mode_series.mode().iloc[0]
    
    # Map dilution level -> tilt
    if dil_level_current == "Low":
        D_prob = 1.05   # slightly favorable
    elif dil_level_current == "Medium":
        D_prob = 0.90   # mild headwind
    elif dil_level_current == "High":
        D_prob = 0.80   # strong headwind, heavy supply
    else:  # "Unknown" or missing
        D_prob = 1.00   # neutral
    
    D_prob = float(np.clip(D_prob, 0.70, 1.10))

    # ---- Apply tilts, compute EV series ----
    rr = float(st.session_state.get("rr_assumed", rr_assumed))

    # RegimeScore from session (set in the Regime block); default neutral = 1.0
    regime_score = float(st.session_state.get("RegimeScore_current", 1.0))
    if not np.isfinite(regime_score) or regime_score <= 0:
        regime_score = 1.0

    # Regime tilts:
    # - R_prob: gentle tilt on probability (square-root of regime_score)
    # - R_rr:   stronger tilt on reward (direct regime_score)
    alpha_reg_prob = 0.5  # controls how strongly regime affects probability
    R_prob = float(np.clip(regime_score ** alpha_reg_prob, 0.7, 1.4))
    R_rr   = float(np.clip(regime_score,                    0.6, 1.6))

    rows = []
    for i, g in enumerate(thr_labels):
        p = p_list[i] if i < len(p_list) else np.nan
        if not np.isfinite(p):
            continue

        # ---------- MODEL-CONFIDENCE BLEND ----------
        # Base rate from DB: nA / (nA + nB) at this cutoff
        if i < len(diag_nA):
            nA_i = float(diag_nA[i])
            nB_i = float(diag_nB[i])
            tot  = nA_i + nB_i
            p_base = nA_i / tot if tot > 0 else np.nan
        else:
            p_base = np.nan

        # Confidence from diagnostics (0.25–1), fallback = 1 (trust model)
        if i < len(diag_conf):
            conf_i = float(diag_conf[i])
        else:
            conf_i = 1.0

        if not np.isfinite(conf_i):
            conf_i = 1.0
        conf_i = float(np.clip(conf_i, 0.0, 1.0))

        # If no usable base rate, just use the model
        if not np.isfinite(p_base):
            p_eff = p
        else:
            p_eff = conf_i * p + (1.0 - conf_i) * p_base
        p_eff = float(np.clip(p_eff, 0.0, 1.0))
        # -------------------------------------------

        # Combine all tilts on probability (liquidity, catalyst, regime, dilution)
        p_env = float(np.clip(p_eff * L_prob * K_prob * R_prob * D_prob, 0.0, 1.0))

        # Regime-adjusted reward side
        rr_env = float(rr * R_rr)

        # Final EV (R), clipped to a reasonable range
        ev_r = float(np.clip(p_env * rr_env - (1.0 - p_env), -3.0, 8.0))

        rows.append({
            "GainCutoff_%": int(g),
            "EV_R": ev_r,
            "P_model": float(p),
            "P_base": float(p_base) if np.isfinite(p_base) else np.nan,
            "P_eff": float(p_eff),
            "P_adj": float(p_env),
            "RR": rr,
            "RR_env": rr_env,
            "LiqTilt": float(L_prob),
            "CatTilt": float(K_prob),
            "RegimeScore": regime_score,
            "RegProbTilt": R_prob,
            "RegRRTilt": R_rr,
            "ModelConf": conf_i,
            "DilutionAdj": float(D_prob),
            "CatLevel": cat_level_current if cat_level_current is not None else "Unknown",
            "DilutionLevel": dil_level_current if dil_level_current is not None else "Unknown",
        })

    df_ev = pd.DataFrame(rows)

    if df_ev.empty:
        st.warning("No EV values to display (insufficient probabilities).")
    else:
        ev_chart = (
            alt.Chart(df_ev)
            .mark_bar()
            .encode(
                x=alt.X("GainCutoff_%:O", title="Gain% cutoff"),
                y=alt.Y("EV_R:Q", title="EV (R)"),
                color=alt.condition(alt.datum["EV_R"] >= 0, alt.value("#015e06"), alt.value("#b30100")),
                tooltip=[
                    alt.Tooltip("GainCutoff_%:O", title="Cutoff (%)"),
                    alt.Tooltip("EV_R:Q",         title="EV (R)",          format=".3f"),
                    alt.Tooltip("P_model:Q",      title="P(model raw)",    format=".2f"),
                    alt.Tooltip("P_base:Q",       title="P(base rate)",    format=".2f"),
                    alt.Tooltip("P_eff:Q",        title="P(effective)",    format=".2f"),
                    alt.Tooltip("P_adj:Q",        title="P(env final)",    format=".2f"),
                    alt.Tooltip("RR:Q",           title="R:R input",       format=".2f"),
                    alt.Tooltip("RR_env:Q",       title="R:R env adj",     format=".2f"),
                    alt.Tooltip("LiqTilt:Q",      title="LiqAdj",     format=".3f"),
                    alt.Tooltip("CatTilt:Q",      title="CatAdj",      format=".3f"),
                    alt.Tooltip("RegimeScore:Q",  title="Regime Score",    format=".2f"),
                    alt.Tooltip("RegProbTilt:Q",  title="RegimeProbAdj", format=".3f"),
                    alt.Tooltip("RegRRTilt:Q",    title="RegimeRRAdj",   format=".3f"),
                    alt.Tooltip("ModelConf:Q",    title="Model conf",      format=".2f"),
                    alt.Tooltip("CatLevel:N",      title="Catalyst level"),
                    alt.Tooltip("DilutionLevel:N", title="Dilution level"),
                ],
            )
        )

        vspace(24)
        st.altair_chart(ev_chart, use_container_width=True)
        st.markdown("---")
