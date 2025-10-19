# stock_ranking_app.py — Add-Stock + Alignment (Distributions of Added Stocks Only)
# Upgrades:
# - Locked feature set & consistent preprocess (winsorize + safe eps) for ALL thresholds
# - CV-based isotonic calibration (fallback to Platt-like only when necessary)
# - Tiny-data guardrails (min N, min pos-rate, distinct proba count)
# - True conditional probability (if available) else clearly named "Transition Ratio"
# - Smoothing slider (Savitzky–Golay), strength-controlled; raw option
# - De-duplicate Add-Stock rows by ticker
# - Per-threshold caching of artifacts; fast UI
# - Immediate selection refresh after delete
# - Feature usage table + optional CatBoost feature importance
# - Strict percent handling (single path)

import streamlit as st
import pandas as pd
import numpy as np
import re, json, io, hashlib
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime

# Core ML
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from scipy.signal import savgol_filter

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rows", {})  # dict keyed by Ticker (de-duplicated)

# ============================== Unified variables ==============================
UNIFIED_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
    "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
]
ALLOWED_LIVE_FEATURES = UNIFIED_VARS[:]
EXCLUDE_FOR_NCA = []
LOCKED_PERCENT_COLS = ["Gap_%","Max_Pull_PM_%","PM$Vol/MC_%"]  # ensure these are 0..100 %

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

def winsorize_series(s: pd.Series, p=0.005) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 5: return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)

def ensure_percent_once(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """If in [0..1], scale by 100; if in [0..100], keep; if >200, leave (likely already % * 100: flagging would be better)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        x = pd.to_numeric(out[c], errors="coerce")
        if x.notna().sum() == 0: 
            out[c] = x
            continue
        q01, q99 = x.quantile(0.01), x.quantile(0.99)
        if (0.0 <= q01 <= 1.0) and (0.0 <= q99 <= 1.5):  # fractions
            out[c] = x * 100.0
        else:
            out[c] = x
    return out

def dict_hash(obj) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        s = str(obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

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

            # FR_x and PM$Vol/MC_% recompute w/ safe eps
            EPS = 1e-9
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else None
            mcap_basis  = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else None

            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / (df[float_basis] + EPS)).replace([np.inf,-np.inf], np.nan)
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / (df[mcap_basis] + EPS) * 100.0).replace([np.inf,-np.inf], np.nan)

            # Max_Push_Daily_% (from fraction or %)
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            df["Max_Push_Daily_%"] = (
                pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")
                if pmh_col is not None else np.nan
            )
            # Normalize % once
            df = ensure_percent_once(df, LOCKED_PERCENT_COLS + ["Max_Push_Daily_%"])

            # Keep only relevant columns
            keep_cols = set(UNIFIED_VARS + ["Max_Push_Daily_%"])
            df = df[[c for c in df.columns if c in keep_cols]].copy()

            # Winsorize ALL modeling columns (stable tails)
            for c in UNIFIED_VARS:
                if c in df.columns:
                    df[c] = winsorize_series(df[c])

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
    EPS = 1e-9
    row = {
        "Ticker": ticker,
        "MC_PM_Max_M": mc_pmmax,
        "Float_PM_Max_M": float_pm,
        "Gap_%": gap_pct,
        "ATR_$": atr_usd,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,
        "FR_x": (pm_vol / (float_pm + EPS)) if float_pm > 0 else np.nan,
        "PM$Vol/MC_%": (pm_dol / (mc_pmmax + EPS) * 100.0) if mc_pmmax > 0 else np.nan,
        "Max_Pull_PM_%": max_pull_pm,
        "RVOL_Max_PM_cum": rvol_pm_cum,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "CatalystYN": catalyst_yn,
    }
    # Winsorize same as base
    for c in UNIFIED_VARS:
        if c in row and c != "Catalyst":
            # wrap in Series to reuse winsorize
            s = pd.Series([row[c]])
            row[c] = winsorize_series(s).iloc[0]
    ss.rows[ticker] = row  # de-duplicate by ticker
    st.success(f"Saved {ticker}.")

# ============================== Core Math Helpers ==============================
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

# ============================== Model Training (cached per threshold) ==============================
@st.cache_data(show_spinner=False)
def train_threshold_artifacts(df_key: str, thr_val: int, df_split_small: pd.DataFrame, feat_lock: list[str]):
    """Return dict: {'ok':bool, 'gA','gB','centers_tbl','nca':{...},'cat':{...}, 'meta':{N,pos_rate,auc?}}"""
    df_split = df_split_small.copy()

    # Group labels for cutoff
    gA, gB = f"≥{int(thr_val)}%", "Rest"
    df_split["__Group__"] = np.where(
        pd.to_numeric(df_split["Max_Push_Daily_%"], errors="coerce") >= float(thr_val), gA, gB
    )

    # Guardrails
    vc = df_split["__Group__"].value_counts()
    if (vc.get(gA, 0) < 12) or (vc.get(gB, 0) < 12):
        return {"ok": False}

    # LOCKED features
    feats = [c for c in feat_lock if c in df_split.columns]
    Xdf = df_split[feats].apply(pd.to_numeric, errors="coerce")
    y = (df_split["__Group__"] == gA).astype(int).values

    # Mask finite rows
    mask = np.isfinite(Xdf.values).all(axis=1)
    Xdf = Xdf.loc[mask]; y = y[mask]
    if len(y) < 60 or np.unique(y).size < 2:
        return {"ok": False}

    pos_rate = float(y.mean())
    if (pos_rate < 0.10) or (pos_rate > 0.90):
        return {"ok": False}

    X_all = Xdf.values.astype(np.float32, copy=False)

    # Centers (medians)
    centers_tbl = df_split.loc[mask, ["__Group__"] + feats].groupby("__Group__")[feats].median().T
    if gA not in centers_tbl.columns or gB not in centers_tbl.columns:
        return {"ok": False}
    centers_tbl = centers_tbl[[gA, gB]]

    # ---------- Train CatBoost with holdout ----------
    test_size = 0.2 if len(y) >= 100 else max(0.15, min(0.25, 20 / len(y)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr_idx, va_idx = next(sss.split(X_all, y))
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    def _has_both_classes(arr): return np.unique(arr).size == 2
    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)

    cb_params = dict(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=200,
        learning_rate=0.05,
        depth=2,
        l2_leaf_reg=10,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    if eval_ok: cb_params.update(dict(od_type="Iter", od_wait=40))
    else:       cb_params.update(dict(od_type="None"))

    model = CatBoostClassifier(**cb_params)
    try:
        if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
        else:       model.fit(X_all, y)
    except Exception:
        model = CatBoostClassifier(**{**cb_params, "od_type": "None"})
        model.fit(X_all, y)
        eval_ok = False

    # ---------- Calibration via 5-fold CV on TRAIN ONLY ----------
    iso_obj = None
    platt_params = None
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cal_x, cal_y = [], []
        for tr, va in skf.split(Xtr, ytr):
            m = CatBoostClassifier(**{**cb_params, "od_type": "None"})
            m.fit(Xtr[tr], ytr[tr])
            p = m.predict_proba(Xtr[va])[:,1].astype(float)
            cal_x.extend(p.tolist()); cal_y.extend(ytr[va].tolist())

        # need some spread
        if len(cal_x) >= 30 and (len(np.unique(cal_x)) >= 5) and _has_both_classes(np.array(cal_y)):
            iso_obj = IsotonicRegression(out_of_bounds="clip").fit(cal_x, cal_y)
        else:
            # weak fallback
            p_raw = model.predict_proba(Xtr)[:,1]
            z0 = p_raw[ytr==0]; z1 = p_raw[ytr==1]
            if z0.size and z1.size:
                m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                platt_params = (m, k)
    except Exception:
        pass

    # meta (lightweight metrics)
    auc = np.nan
    try:
        from sklearn.metrics import roc_auc_score
        p_va = model.predict_proba(Xva)[:,1] if eval_ok else model.predict_proba(X_all)[:,1]
        y_va = yva if eval_ok else y
        if _has_both_classes(y_va):
            auc = float(roc_auc_score(y_va, p_va))
    except Exception:
        pass

    return {
        "ok": True,
        "gA": gA, "gB": gB,
        "feats": feats,
        "centers_tbl": centers_tbl,
        "cat": {"cb": model, "iso": iso_obj, "platt": platt_params},
        "meta": {"N": int(len(y)), "pos_rate": float(pos_rate), "auc": auc}
    }

def _cat_predict_prob_calibrated(model_pack: dict, Xrow_1d: np.ndarray) -> float:
    if not model_pack: return np.nan
    cb = model_pack.get("cb")
    if cb is None: return np.nan
    try:
        z = float(cb.predict_proba(Xrow_1d.reshape(1, -1))[0,1])
    except Exception:
        return np.nan
    iso: IsotonicRegression | None = model_pack.get("iso")
    pl = model_pack.get("platt")
    if iso is not None:
        pA = float(np.clip(iso.predict([z])[0], 0.0, 1.0))
    elif pl:
        m, k = pl
        pA = 1.0 / (1.0 + np.exp(-k*(z - m)))
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
            counts[gA_] += 0.5; counts[gB_] += 0.5
    total = counts[gA_] + counts[gB_]
    return {"A_pct": 100.0 * counts[gA_] / total if total > 0 else 0.0,
            "B_pct": 100.0 * counts[gB_] / total if total > 0 else 0.0}

# ============================== Alignment ==============================
st.markdown("---")
st.subheader("Alignment")

base_df = ss.get("base_df", pd.DataFrame())
if base_df.empty:
    st.warning("Upload your Excel and click **Build model stocks**. Alignment distributions are disabled until then.")
    st.stop()

if len(ss.rows) == 0:
    st.info("Add at least one stock to compute distributions across cutoffs.")
    st.stop()

all_added_tickers = list(ss.rows.keys())

if "align_sel_tickers" not in st.session_state:
    st.session_state["align_sel_tickers"] = all_added_tickers[:]

def _clear_selection():
    st.session_state["align_sel_tickers"] = []

def _delete_selected():
    tickers_to_delete = st.session_state.get("align_sel_tickers", [])
    if tickers_to_delete:
        for t in tickers_to_delete:
            ss.rows.pop(t, None)
        # Refresh options
        st.session_state["align_sel_tickers"] = []
        st.rerun()

col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    selected_tickers = st.multiselect(
        "Select stocks to include in the charts below. The delete button will act on this selection.",
        options=list(ss.rows.keys()),
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
    st.error("Your DB is missing column: Max_Push_Daily_% (as %).")
    st.stop()

# Feature lock
FEATS_LOCKED = [c for c in UNIFIED_VARS if c in base_df.columns]
if not FEATS_LOCKED:
    st.error("No usable numeric features found after loading. Ensure your Excel mapped numeric columns.")
    st.stop()

gain_cutoffs = list(range(25, 301, 25))

# Prepare df key for cache
df_key = dict_hash({
    "cols": FEATS_LOCKED,
    "data_hash": str(pd.util.hash_pandas_object(base_df[FEATS_LOCKED + ["Max_Push_Daily_%"]].fillna(-999), index=False).sum())
})

added_df = pd.DataFrame([ss.rows[t] for t in selected_tickers])
# Ensure numeric and winsorize same as base
for c in FEATS_LOCKED:
    if c in added_df.columns:
        added_df[c] = winsorize_series(pd.to_numeric(added_df[c], errors="coerce"))

# Series containers
thr_labels = []
series_cent_A_med, series_cent_B_med = [], []
series_cat_med = []
series_true_cond = []  # true conditional (CatBoost), if available

# Optional: show feature importance for the *last valid threshold*
last_fi = None
last_feats = None
last_meta = None

with st.spinner("Calculating distributions across all cutoffs..."):
    for thr in gain_cutoffs:
        art = train_threshold_artifacts(df_key, thr, base_df[FEATS_LOCKED + ["Max_Push_Daily_%"]], FEATS_LOCKED)
        if not art.get("ok"): 
            continue

        gA, gB = art["gA"], art["gB"]
        centers_tbl = art["centers_tbl"]
        feats_used = art["feats"]
        cat_pack = art["cat"]

        # predict for added rows (mask finite)
        if added_df.empty: 
            continue
        Xadd = added_df[feats_used].apply(pd.to_numeric, errors="coerce")
        mask = np.isfinite(Xadd.values).all(axis=1)
        if mask.sum() == 0: 
            continue
        Xadd_np = Xadd.loc[mask].values

        # CatBoost calibrated P(A)
        pC = []
        for i in range(Xadd_np.shape[0]):
            p = _cat_predict_prob_calibrated(cat_pack, Xadd_np[i,:])
            if np.isfinite(p): pC.append(100.0 * p)

        # Median centers alignment
        pA_centers, pB_centers = [], []
        for _, r in added_df.loc[mask].iterrows():
            sc = _compute_alignment_median_centers(r.to_dict(), centers_tbl)
            if sc:
                pA_centers.append(sc['A_pct']); pB_centers.append(sc['B_pct'])

        if not any([pC, pA_centers]):
            continue

        thr_labels.append(int(thr))
        series_cat_med.append(float(np.nanmedian(pC)) if pC else np.nan)
        series_cent_A_med.append(float(np.nanmedian(pA_centers)) if pA_centers else np.nan)
        series_cent_B_med.append(float(np.nanmedian(pB_centers)) if pB_centers else np.nan)

        # store last feature importance
        try:
            fi_vals = cat_pack["cb"].get_feature_importance(type='FeatureImportance')
            last_fi = pd.DataFrame({"Feature": feats_used, "Importance": fi_vals}).sort_values("Importance", ascending=False)
            last_feats = feats_used
            last_meta = art.get("meta", {})
        except Exception:
            last_fi = None

# Handle no thresholds
if not thr_labels:
    st.info("Not enough data across cutoffs to train stable models. Try a larger database.")
    st.stop()

# ============================== Absolute Probability Chart ==============================
data_abs = []
gA_label = f"≥...% (Median Centers)"
gB_label = f"Rest (Median Centers)"
cat_label = f"CatBoost (Calibrated): P(≥...%)"

for i, thr in enumerate(thr_labels):
    data_abs += [
        {"GainCutoff_%": thr, "Series": gA_label, "Value": series_cent_A_med[i]},
        {"GainCutoff_%": thr, "Series": gB_label, "Value": series_cent_B_med[i]},
        {"GainCutoff_%": thr, "Series": cat_label, "Value": series_cat_med[i]},
    ]
df_abs = pd.DataFrame(data_abs).dropna(subset=["Value"])

color_domain_abs = [gA_label, gB_label, cat_label]
color_range_abs  = ["#3b82f6", "#ef4444", "#8b5cf6"]

abs_chart = (
    alt.Chart(df_abs)
    .mark_bar()
    .encode(
        x=alt.X("GainCutoff_%:O", title=f"Gain% cutoff (step {gain_cutoffs[1]-gain_cutoffs[0]})"),
        y=alt.Y("Value:Q", title="Median Alignment / Calibrated P(A) (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain_abs, range=color_range_abs), legend=alt.Legend(title="Series")),
        xOffset="Series:N",
        tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
    )
    .properties(height=420, title="Absolute Probability of Reaching Gain Cutoff")
)
st.altair_chart(abs_chart, use_container_width=True)

# Export buttons for absolute chart
def create_export_buttons(df, chart_obj, file_prefix):
    png_bytes = b""
    try:
        x_axis_col = df.columns[0]
        pivot = df.pivot(index=x_axis_col, columns="Series", values="Value").sort_index()
        series_names = list(pivot.columns)
        chart_dict = chart_obj.to_dict()
        unique_colors = {}
        try:
            domain = chart_dict['encoding']['color']['scale']['domain']
            range_  = chart_dict['encoding']['color']['scale']['range']
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
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=0)
        ax.set_ylim(0, 100)
        ax.set_xlabel(x_axis_col); ax.set_ylabel("Value (%)")
        ax.legend(loc="upper left", frameon=False)
        ax.set_title(file_prefix)
        buf = io.BytesIO(); fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buf.getvalue()
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label=f"Download PNG ({file_prefix})",
            data=png_bytes,
            file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
            key=f"dl_png_{file_prefix}",
            disabled=not png_bytes
        )
    with col2:
        spec = chart_obj.to_dict()
        html_template = f'<!doctype html><html><head><meta charset="utf-8"><title>{file_prefix}</title><script src="https://cdn.jsdelivr.net/npm/vega@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script></head><body><div id="vis"></div><script>const spec = {json.dumps(spec)}; vegaEmbed("#vis", spec, {{"actions": true}});</script></body></html>'
        st.download_button(
            label=f"Download HTML ({file_prefix})",
            data=html_template.encode("utf-8"),
            file_name=f"{file_prefix}.html",
            mime="text/html",
            use_container_width=True,
            key=f"dl_html_{file_prefix}"
        )

create_export_buttons(df_abs, abs_chart, "alignment_absolute")

# ============================== Conditional Chart (True conditional if possible) ==============================
st.markdown("---")
left, right = st.columns([3,2], vertical_alignment="center")
with left:
    st.subheader("Conditional Probability / Transition")
with right:
    smooth_strength = st.slider("Smoothing strength", 0, 10, value=3, help="0 = raw data, higher = stronger smoothing")

# We compute "true conditional" per added stock using CatBoost calibrated P at each threshold, if consecutive thresholds exist.
thr_to_idx = {t:i for i,t in enumerate(thr_labels)}
cat_array = np.array(series_cat_med, dtype=float)

def smooth_series(y: np.ndarray, strength: int) -> np.ndarray:
    y = y.astype(float)
    if strength <= 0 or np.isnan(y).sum() > len(y) - 5 or len(y) < 5:
        return y
    # window length grows with strength; must be odd and <= len(y)
    win = min(len(y) - (1 - len(y) % 2), max(5, 2*strength + 3))
    if win % 2 == 0: win += 1
    win = min(win, len(y) - (1 - len(y) % 2))
    if win < 5: return y
    try:
        return savgol_filter(y, window_length=win, polyorder=2, mode="interp")
    except Exception:
        return y

# Build true conditional if we have enough consecutive points
cond_rows = []
labels_seq = []
if len(thr_labels) >= 2:
    cat_smooth = smooth_series(cat_array, smooth_strength)
    for i in range(len(thr_labels)-1):
        t0, t1 = thr_labels[i], thr_labels[i+1]
        p0, p1 = cat_smooth[i], cat_smooth[i+1]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 1e-6:
            cond = np.clip((p1 / p0) * 100.0, 0, 100)  # approximation to Pr(≥t1 | ≥t0)
            cond_rows.append({"Transition": f"{t0}% → {t1}%", "Series": "CatBoost (Calibrated)", "Value": cond})
            labels_seq.append(f"{t0}% → {t1}%")

# If no valid true conditional, fallback to "Transition Ratio" across any available series (centers + cat)
use_fallback = len(cond_rows) == 0
if use_fallback:
    # Use smoothed versions of each series
    sA = smooth_series(np.array(series_cent_A_med, dtype=float), smooth_strength)
    sC = smooth_series(np.array(series_cat_med, dtype=float), smooth_strength)
    for i in range(len(thr_labels)-1):
        t0, t1 = thr_labels[i], thr_labels[i+1]
        for name, arr in [("Median Centers", sA), ("CatBoost (Calibrated)", sC)]:
            v0, v1 = arr[i], arr[i+1]
            if np.isfinite(v0) and np.isfinite(v1) and v0 > 1e-6:
                cond_rows.append({"Transition": f"{t0}% → {t1}%", "Series": f"{name} — Transition Ratio", "Value": np.clip((v1/v0)*100.0, 0, 100)})
                labels_seq.append(f"{t0}% → {t1}%")

if cond_rows:
    df_cond = pd.DataFrame(cond_rows)
    cond_series_names = df_cond["Series"].unique().tolist()
    cond_colors = ["#8b5cf6","#3b82f6","#8b5cf6"][:len(cond_series_names)]

    title_suffix = "True Conditional (CatBoost)" if not use_fallback else "Transition Ratio (approx.)"
    cond_chart = (
        alt.Chart(df_cond)
        .mark_bar()
        .encode(
            x=alt.X("Transition:O", title="Gain% Transition", sort=list(dict.fromkeys(labels_seq))),
            y=alt.Y("Value:Q", title="Probability / Ratio (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", scale=alt.Scale(domain=cond_series_names, range=cond_colors), legend=alt.Legend(title="Series")),
            xOffset="Series:N",
            tooltip=["Transition:O", "Series:N", alt.Tooltip("Value:Q", format=".1f")],
        )
        .properties(height=420, title=f"Conditional {title_suffix} (smoothing={smooth_strength})")
    )
    st.altair_chart(cond_chart, use_container_width=True)
else:
    st.info("Not enough sequential data to calculate conditional transitions.")

# ============================== Feature Usage & Importance ==============================
st.markdown("---")
cA, cB = st.columns([2,3])

with cA:
    st.subheader("Features used (last valid threshold)")
    if last_feats:
        st.write(pd.DataFrame({"Feature": last_feats}))
    else:
        st.caption("No threshold produced a valid model.")

with cB:
    st.subheader("CatBoost feature importance (last valid threshold)")
    if last_fi is not None and not last_fi.empty:
        fi_chart = (
            alt.Chart(last_fi.head(15))
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Importance"),
                y=alt.Y("Feature:N", sort='-x', title="Feature"),
                tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".2f")]
            )
            .properties(height=400)
        )
        st.altair_chart(fi_chart, use_container_width=True)
        meta_txt = f"N={last_meta.get('N','?')}  |  Pos-rate={last_meta.get('pos_rate',float('nan')):.2f}  |  AUC={last_meta.get('auc',float('nan')):.3f}"
        st.caption(meta_txt)
    else:
        st.caption("Importance unavailable.")
