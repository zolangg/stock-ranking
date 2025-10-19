# stock_ranking_app.py — Minimal Alignment (Distributions Only)
# Spec:
# - Remove prediction daily vol and any data table
# - Keep only a section named "Alignment" that shows distributions of NCA and CatBoost across Gain% cutoffs
# - Gain% cutoffs fixed from 0..600 step 25 (no user selection)
# - Split logic: Gain% >= cutoff  vs  Rest
# - Robust guards; CatBoost optional (warn if missing)

import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, io
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")
st.info("Upload your Excel and click **Build model stocks**. Then scroll to **Alignment** for NCA/CatBoost distributions across Gain% cutoffs (0 → 600).")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("var_core", [])
ss.setdefault("var_moderate", [])
ss.setdefault("__catboost_warned", False)

# ============================== Helpers ==============================
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

@st.cache_data(show_spinner=False)
def _hash_bytes(b: bytes) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(b).hexdigest()

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

            # map numeric fields (add as needed)
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
            def _to_binary_local(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1.0
                if sv in {"0","false","no","n","f"}: return 0.0
                try:
                    fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                except: return np.nan
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

            # derived
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

            # FT groups
            def _to_binary(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1
                if sv in {"0","false","no","n","f"}: return 0
                try:
                    fv = float(sv); return 1 if fv >= 0.5 else 0
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

            ss.base_df = df

            # Variable lists (fallback if not defined in your environment)
            try:
                VAR_CORE
            except NameError:
                VAR_CORE = ["MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum"]
            try:
                VAR_MODERATE
            except NameError:
                VAR_MODERATE = []

            ss.var_core = [v for v in VAR_CORE if v in df.columns]
            ss.var_moderate = [v for v in VAR_MODERATE if v in df.columns]

            st.success(f"Loaded “{sel_sheet}”. Base ready.")
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# Unified feature set for models
VAR_ALL = (ss.get("var_core", []) or []) + (ss.get("var_moderate", []) or [])
try:
    ALLOWED_LIVE_FEATURES
except NameError:
    ALLOWED_LIVE_FEATURES = VAR_ALL[:]  # allow all by default
try:
    EXCLUDE_FOR_NCA
except NameError:
    EXCLUDE_FOR_NCA = []  # exclude none by default

# ============================== Simple Isotonic helpers ==============================
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
    bx = []; by = []
    i = 0
    for mean, n in blocks:
        by.extend([mean]*int(n))
        bx.extend([x[i + j] for j in range(int(n))])
        i += int(n)
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
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42, max_iter=400)
        z = nca.fit_transform(Xs, y).ravel()
        used = "nca"
        components = nca.components_
    except Exception:
        X0 = Xs[y==0]; X1 = Xs[y==1]
        if X0.shape[0] < 2 or X1.shape[0] < 2: return {}
        m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
        S0 = np.cov(X0, rowvar=False); S1 = np.cov(X1, rowvar=False)
        Sw = S0 + S1 + 1e-3*np.eye(Xs.shape[1])
        w_vec = np.linalg.solve(Sw, (m1 - m0))
        w_vec = w_vec / (np.linalg.norm(w_vec) + 1e-12)
        z = (Xs @ w_vec)

    # Orient so larger → A
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

def _nca_predict_proba_row(xrow: pd.Series, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = []
    for f in feats:
        val = pd.to_numeric(xrow.get(f), errors="coerce")
        v = float(val) if pd.notna(val) else np.nan
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

# ============================== CatBoost ==============================
try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except Exception:
    _CATBOOST_OK = False

def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
    present = set(df_groups["__Group__"].dropna().unique().tolist())
    if not ({gA_label, gB_label} <= present): return {}
    if not _CATBOOST_OK:
        if not ss.get("__catboost_warned", False):
            st.info("CatBoost not installed. Run `pip install catboost` to enable the purple CatBoost series.")
            ss["__catboost_warned"] = True
        return {}

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

    def _has_both_classes(arr):
        return np.unique(arr).size == 2

    eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)

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
        else:       model.fit(Xtr, ytr)
    except Exception:
        try:
            model = CatBoostClassifier(**{**params, "od_type": "None"})
            model.fit(X_all, y_all)
            eval_ok = False
        except Exception:
            return {}

    # Calibration (optional; use simple Platt-ish if no valid iso)
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
    except Exception:
        pass

    return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label,
            "cb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

def _cat_predict_proba_row(xrow: pd.Series, model: dict) -> float:
    if not model or not model.get("ok"): return np.nan
    feats = model["feats"]
    vals = []
    for f in feats:
        val = pd.to_numeric(xrow.get(f), errors="coerce")
        v = float(val) if pd.notna(val) else np.nan
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

# ============================== Alignment (Distributions Only) ==============================
st.markdown("---")
st.subheader("Alignment")

base_df = ss.get("base_df", pd.DataFrame())
if base_df.empty:
    st.warning("Upload your Excel and click **Build model stocks**. Alignment distributions are disabled until then.")
    st.stop()

# Required columns
missing = []
if "Max_Push_Daily_%" not in base_df.columns:
    missing.append("Max Push Daily (%) → expected as Max_Push_Daily_%")
if missing:
    st.error("Your DB is missing required columns:\n- " + "\n- ".join(missing))
    st.caption("Fix your column headers or add a mapping in the loader helpers.")
    st.stop()

var_core = ss.get("var_core", [])
var_mod  = ss.get("var_moderate", [])
var_all  = [v for v in (var_core + var_mod) if v in base_df.columns]
if not var_all:
    st.error("No usable numeric features found after loading. Ensure your Excel has mapped numeric columns.")
    st.stop()

# Cutoffs 0 → 600 step 25
gain_cutoffs = list(range(0, 601, 25))

# Utility to split by cutoff
def _make_split(df_base: pd.DataFrame, thr_val: float):
    df_tmp = df_base.copy()
    df_tmp["__Group__"] = np.where(
        pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val,
        f"≥{int(thr_val)}%", "Rest"
    )
    gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
    return df_tmp, gA_, gB_

# Compute per-cutoff median predicted probs over the whole base_df
thr_labels = []
series_N_med, series_C_med = [], []

for thr_val in gain_cutoffs:
    df_split, gA, gB = _make_split(base_df, float(thr_val))

    # Skip thresholds where either class is missing or tiny
    vc = df_split["__Group__"].value_counts()
    if (vc.get(gA, 0) < 10) or (vc.get(gB, 0) < 10):
        continue

    # Train both models on the split
    nca_model = _train_nca_or_lda(df_split, gA, gB, var_all) or {}
    cat_model = _train_catboost_once(df_split, gA, gB, var_all) or {}

    # If both empty, skip
    if not nca_model and not cat_model:
        continue

    # Build prediction frame (finite rows only for required feats)
    req_feats = []
    if nca_model: req_feats += nca_model.get("feats", [])
    if cat_model: req_feats += cat_model.get("feats", [])
    req_feats = sorted(set(req_feats))
    if not req_feats:
        continue

    Xdf = df_split[req_feats].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(Xdf.values).all(axis=1)
    df_pred = df_split.loc[mask].copy()
    if df_pred.shape[0] < 20:
        continue

    pN, pC = [], []
    if nca_model:
        pN = [ _nca_predict_proba_row(df_pred.iloc[i], nca_model) for i in range(df_pred.shape[0]) ]
    if cat_model:
        pC = [ _cat_predict_proba_row(df_pred.iloc[i], cat_model) for i in range(df_pred.shape[0]) ]

    pN = np.array([x for x in pN if np.isfinite(x)], dtype=float)
    pC = np.array([x for x in pC if np.isfinite(x)], dtype=float)

    if pN.size==0 and pC.size==0:
        continue

    thr_labels.append(int(thr_val))
    series_N_med.append(float(np.nanmedian(pN)*100.0) if pN.size else np.nan)
    series_C_med.append(float(np.nanmedian(pC)*100.0) if pC.size else np.nan)

if not thr_labels:
    st.info("Not enough data across cutoffs to train both classes. Try a broader DB.")
else:
    # Build tidy frame for Altair
    data = []
    for i, thr in enumerate(thr_labels):
        data.append({"GainCutoff_%": thr, "Series": "NCA: P(A)", "Value": series_N_med[i]})
        data.append({"GainCutoff_%": thr, "Series": "CatBoost: P(A)", "Value": series_C_med[i]})
    df_long = pd.DataFrame(data)

    # Chart (bars grouped by series at each cutoff)
    color_domain = ["NCA: P(A)", "CatBoost: P(A)"]
    color_range  = ["#10b981", "#8b5cf6"]  # green, purple

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("GainCutoff_%:O", title="Gain% cutoff (0 → 600, step 25)"),
            y=alt.Y("Value:Q", title="Median predicted probability for class A (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
            xOffset="Series:N",
            tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

    # Optional: PNG export via Matplotlib
    png_bytes = None
    try:
        pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
        series_names = list(pivot.columns)
        color_map = {"NCA: P(A)": "#10b981", "CatBoost: P(A)": "#8b5cf6"}
        colors = [color_map.get(s, "#999999") for s in series_names]
        thresholds = pivot.index.tolist()
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
        ax.set_ylabel("Median predicted probability for class A (%)")
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
vegaEmbed("#vis", spec, {{actions: true}});
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
