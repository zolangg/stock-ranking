# app.py — Premarket Stock Ranking (Median-only centers; Group UI + Gain% filter; 3σ coloring; delete UI)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== Optional ML imports (safe fallback) ==============================
_SK_AVAILABLE = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    _SK_AVAILABLE = False

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

# pattern-recognition model cache
ss.setdefault("sim_model", {})   # { scaler, X_train, X_scaled, y, rf, leaf_train, df_meta, feat_names }

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

# ======== 9-feature “face” for similarity (exact columns inside app) ========
FACE_VARS = [
    "MC_PM_Max_M",
    "Float_PM_Max_M",
    "Catalyst",
    "ATR_$",
    "Gap_%",
    "Max_Pull_PM_%",
    "PM_Vol_M",
    "PM$Vol/MC_%",
    "RVOL_Max_PM_cum",
]

# ============================== LASSO (unchanged) ==============================
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

def train_ratio_winsor_iso(df: pd.DataFrame, lo_q=0.01, hi_q=0.99) -> dict:
    eps = 1e-6
    mcap_series  = df["MC_PM_Max_M"]    if "MC_PM_Max_M"    in df.columns else df.get("MarketCap_M$")
    float_series = df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df.get("Float_M")
    need_min = {"ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"}
    if mcap_series is None or float_series is None or not need_min.issubset(df.columns): return {}

    PM  = pd.to_numeric(df["PM_Vol_M"],    errors="coerce").values
    DV  = pd.to_numeric(df["Daily_Vol_M"], errors="coerce").values
    valid_pm = np.isfinite(PM) & np.isfinite(DV) & (PM > 0) & (DV > 0)
    if valid_pm.sum() < 50: return {}

    ln_mcap   = np.log(np.clip(pd.to_numeric(mcap_series, errors="coerce").values,  eps, None))
    ln_gapf   = np.log(np.clip(pd.to_numeric(df["Gap_%"], errors="coerce").values,  0,   None) / 100.0 + eps)
    ln_atr    = np.log(np.clip(pd.to_numeric(df["ATR_$"], errors="coerce").values,  eps, None))
    ln_pm     = np.log(np.clip(pd.to_numeric(df["PM_Vol_M"], errors="coerce").values, eps, None))
    ln_pm_dol = np.log(np.clip(pd.to_numeric(df["PM_$Vol_M$"], errors="coerce").values, eps, None))
    ln_fr     = np.log(np.clip(pd.to_numeric(df["FR_x"], errors="coerce").values,   eps, None))
    ln_float_pmmax = np.log(np.clip(pd.to_numeric(df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df["Float_M"], errors="coerce").values, eps, None))
    maxpullpm      = pd.to_numeric(df.get("Max_Pull_PM_%", np.nan), errors="coerce").values
    ln_rvolmaxpm   = np.log(np.clip(pd.to_numeric(df.get("RVOL_Max_PM_cum", np.nan), errors="coerce").values, eps, None))
    pm_dol_over_mc = pd.to_numeric(df.get("PM$Vol/MC_%", np.nan), errors="coerce").values
    catalyst_raw   = df.get("Catalyst", np.nan)
    catalyst       = pd.to_numeric(catalyst_raw, errors="coerce").fillna(0.0).clip(0,1).values

    multiplier_all = np.maximum(DV / PM, 1.0)
    y_ln_all = np.log(multiplier_all)

    feats = [
        ("ln_mcap_pmmax",  ln_mcap),
        ("ln_gapf",        ln_gapf),
        ("ln_atr",         ln_atr),
        ("ln_pm",          ln_pm),
        ("ln_pm_dol",      ln_pm_dol),
        ("ln_fr",          ln_fr),
        ("catalyst",       catalyst),
        ("ln_float_pmmax", ln_float_pmmax),
        ("maxpullpm",      maxpullpm),
        ("ln_rvolmaxpm",   ln_rvolmaxpm),
        ("pm_dol_over_mc", pm_dol_over_mc),
    ]
    X_all = np.hstack([arr.reshape(-1,1) for _, arr in feats])

    mask = valid_pm & np.isfinite(y_ln_all) & np.isfinite(X_all).all(axis=1)
    if mask.sum() < 50: return {}
    X_all = X_all[mask]; y_ln = y_ln_all[mask]
    PMm = PM[mask]; DVv = DV[mask]

    n = X_all.shape[0]
    split = max(10, int(n * 0.8))
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr = y_ln[:split]

    winsor_bounds = {}
    name_to_idx = {name:i for i,(name,_) in enumerate(feats)}
    def _winsor_feature(col_idx):
        arr_tr = X_tr[:, col_idx]
        lo, hi = _compute_bounds(arr_tr[np.isfinite(arr_tr)])
        winsor_bounds[feats[col_idx][0]] = (lo, hi)
        X_tr[:, col_idx] = _apply_bounds(arr_tr, lo, hi)
        X_va[:, col_idx] = _apply_bounds(X_va[:, col_idx], lo, hi)
    for nm in ["maxpullpm", "pm_dol_over_mc"]:
        if nm in name_to_idx: _winsor_feature(name_to_idx[nm])

    mult_tr = np.exp(y_tr)
    m_lo, m_hi = _compute_bounds(mult_tr)
    mult_tr_w = _apply_bounds(mult_tr, m_lo, m_hi)
    y_tr = np.log(mult_tr_w)

    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs_tr = (X_tr - mu) / sd

    folds = _kfold_indices(len(y_tr), k=min(5, max(2, len(y_tr)//10)), seed=42)
    lam_grid = np.geomspace(0.001, 1.0, 26)
    cv_mse = []
    for lam in lam_grid:
        errs = []
        for vi in range(len(folds)):
            te_idx = folds[vi]; tr_idx = np.hstack([folds[j] for j in range(len(folds)) if j != vi])
            Xtr, ytr = Xs_tr[tr_idx], y_tr[tr_idx]
            Xte, yte = Xs_tr[te_idx], y_tr[te_idx]
            w = _lasso_cd_std(Xtr, ytr, lam=lam, max_iter=1400)
            yhat = Xte @ w
            errs.append(np.mean((yhat - yte)**2))
        cv_mse.append(np.mean(errs))
    lam_best = float(lam_grid[int(np.argmin(cv_mse))])
    w_l1 = _lasso_cd_std(Xs_tr, y_tr, lam=lam_best, max_iter=2000)
    sel = np.flatnonzero(np.abs(w_l1) > 1e-8)
    if sel.size == 0: return {}

    Xtr_sel = X_tr[:, sel]
    X_design = np.column_stack([np.ones(Xtr_sel.shape[0]), Xtr_sel])
    coef_ols, *_ = np.linalg.lstsq(X_design, y_tr, rcond=None)
    b0 = float(coef_ols[0]); bet = coef_ols[1:].astype(float)

    iso_bx = np.array([], dtype=float); iso_by = np.array([], dtype=float)
    if X_va.shape[0] >= 8:
        Xva_sel = X_va[:, sel]
        yhat_va_ln = (np.column_stack([np.ones(Xva_sel.shape[0]), Xva_sel]) @ coef_ols).astype(float)
        mult_pred_va = np.exp(yhat_va_ln)
        mult_true_all = np.maximum(DVv / PMm, 1.0)
        mult_va_true = mult_true_all[split:]
        finite = np.isfinite(mult_pred_va) & np.isfinite(mult_va_true)
        if finite.sum() >= 8 and np.unique(mult_pred_va[finite]).size >= 3:
            iso_bx, iso_by = _pav_isotonic(mult_pred_va[finite], mult_va_true[finite])

    return {
        "eps": eps,
        "terms": [feats[i][0] for i in sel],
        "b0": b0, "betas": bet, "sel_idx": sel.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
        "winsor_bounds": {k: (float(v[0]) if np.isfinite(v[0]) else np.nan,
                              float(v[1]) if np.isfinite(v[1]) else np.nan)
                          for k, v in winsor_bounds.items()},
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
        "feat_order": [nm for nm,_ in feats],
    }

# ============================== Predict ==============================
def predict_daily_calibrated(row: dict, model: dict) -> float:
    if not model or "betas" not in model: return np.nan
    eps = float(model.get("eps", 1e-6))
    feat_order = model["feat_order"]
    winsor_bounds = model.get("winsor_bounds", {})
    sel = model.get("sel_idx", [])
    b0 = float(model["b0"]); bet = np.array(model["betas"], dtype=float)

    def safe_log(v):
        v = float(v) if v is not None else np.nan
        return np.log(np.clip(v, eps, None)) if np.isfinite(v) else np.nan

    ln_mcap_pmmax  = safe_log(row.get("MC_PM_Max_M") or row.get("MarketCap_M$"))
    ln_gapf        = np.log(np.clip((row.get("Gap_%") or 0.0)/100.0 + eps, eps, None)) if row.get("Gap_%") is not None else np.nan
    ln_atr         = safe_log(row.get("ATR_$"))
    ln_pm          = safe_log(row.get("PM_Vol_M"))
    ln_pm_dol      = safe_log(row.get("PM_$Vol_M$"))
    ln_fr          = safe_log(row.get("FR_x"))
    catalyst       = 1.0 if (str(row.get("CatalystYN","No")).lower()=="yes" or float(row.get("Catalyst",0))>=0.5) else 0.0
    ln_float_pmmax = safe_log(row.get("Float_PM_Max_M") or row.get("Float_M"))
    maxpullpm      = float(row.get("Max_Pull_PM_%")) if row.get("Max_Pull_PM_%") is not None else np.nan
    ln_rvolmaxpm   = safe_log(row.get("RVOL_Max_PM_cum"))
    pm_dol_over_mc = float(row.get("PM$Vol/MC_%")) if row.get("PM$Vol/MC_%") is not None else np.nan

    feat_map = {
        "ln_mcap_pmmax":  ln_mcap_pmmax, "ln_gapf": ln_gapf, "ln_atr": ln_atr, "ln_pm": ln_pm,
        "ln_pm_dol": ln_pm_dol, "ln_fr": ln_fr, "catalyst": catalyst,
        "ln_float_pmmax": ln_float_pmmax, "maxpullpm": maxpullpm,
        "ln_rvolmaxpm": ln_rvolmaxpm, "pm_dol_over_mc": pm_dol_over_mc,
    }

    X_vec = []
    for nm in feat_order:
        v = feat_map.get(nm, np.nan)
        if not np.isfinite(v): return np.nan
        lo, hi = winsor_bounds.get(nm, (np.nan, np.nan))
        if np.isfinite(lo) or np.isfinite(hi):
            v = float(np.clip(v, lo if np.isfinite(lo) else v, hi if np.isfinite(hi) else v))
        X_vec.append(v)
    X_vec = np.array(X_vec, dtype=float)
    if not sel: return np.nan
    yhat_ln = b0 + float(np.dot(np.array(X_vec)[sel], bet))
    raw_mult = np.exp(yhat_ln) if np.isfinite(yhat_ln) else np.nan
    if not np.isfinite(raw_mult): return np.nan

    iso_bx = np.array(model.get("iso_bx", []), dtype=float)
    iso_by = np.array(model.get("iso_by", []), dtype=float)
    cal_mult = float(_iso_predict(iso_bx, iso_by, np.array([raw_mult]))[0]) if (iso_bx.size>=2 and iso_by.size>=2) else float(raw_mult)
    cal_mult = max(cal_mult, 1.0)

    PM = float(row.get("PM_Vol_M") or np.nan)
    if not np.isfinite(PM) or PM <= 0: return np.nan
    return float(PM * cal_mult)

# ============================== Summaries ==============================
def _summaries_median_and_mad(df_in: pd.DataFrame, var_all: list[str], group_col: str, labels_map=None):
    avail = [v for v in var_all if v in df_in.columns]
    if not avail:
        empty = pd.DataFrame()
        return {"med_tbl": empty, "mad_tbl": empty, "avail": []}
    g = df_in.groupby(group_col, observed=True)[avail]
    med_tbl = g.median(numeric_only=True).T
    mad_tbl = df_in.groupby(group_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad)).T
    if labels_map is not None:
        med_tbl = med_tbl.rename(columns=labels_map)
        mad_tbl = mad_tbl.rename(columns=labels_map)
    return {"med_tbl": med_tbl, "mad_tbl": mad_tbl, "avail": avail}

def _make_top_flag(series_pct: pd.Series, top_cut: int) -> tuple[pd.Series, float]:
    s = pd.to_numeric(series_pct, errors="coerce")
    mask = s.notna()
    if not mask.any(): return pd.Series(False, index=series_pct.index), np.nan
    thr = float(np.nanpercentile(s[mask].values, 100 - top_cut))
    flag = (s >= thr) & mask
    return flag, thr

# ============================== Upload / Build ==============================
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

def _safe_to_binary(v):
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        fv = float(sv); return 1 if fv >= 0.5 else 0
    except: return np.nan

def _safe_to_binary_float(v):
    x = _safe_to_binary(v)
    if np.isnan(x): return np.nan
    return float(x)

# ------------- Pattern recognition: training from DB -------------
def _build_similarity_model_from_df(df: pd.DataFrame, df_meta_cols: list[str]):
    model = {}
    # Need FT labels and required face vars
    needed = ["FT01"] + FACE_VARS
    if not set(needed).issubset(df.columns):
        return {}

    df_face = df[needed].dropna()
    if df_face.shape[0] < 30:
        return {}

    # Targets
    y = df_face["FT01"].astype(int).values
    # Features
    X = df_face[FACE_VARS].astype(float).values

    # Meta rows aligned with X rows
    df_meta = df_face.reset_index(drop=True)
    meta_keep = [c for c in df.columns if c in df_meta_cols]
    meta_frame = df.loc[df_face.index, meta_keep].reset_index(drop=True)

    # Standardize
    if _SK_AVAILABLE:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
    else:
        # manual standardization
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        scaler = {"mu": mu, "sd": sd}
        Xs = (X - mu) / sd

    rf = None
    leaf_train = None
    if _SK_AVAILABLE:
        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(Xs, y)
            leaf_train = rf.apply(Xs)  # (n_samples, n_trees)
        except Exception:
            rf = None
            leaf_train = None

    model = {
        "scaler": scaler,
        "X_train": X,
        "X_scaled": Xs,
        "y": y,
        "rf": rf,
        "leaf_train": leaf_train,
        "feat_names": FACE_VARS,
        "df_meta": meta_frame,  # must include TickerDB, FT01, Max_Push_Daily_%, etc.
    }
    return model

def _scale_query(x_vec: np.ndarray, scaler):
    if _SK_AVAILABLE and isinstance(scaler, StandardScaler):
        return scaler.transform(x_vec.reshape(1, -1)).ravel()
    else:
        mu = scaler["mu"]; sd = scaler["sd"]
        return ((x_vec - mu) / sd).ravel()

def _cosine_distance_to_train(x_scaled: np.ndarray, X_scaled_train: np.ndarray):
    # 1 - cosine similarity
    # cos = (a·b) / (||a|| ||b||)
    denom = (np.linalg.norm(x_scaled) * np.linalg.norm(X_scaled_train, axis=1))
    denom[denom == 0] = 1e-12
    cos_sim = (X_scaled_train @ x_scaled) / denom
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim  # distance

def _corr_distance_to_train(x_scaled: np.ndarray, X_scaled_train: np.ndarray):
    # correlation distance = 1 - Pearson corr
    xs = (x_scaled - x_scaled.mean())
    xs_sd = xs.std()
    if xs_sd == 0: xs_sd = 1.0
    xsn = xs / xs_sd
    Xt = X_scaled_train - X_scaled_train.mean(axis=1, keepdims=True)
    Xt_sd = X_scaled_train.std(axis=1, keepdims=True)
    Xt_sd[Xt_sd == 0] = 1.0
    Xtn = Xt / Xt_sd
    corr = (Xtn @ xsn) / (xsn.size - 1)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr

def _rf_proximity_distance(x_scaled: np.ndarray, rf, leaf_train: np.ndarray):
    # distance = 1 - proximity; proximity = fraction of trees sharing same leaf
    if rf is None or leaf_train is None:
        return None
    leaf_q = rf.apply(x_scaled.reshape(1, -1)).ravel()  # (n_trees,)
    same = (leaf_train == leaf_q)  # (n_samples, n_trees)
    prox = same.mean(axis=1)
    return 1.0 - prox  # smaller = closer

def _row_to_face_vector(row: dict):
    vals = []
    for f in FACE_VARS:
        v = row.get(f)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        vals.append(float(v))
    return np.array(vals, dtype=float)

# ============================== Upload/build button handler ==============================
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

            # map fields
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
            if cand_catalyst:
                df["Catalyst"] = raw[cand_catalyst].map(_safe_to_binary_float)

            # derived
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # scale % fields (DB stores fractions)
            if "Gap_%" in df.columns:            df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "PM_Vol_%" in df.columns:         df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns:    df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT groups
            df["FT01"] = pd.Series(raw[col_group]).map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

            # Max Push Daily (%) fraction -> %
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            if pmh_col is not None:
                pmh_raw = pd.to_numeric(pd.Series(raw[pmh_col]).map(_to_float), errors="coerce")
                df["Max_Push_Daily_%"] = pmh_raw * 100.0
            else:
                df["Max_Push_Daily_%"] = np.nan

            # keep Ticker for matching table if present
            tcol = _pick(raw, ["ticker","symbol","name"])
            if tcol is not None:
                df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()
            else:
                df["TickerDB"] = ""

            ss.base_df = df
            ss.var_core = [v for v in VAR_CORE if v in df.columns]
            ss.var_moderate = [v for v in VAR_MODERATE if v in df.columns]

            # train model once (on full base) — unchanged
            ss.lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            # -------- Build similarity model on the 9-variable "face" --------
            # meta cols we want to show in the neighbor table
            meta_cols = ["TickerDB","FT01","Max_Push_Daily_%"] + FACE_VARS
            ss.sim_model = _build_similarity_model_from_df(df, df_meta_cols=meta_cols) or {}

            st.success(f"Loaded “{sel_sheet}”. Base ready.")
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
        rvol_pm_cum = st.number_input("Premarket Max RVOL", 0.0, step=0.01, format="%.2f")
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
    pred = predict_daily_calibrated(row, ss.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan
    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Alignment ==============================
st.markdown("### Alignment")

# --- compact top row: radios (left) + Gain% dropdown (right) ---
col_mode, col_gain = st.columns([2.8, 1.0])
with col_mode:
    mode = st.radio(
        "",
        [
            "FT vs Fail (Gain% cutoff on FT=1 only)",
            "FT=1 (High vs Low cutoff)",
            "Gain% vs Rest",
        ],
        horizontal=True,
        key="cmp_mode",
        label_visibility="collapsed",
    )
with col_gain:
    gain_choices = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    gain_min = st.selectbox(
        "",
        gain_choices,
        index=gain_choices.index(100) if 100 in gain_choices else 0,
        key="gain_min_pct",
        help="Threshold on Max Push Daily (%).",
        label_visibility="collapsed",
    )

# --- base data guardrails ---
base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty:
    if ss.rows:
        st.info("Upload DB and click **Build model stocks** to compute group centers first.")
    else:
        st.info("Upload DB (and/or add at least one stock) to compute alignment.")
    st.stop()

if "Max_Push_Daily_%" not in base_df.columns:
    st.error("Column “Max Push Daily (%)” not found in DB (expected as Max_Push_Daily_% after load).")
    st.stop()
if "FT01" not in base_df.columns:
    st.error("FT01 column not found (expected after load).")
    st.stop()

# ---------- build comparison dataframe + group labels ----------
df_cmp = base_df.copy()
thr = float(gain_min)

if mode == "Gain% vs Rest":
    df_cmp["__Group__"] = np.where(
        pd.to_numeric(df_cmp["Max_Push_Daily_%"], errors="coerce") >= thr,
        f"≥{int(thr)}%",
        "Rest",
    )
    gA, gB = f"≥{int(thr)}%", "Rest"
    status_line = f"Gain% split at ≥ {int(thr)}%"

elif mode == "FT=1 (High vs Low cutoff)":
    df_cmp = df_cmp[df_cmp["FT01"] == 
