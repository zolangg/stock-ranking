# app.py — Premarket Ranking with OOF Pred Model + Head-Specific Feature Selection + NCA & CatBoost-Leaf kernel-kNN
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, warnings
warnings.filterwarnings("ignore")

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking (NCA + CatBoost Leaf)", layout="wide")
st.title("Premarket Stock Ranking — NCA + Leaf-kNN Similarity")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])           # user-added stocks
ss.setdefault("last", {})           # last added
ss.setdefault("pred_full", {})      # full daily-volume predictor (for queries)
ss.setdefault("base_df", pd.DataFrame())

ss.setdefault("nca_model", {})      # medians, scaler, pca(optional), nca, X_emb, y, feat_names, df_meta, guard
ss.setdefault("leaf_model", {})     # cat, emb, metric, y, feat_names, df_meta
ss.setdefault("feat_sel", {"nca": [], "leaf": []})

# ============================== Core deps ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import pairwise_distances, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError:
    st.error("Missing scikit-learn. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

# CatBoost ONLY (no RF fallback)
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# ============================== Small utils ==============================
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
        if "," in ss_ and "." not in ss_:
            ss_ = ss_.replace(",", ".")
        else:
            ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

def _safe_to_binary(v):
    sv = str(v).strip().lower()
    if sv in {"1","true","yes","y","t"}: return 1
    if sv in {"0","false","no","n","f"}: return 0
    try:
        fv = float(sv); return 1 if fv >= 0.5 else 0
    except: return np.nan

def _safe_to_binary_float(v):
    x = _safe_to_binary(v)
    return float(x) if x in (0,1) else np.nan

def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med = float(np.median(s)); return float(np.median(np.abs(s - med)))

def _compute_bounds(arr: np.ndarray, lo_q=0.01, hi_q=0.99):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (np.nan, np.nan)
    return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

def _apply_bounds(arr: np.ndarray, lo: float, hi: float):
    out = arr.copy()
    if np.isfinite(lo): out = np.maximum(out, lo)
    if np.isfinite(hi): out = np.minimum(out, hi)
    return out

# ---------- Isotonic ----------
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

# ============================== VARIABLES ==============================
FEAT12 = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM_$Vol_M$","PM$Vol/MC_%",
    "RVOL_Max_PM_cum","FR_x","PM_Vol_%"
]

# ============================== Daily Volume Predictor (LASSO→OLS→Isotonic) ==============================
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
            te_idx = np.array(folds[vi])
            tr_idx = np.hstack([folds[j] for j in range(len(folds)) if j != vi])
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
        "terms": [],  # not used downstream
        "b0": b0, "betas": bet, "sel_idx": sel.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
        "winsor_bounds": {},  # not needed here
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
        "feat_order": [nm for nm,_ in feats],
    }

def _predict_daily_row(row: dict, model: dict) -> float:
    if not model or "betas" not in model: return np.nan
    eps = float(model.get("eps", 1e-6))
    feat_order = model["feat_order"]; sel = model.get("sel_idx", [])
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

def build_oof_predvol(df: pd.DataFrame, y_col="FT01", k=5, seed=42):
    """Return: oof_pred (Series), full_model (dict)"""
    idx = df.index.to_numpy()
    y = df[y_col].astype(int).values
    folds = StratifiedKFold(n_splits=min(k, max(2, min(5, np.unique(y).size*2))), shuffle=True, random_state=seed)
    oof = pd.Series(np.nan, index=df.index, dtype=float)
    for tr_idx, va_idx in folds.split(idx.reshape(-1,1), y):
        tr_ids = idx[tr_idx]; va_ids = idx[va_idx]
        model_tr = train_ratio_winsor_iso(df.loc[tr_ids])
        if not model_tr: continue
        for i in va_ids:
            row = df.loc[i].to_dict()
            oof.at[i] = _predict_daily_row(row, model_tr)
    # full model
    full_model = train_ratio_winsor_iso(df)
    return oof, full_model

# ============================== Kernel helpers ==============================
def _elbow_kstar(sorted_vals, k_min=3, k_max=15, max_rank=30):
    n = len(sorted_vals)
    if n <= k_min: return max(1, n)
    upto = min(max_rank, n-1)
    if upto < 2: return min(k_max, max(k_min, n))
    gaps = sorted_vals[:upto] - sorted_vals[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

def _kernel_weights(dists, k_star, floor=1e-12):
    d = np.asarray(dists)[:k_star]
    if d.size == 0: return d, np.array([])
    positive = d[d>0]
    bw = np.median(positive) if positive.size else (np.mean(d)+1e-6)
    bw = max(bw, 1e-6)
    w = np.exp(-(d/bw)**2)
    return d, w

# ============================== Heads ==============================
def train_nca_head(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    # Impute medians (fit on train), standardize, optional PCA, NCA
    use = df[feat_names + [y_col]].copy()
    y = use[y_col].astype(int).values
    X = use[feat_names].astype(float)

    medians = X.median(axis=0, skipna=True)
    X_imp = X.fillna(medians)

    scaler = StandardScaler().fit(X_imp.values)
    Xs = scaler.transform(X_imp.values)

    # optional PCA for small n (keep up to 95% variance, min 6 comps cap)
    n_components_pca = min(max(6, 1), Xs.shape[1])
    pca = None
    if Xs.shape[1] > 10 and Xs.shape[0] < 400:
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        Xp = pca.fit_transform(Xs)
    else:
        Xp = Xs

    nca = NeighborhoodComponentsAnalysis(
        n_components=min(6, Xp.shape[1]), random_state=42, max_iter=250
    )
    X_emb = nca.fit_transform(Xp, y)

    guard = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    guard.fit(X_emb)

    meta_cols = []
    for c in ("TickerDB","Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta = df[meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=df.index)

    return {
        "medians": medians.to_dict(),
        "scaler": scaler,
        "pca": pca,
        "nca": nca,
        "X_emb": X_emb,
        "y": y,
        "feat_names": feat_names,
        "df_meta": meta,
        "guard": guard,
        "head_name": "NCA-kNN"
    }

def _nca_score(model, row_dict):
    feats = model["feat_names"]
    x = np.array([row_dict.get(f, np.nan) for f in feats], dtype=float)[None, :]
    # impute medians
    med = model.get("medians", {})
    if med:
        xr = x.copy()
        for j, f in enumerate(feats):
            if not np.isfinite(xr[0, j]):
                m = med.get(f, np.nan)
                xr[0, j] = m if np.isfinite(m) else 0.0
        x = xr
    xs = model["scaler"].transform(x)
    pca = model.get("pca", None)
    if pca is not None:
        xp = pca.transform(xs)
    else:
        xp = xs
    xn = model["nca"].transform(xp)
    Xn = model["X_emb"]; y = model["y"]
    d = pairwise_distances(xn, Xn, metric="euclidean").ravel()
    order = np.argsort(d); d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(15, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    mask = np.isfinite(w_top) & (w_top > 1e-12)
    if mask.sum() == 0: return 0.0
    idx = idx[mask]; w_top = w_top[mask]
    s = w_top.sum()
    if s <= 0: return 0.0
    w_norm = w_top / s
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0
    return p1

def train_leaf_head(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    if not CATBOOST_AVAILABLE:
        st.error("CatBoost is required for the Leaf head. Add `catboost` to requirements.txt."); st.stop()

    use = df[feat_names + [y_col]].copy()
    y = use[y_col].astype(int).values
    X = use[feat_names].astype(float).values  # CatBoost handles NaNs

    # split small eval set for early stopping
    n = len(y)
    idx = np.arange(n)
    rs = np.random.RandomState(42)
    rs.shuffle(idx)
    split = max(8, int(n*0.85))
    tr_idx, ev_idx = idx[:split], idx[split:]

    train_pool = Pool(X[tr_idx], label=y[tr_idx])
    eval_pool  = Pool(X[ev_idx], label=y[ev_idx]) if ev_idx.size>0 else None

    class_w = [1.0, float((y==0).sum() / max(1,(y==1).sum()))]
    cat = CatBoostClassifier(
        depth=6, learning_rate=0.06, loss_function="Logloss",
        random_seed=42, iterations=1200, l2_leaf_reg=6,
        class_weights=class_w, verbose=False, early_stopping_rounds=50,
        bootstrap_type='No', random_strength=0.0
    )
    if eval_pool is not None:
        cat.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    else:
        cat.fit(train_pool)

    emb = cat.calc_leaf_indexes(X)  # (n, n_trees)
    emb = np.asarray(emb, dtype=float)  # cosine on floats
    metric = "cosine"

    meta_cols = []
    for c in ("TickerDB","Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta = df[meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame(index=df.index)

    return {
        "cat": cat,
        "emb": emb,
        "metric": metric,
        "y": y,
        "feat_names": feat_names,
        "df_meta": meta,
        "head_name": "CatBoost-leaf-kNN"
    }

def _leaf_score(model, row_dict):
    feats = model["feat_names"]
    x = np.array([row_dict.get(f, np.nan) for f in feats], dtype=float)[None, :]
    cat: CatBoostClassifier = model["cat"]
    emb_q = cat.calc_leaf_indexes(x)
    emb_q = np.asarray(emb_q, dtype=float)
    Emb = model["emb"]; y = model["y"]; metric = model.get("metric", "cosine")
    d = pairwise_distances(emb_q, Emb, metric=metric).ravel()
    order = np.argsort(d); d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(15, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    mask = np.isfinite(w_top) & (w_top > 1e-12)
    if mask.sum() == 0: return 0.0
    idx = idx[mask]; w_top = w_top[mask]
    s = w_top.sum()
    if s <= 0: return 0.0
    w_norm = w_top / s
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0
    return p1

# ============================== CV metric & feature selection ==============================
def _cv_scores_for_head(df: pd.DataFrame, feats: list[str], head: str, folds=5, seed=42):
    """Return list of per-fold metrics (ROC AUC if well-defined else accuracy)"""
    y = df["FT01"].astype(int).values
    idx = df.index.to_numpy()
    skf = StratifiedKFold(n_splits=min(folds, max(2, np.unique(y).size*2))), 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    metrics = []
    for tr_ids, va_ids in skf.split(idx.reshape(-1,1), y):
        tr_idx, va_idx = idx[tr_ids], idx[va_ids]
        tr_df, va_df = df.loc[tr_idx], df.loc[va_idx]
        if head == "nca":
            mdl = train_nca_head(tr_df, feats)
            preds = [ _nca_score(mdl, va_df.loc[i, :].to_dict()) for i in va_df.index ]
        else:
            mdl = train_leaf_head(tr_df, feats)
            preds = [ _leaf_score(mdl, va_df.loc[i, :].to_dict()) for i in va_df.index ]

        y_true = va_df["FT01"].astype(int).values
        p = np.asarray(preds, dtype=float)
        # try ROC AUC; if invalid (single class), fallback to acc
        try:
            if np.unique(y_true).size == 2 and not np.all(np.isnan(p)):
                auc = roc_auc_score(y_true, p)
                metrics.append(auc)
                continue
        except Exception:
            pass
        y_hat = (p >= 50.0).astype(int)
        acc = (y_hat == y_true).mean()
        metrics.append(acc)
    return metrics

def select_features_with_drop_one(df: pd.DataFrame, base_feats: list[str], head: str, stability=0.6, max_keep=10, min_keep=6):
    # Start from all available base feats in df
    feats = [f for f in base_feats if f in df.columns]
    if len(feats) <= min_keep:
        return feats

    base_fold_scores = _cv_scores_for_head(df, feats, head=head)
    base_mean = float(np.mean(base_fold_scores)) if base_fold_scores else 0.0

    keep = []
    for f in feats:
        test_feats = [x for x in feats if x != f]
        fold_scores = _cv_scores_for_head(df, test_feats, head=head)
        mean_sc = float(np.mean(fold_scores)) if fold_scores else 0.0
        delta = base_mean - mean_sc  # positive delta means f helps
        if delta > 0:
            keep.append(f)

    if len(keep) < min_keep:
        # fallback: keep top `min_keep` by single-feature drop delta
        deltas = []
        for f in feats:
            test_feats = [x for x in feats if x != f]
            fold_scores = _cv_scores_for_head(df, test_feats, head=head)
            mean_sc = float(np.mean(fold_scores)) if fold_scores else 0.0
            deltas.append((f, base_mean - mean_sc))
        deltas.sort(key=lambda x: x[1], reverse=True)
        keep = [f for f,_ in deltas[:min_keep]]

    if len(keep) > max_keep:
        keep = keep[:max_keep]
    return keep

# ============================== Upload / Build ==============================
st.subheader("Upload Database")
uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build models", use_container_width=True, key="db_build_btn")

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
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_safe_to_binary_float)

            # derived FR_x and PM$Vol/MC_%
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
            df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["FT01"] = df["FT01"].astype(int)

            # OOF PredVol_M for PM_Vol_% (no leakage)
            if {"PM_Vol_M","Daily_Vol_M","ATR_$","Gap_%"}.issubset(df.columns):
                oof, full_model = build_oof_predvol(df)
                ss.pred_full = full_model or {}
                df["PredVol_M_OOF"] = oof
                with np.errstate(divide='ignore', invalid='ignore'):
                    df["PM_Vol_%"] = 100.0 * df["PM_Vol_M"] / df["PredVol_M_OOF"]
            else:
                ss.pred_full = {}
                df["PredVol_M_OOF"] = np.nan

            ss.base_df = df

            # Clean training rows: keep those with FT01 and at least some features
            feat_base = [f for f in FEAT12 if f in df.columns]
            train_df = df.dropna(subset=["FT01"]).copy()

            # Head-specific feature selection
            feat_nca  = select_features_with_drop_one(train_df, feat_base, head="nca",  max_keep=10, min_keep=6)
            feat_leaf = select_features_with_drop_one(train_df, feat_base, head="leaf", max_keep=10, min_keep=6)
            ss.feat_sel = {"nca": feat_nca, "leaf": feat_leaf}

            # Train heads
            ss.nca_model  = train_nca_head(train_df, feat_nca) if len(feat_nca) >= 3 else {}
            ss.leaf_model = train_leaf_head(train_df, feat_leaf) if (CATBOOST_AVAILABLE and len(feat_leaf) >= 3) else {}

            st.success(f"Loaded “{sel_sheet}”. Heads ready. NCA feats: {feat_nca}. Leaf feats: {feat_leaf}.")
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
    fr   = (pm_vol / float_pm) if float_pm > 0 else np.nan
    pmmc = (pm_dol / mc_pmmax * 100.0) if mc_pmmax > 0 else np.nan
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
    # Predicted daily volume from FULL model (queries only)
    pred = _predict_daily_row(row, ss.get("pred_full", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Similarity Bars ==============================
st.markdown("### Similarity (FT=1)")

def _bar(pct, label):
    pct = 0.0 if pct is None or not np.isfinite(pct) else float(pct)
    w = max(0.0, min(100.0, pct))
    color = "#3b82f6" if label == "NCA" else ("#10b981" if label == "Leaf" else "#111827")
    return f"""
<div style="display:flex;align-items:center;gap:8px;">
  <div style="width:140px;height:12px;background:#eee;border-radius:8px;position:relative;overflow:hidden">
    <span style="position:absolute;left:0;top:0;bottom:0;width:{w}%;background:{color};"></span>
  </div>
  <div style="font-size:12px;color:#374151;min-width:52px;">{label}: {int(round(w))}%</div>
</div>"""

if not ss.rows:
    st.info("Add a stock above to see similarity.")
else:
    feat_nca  = ss.feat_sel.get("nca", [])
    feat_leaf = ss.feat_sel.get("leaf", [])
    for row in ss.rows[::-1]:
        tkr = row.get("Ticker","—")
        # head scores
        p_nca = _nca_score(ss.nca_model, row) if ss.nca_model else np.nan
        p_leaf = _leaf_score(ss.leaf_model, row) if ss.leaf_model else np.nan
        vals = [v for v in [p_nca, p_leaf] if np.isfinite(v)]
        p_avg = float(np.mean(vals)) if vals else np.nan

        st.markdown(f"**{tkr}**")
        st.markdown(_bar(p_nca, "NCA"), unsafe_allow_html=True)
        st.markdown(_bar(p_leaf, "Leaf"), unsafe_allow_html=True)
        st.markdown(_bar(p_avg, "Average"), unsafe_allow_html=True)

        # summary of inputs (12 vars + PredVol)
        def fmt(x): 
            return "" if x is None or (isinstance(x,float) and not np.isfinite(x)) else f"{x:.3f}"
        cols = st.columns(4)
        fields = ["MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%","Max_Pull_PM_%",
                  "PM_Vol_M","PM_$Vol_M$","PM$Vol/MC_%","RVOL_Max_PM_cum","FR_x","PM_Vol_%","PredVol_M"]
        labels = ["MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%","Max_Pull_PM_%",
                  "PM_Vol_M","PM_$Vol_M$","PM$Vol/MC_%","RVOL_Max_PM_cum","FR_x","PM_Vol_%","PredVol_M"]
        for i, lab in enumerate(labels):
            col = cols[i % 4]
            col.caption(f"{lab}: {fmt(row.get(fields[i]))}")
        st.markdown("---")

# ============================== Notes ==============================
st.caption("NCA: standardized metric learning on selected features. Leaf: CatBoost leaf-embedding + kernel-kNN. Average is the mean of available heads.")
