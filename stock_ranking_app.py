# app.py — Premarket Ranking with Pred Model + NCA kernel-kNN + CatBoost Leaf-Embedding kNN (no RF fallback)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking (NCA + CatBoost-Leaf kNN)", layout="wide")
st.title("Premarket Stock Ranking — NCA + CatBoost-Leaf kNN Similarity")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("lassoA", {})           # daily volume predictor (multiplier model)
ss.setdefault("base_df", pd.DataFrame())

ss.setdefault("nca_model", {})        # scaler, nca, X_emb, y, feat_names, df_meta, guard
ss.setdefault("leaf_model", {})       # scaler, cat_model, emb (train leaf indexes), y, feat_names, df_meta

# ============================== Core deps ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.ensemble import IsolationForest
except ModuleNotFoundError:
    st.error("Missing scikit-learn. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    st.error("Missing CatBoost. Add `catboost` to requirements.txt and redeploy.")
    st.stop()

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

# ---------- MAD & winsor ----------
def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))

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

# ============================== Features ==============================
# 9 base + 2 derived (PM_Vol_% via predicted daily volume; FR_x via float)
RF_FEATURES = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum",
    "FR_x","PM_Vol_%"
]

# ============================== Daily Volume Predictor ==============================
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

def predict_daily_calibrated_df(df: pd.DataFrame, model: dict) -> pd.Series:
    """Vectorized prediction for base_df to compute PredVol_M and PM_Vol_%."""
    if not model or "betas" not in model: return pd.Series(np.nan, index=df.index)
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        out.append(predict_daily_calibrated(row, model))
    return pd.Series(out, index=df.index, dtype=float)

# ============================== Heads: NCA + CatBoost Leaf-Embedding ==============================
def _build_nca(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    need = [y_col] + feat_names
    df_face = df[need].dropna()
    if df_face.shape[0] < 30:
        return {}

    y = df_face[y_col].astype(int).values
    X = df_face[feat_names].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # NCA in standardized space
    nca = NeighborhoodComponentsAnalysis(
        n_components=min(Xs.shape[1], 8), random_state=42, max_iter=300
    )
    Xn = nca.fit_transform(Xs, y)

    # Outlier guard (optional)
    guard = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    guard.fit(Xn)

    # meta
    meta_cols = ["FT01"]
    for c in ("TickerDB","Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True)

    return {"scaler":scaler, "nca":nca, "X_emb":Xn, "y":y,
            "feat_names":feat_names, "df_meta":meta, "guard": guard, "head_name":"NCA-kNN"}

def _build_leaf_head(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    need = [y_col] + feat_names
    df_face = df[need].dropna()
    if df_face.shape[0] < 30:
        return {}

    y = df_face[y_col].astype(int).values
    X = df_face[feat_names].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    cat = CatBoostClassifier(
        depth=6, learning_rate=0.08, loss_function="Logloss",
        random_state=42, iterations=600, verbose=False, l2_leaf_reg=6,
        class_weights=[1.0, float((y==0).sum() / max(1,(y==1).sum()))]
    )
    cat.fit(Xs, y)
    leaf_mat = cat.calc_leaf_indexes(Xs)  # (n_samples, n_trees)
    leaf_mat = np.array(leaf_mat, dtype=int)
    emb = leaf_mat  # integer leaf IDs; cosine on index vectors works OK across trees

    # meta
    meta_cols = ["FT01"]
    for c in ("TickerDB","Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True)

    return {"scaler":scaler, "cat_model":cat, "emb":emb, "y":y,
            "feat_names":feat_names, "df_meta":meta, "head_name":"CatBoost-leaf-kNN", "metric":"cosine"}

def _elbow_kstar(sorted_vals, k_min=3, k_max=25, max_rank=30):
    n = len(sorted_vals)
    if n <= k_min: return max(1, n)
    upto = min(max_rank, n-1)
    if upto < 2: return min(k_max, max(k_min, n))
    gaps = sorted_vals[:upto] - sorted_vals[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

def _kernel_weights(dists, k_star):
    d = np.asarray(dists)[:k_star]
    if d.size == 0: return d, np.array([])
    # Adaptive bandwidth: median of top-k distances (avoid 0)
    bw = np.median(d[d>0]) if np.any(d>0) else (np.mean(d)+1e-6)
    bw = max(bw, 1e-6)
    w = np.exp(-(d/bw)**2)
    return d, w

def _nca_score_and_neighbors(model, row_dict, feat_names):
    # vectorize row
    vec = []
    for f in feat_names:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)
    xn = model["nca"].transform(xs)

    # outlier guard
    oos = model["guard"].decision_function(xn.reshape(1,-1))[0]  # higher=more inlier
    Xn = model["X_emb"]; y = model["y"]

    # distances in NCA space (euclidean)
    d = pairwise_distances(xn, Xn, metric="euclidean").ravel()
    order = np.argsort(d)
    d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(25, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0,"k":0,"neighbors":[],"oos":float(oos)}

    y_top = y[idx]
    w_norm = w_top / (w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y_top==1).astype(float), w_norm)) * 100.0

    # neighbors info
    meta = model["df_meta"]
    rows = [{"__group__": f"NCA Top-{k_star} (kernel-weighted)"}]
    for rnk, (ii, dist, ww) in enumerate(zip(idx, d_top, w_norm), 1):
        rec = {"#": rnk, "FT": int(y[ii]), "Weight": float(ww), "Dist": float(dist)}
        if not meta.empty:
            m = meta.iloc[ii]
            rec["Ticker"] = m.get("TickerDB", m.get("Ticker", ""))
            for f in feat_names:
                val = m.get(f, np.nan)
                rec[f] = float(val) if (val is not None and np.isfinite(val)) else None
        rows.append(rec)
    return {"p1": p1, "k": int(k_star), "neighbors": rows, "oos": float(oos)}

def _leaf_score_and_neighbors(model, row_dict, feat_names):
    vec = []
    for f in feat_names:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)

    # embed query
    leaf_row = model["cat_model"].calc_leaf_indexes(xs)
    emb_q = np.array(leaf_row, dtype=int)

    Emb = model["emb"]
    metric = model.get("metric","cosine")

    # distances in leaf space
    d = pairwise_distances(emb_q, Emb, metric=metric).ravel()
    order = np.argsort(d)
    d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(25, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0,"k":0,"neighbors":[]}

    y = model["y"]
    w_norm = w_top / (w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0

    meta = model["df_meta"]
    rows = [{"__group__": f"{model['head_name']} Top-{k_star} (kernel-weighted)"}]
    for rnk, (ii, dist, ww) in enumerate(zip(idx, d_top, w_norm), 1):
        rec = {"#": rnk, "FT": int(y[ii]), "Weight": float(ww), "Dist": float(dist)}
        if not meta.empty:
            m = meta.iloc[ii]
            rec["Ticker"] = m.get("TickerDB", m.get("Ticker", ""))
            for f in feat_names:
                val = m.get(f, np.nan)
                rec[f] = float(val) if (val is not None and np.isfinite(val)) else None
        rows.append(rec)
    return {"p1": p1, "k": int(k_star), "neighbors": rows}

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
            add_num(df, "Daily_Vol_M",      ["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])
            add_num(df, "Max_Pull_PM_%",    ["max pull pm (%)","max pull pm %","max pull pm","max_pull_pm_%"])
            add_num(df, "RVOL_Max_PM_cum",  ["rvol max pm (cum)","rvol max pm cum","rvol_max_pm (cum)","rvol_max_pm_cum","premarket max rvol","premarket max rvol (cum)"])
            # catalyst
            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_safe_to_binary_float)

            # derived basic ratios (pre-prediction)
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # % fields from fractions if needed (DB commonly stores as fractions)
            if "Gap_%" in df.columns:            df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns:    df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT labels
            df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()

            # Max Push Daily (%) if present (not used in features here)
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            if pmh_col is not None:
                pmh_raw = pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")
                df["Max_Push_Daily_%"] = pmh_raw * 100.0
            else:
                df["Max_Push_Daily_%"] = np.nan

            # ================== Train daily-volume predictor ==================
            # Need FR_x to exist before training; require min columns:
            need_min = {"ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"}
            if not need_min.issubset(df.columns):
                st.error(f"DB missing columns for daily-volume model: {sorted(need_min - set(df.columns))}")
                st.stop()
            ss.lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            # Predict daily volume for each DB row -> PM_Vol_% (pred-based)
            pred_series = predict_daily_calibrated_df(df, ss.lassoA)
            df["PredVol_M"] = pred_series
            df["PM_Vol_%"] = (df["PM_Vol_M"] / df["PredVol_M"] * 100.0).replace([np.inf,-np.inf], np.nan)

            ss.base_df = df

            # Build similarity heads (require RF_FEATURES present)
            miss = [c for c in RF_FEATURES if c not in df.columns]
            if miss:
                st.error(f"DB missing required similarity features: {miss}")
                st.stop()

            ss.nca_model  = _build_nca(df, RF_FEATURES, y_col="FT01")
            ss.leaf_model = _build_leaf_head(df, RF_FEATURES, y_col="FT01")

            if not ss.nca_model or not ss.leaf_model:
                st.error("Could not build models. Need at least 30 non-NaN rows for all required features.")
                st.stop()

            st.success(f"Loaded “{sel_sheet}”. Models ready. Rows used: NCA={ss.nca_model['X_emb'].shape[0]}, Leaf={ss.leaf_model['emb'].shape[0]}")
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
    # predict daily volume -> PM_Vol_% (pred-based)
    pred = predict_daily_calibrated(row, ss.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Alignment (NCA vs CatBoost-Leaf) ==============================
st.markdown("### Alignment")

base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty or (not ss.get("nca_model") or not ss.get("leaf_model")):
    st.info("Upload DB and click **Build models** to compute embeddings first.")
    st.stop()

# Build summaries
summaries, details = [], {}
for row in ss.rows:
    tkr = row.get("Ticker","—")

    # NCA
    s_nca = _nca_score_and_neighbors(ss.nca_model, row, RF_FEATURES)
    # Leaf (CatBoost)
    s_leaf = _leaf_score_and_neighbors(ss.leaf_model, row, RF_FEATURES)

    if not s_nca and not s_leaf:
        summaries.append({"Ticker": tkr, "NCA":0.0,"Leaf":0.0,"Avg":0.0,"kN":0,"kL":0})
        details[tkr] = [{"__group__":"Missing inputs for similarity (need all 11 variables)."}]
        continue

    p1_n = float(s_nca["p1"]) if s_nca else 0.0
    p1_l = float(s_leaf["p1"]) if s_leaf else 0.0
    p1_avg = (p1_n + p1_l) / 2.0

    summaries.append({
        "Ticker": tkr, "NCA": p1_n, "Leaf": p1_l, "Avg": p1_avg,
        "kN": int(s_nca["k"]) if s_nca else 0, "kL": int(s_leaf["k"]) if s_leaf else 0
    })

    rows = []
    if s_nca and s_nca.get("neighbors"): rows.extend(s_nca["neighbors"])
    if s_leaf and s_leaf.get("neighbors"): rows.extend(s_leaf["neighbors"])
    details[tkr] = rows if rows else [{"__group__":"No neighbors."}]

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": summaries, "details": details})

html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:130px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:32px;text-align:center}
  .nca>span{background:#3b82f6}     /* blue */
  .leaf>span{background:#22c55e}    /* green */
  .avg>span{background:#f59e0b}     /* amber */
  #align td:nth-child(2),#align th:nth-child(2),#align td:nth-child(3),#align th:nth-child(3),#align td:nth-child(4),#align th:nth-child(4){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>NCA (FT=1%)</th><th>Leaf (FT=1%)</th><th>Average</th><th>kN|kL</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCell(valRaw, cls){
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const text=isNaN(valRaw)?'':Math.round(w);
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function childHTML(t){
      const rows=(data.details||{})[t]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No neighbors.</div>';
      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="6">'+r.__group__+'</td></tr>';
        return `<tr>
          <td>${r["#"]||""}</td><td>${r.FT==null?"":(r.FT===1?"FT=1":"FT=0")}</td>
          <td>${(r.Weight==null||isNaN(r.Weight))?"":Number(r.Weight).toFixed(4)}</td>
          <td>${(r.Dist==null||isNaN(r.Dist))?"":Number(r.Dist).toFixed(4)}</td>
          <td>${r.Ticker||""}</td>
          <td>${(r["MC_PM_Max_M"]??"")}</td>
          <td>${(r["Float_PM_Max_M"]??"")}</td>
          <td>${(r["Catalyst"]??"")}</td>
          <td>${(r["ATR_$"]??"")}</td>
          <td>${(r["Gap_%"]??"")}</td>
          <td>${(r["Max_Pull_PM_%"]??"")}</td>
          <td>${(r["PM_Vol_M"]??"")}</td>
          <td>${(r["PM$Vol/MC_%"]??"")}</td>
          <td>${(r["RVOL_Max_PM_cum"]??"")}</td>
          <td>${(r["FR_x"]??"")}</td>
          <td>${(r["PM_Vol_%"]??"")}</td>
        </tr>`;
      }).join('');
      return `<table class="child-table">
        <thead><tr>
          <th>#</th><th>FT</th><th>Weight</th><th>Dist</th><th>Ticker</th>
          <th>MC_PM_Max_M</th><th>Float_PM_Max_M</th><th>Catalyst</th><th>ATR_$</th><th>Gap_%</th>
          <th>Max_Pull_PM_%</th><th>PM_Vol_M</th><th>PM$Vol/MC_%</th><th>RVOL_Max_PM_cum</th><th>FR_x</th><th>PM_Vol_%</th>
        </tr></thead>
        <tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:'NCA',  render:(v)=>barCell(v,'nca')},
          {data:'Leaf', render:(v)=>barCell(v,'leaf')},
          {data:'Avg',  render:(v)=>barCell(v,'avg')},
          {data:null,   render:(r)=>`${r.kN||0}|${r.kL||0}`}
        ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { row.child(childHTML(row.data().Ticker)).show(); $(this).addClass('shown'); }
      });
    });
  </script>
</body></html>
"""
html = html.replace("%%PAYLOAD%%", SAFE_JSON_DUMPS(payload))
components.html(html, height=620, scrolling=True)

# ============================== Delete Control ==============================
tickers = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
unique_tickers, _seen = [], set()
for t in tickers:
    if t and t not in _seen:
        unique_tickers.append(t); _seen.add(t)

del_cols = st.columns([4, 1])
with del_cols[0]:
    to_delete = st.multiselect(
        "",
        options=unique_tickers,
        default=[],
        key="del_selection",
        placeholder="Select tickers…",
        label_visibility="collapsed",
    )
with del_cols[1]:
    if st.button("Delete", use_container_width=True, key="delete_btn"):
        if to_delete:
            ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(to_delete)]
            st.success(f"Deleted: {', '.join(to_delete)}")
            do_rerun()
        else:
            st.info("No tickers selected.")
