# app.py — Premarket Ranking with OOF Pred Model + NCA kernel-kNN + CatBoost Leaf-Embedding kNN
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, math

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking (NCA + CatBoost Leaf)", layout="wide")
st.title("Premarket Stock Ranking — NCA + Leaf-kNN Similarity")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("pred_full", {})         # full daily volume predictor (for queries)
ss.setdefault("nca_model", {})         # scaler, nca, X_emb, y, feat_names, df_meta, guard, sel_feats
ss.setdefault("leaf_model", {})        # scaler, cb_model, emb, metric, y, feat_names, df_meta, sel_feats

# ============================== Core deps ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.ensemble import IsolationForest
except ModuleNotFoundError:
    st.error("Missing scikit-learn. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

try:
    from catboost import CatBoostClassifier, Pool
except ModuleNotFoundError:
    st.error("CatBoost not installed. Add `catboost>=1.2.5` to requirements.txt and redeploy.")
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
    try: return 1 if float(sv) >= 0.5 else 0
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
            level_y = np.delete(level_y, i+1); level_n = np.delete(level_n, i+1); xs = np.delete(xs, i+1)
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
BASE9 = ["MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
         "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum"]
EXTRA3 = ["FR_x","PM_Vol_%","PM_$Vol_M$"]
FEAT12 = BASE9 + EXTRA3

# ============================== Daily Volume Predictor (LASSO -> OLS -> isotonic) ==============================
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

def _fit_daily_predictor(df: pd.DataFrame, idx_rows=None):
    # returns predictor model dict or {}
    eps = 1e-6
    cols_need = ["ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"]
    has_need = all(c in df.columns for c in cols_need)
    mcap_series  = df["MC_PM_Max_M"] if "MC_PM_Max_M" in df.columns else df.get("MarketCap_M$")
    float_series = df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df.get("Float_M")
    if not has_need or mcap_series is None or float_series is None: return {}

    D = df if idx_rows is None else df.iloc[idx_rows]
    PM  = pd.to_numeric(D["PM_Vol_M"],    errors="coerce").values
    DV  = pd.to_numeric(D["Daily_Vol_M"], errors="coerce").values
    valid = np.isfinite(PM) & np.isfinite(DV) & (PM>0) & (DV>0)
    if valid.sum() < 50: return {}

    ln_mcap   = np.log(np.clip(pd.to_numeric((mcap_series if idx_rows is None else mcap_series.iloc[idx_rows]), errors="coerce").values,  eps, None))
    ln_gapf   = np.log(np.clip(pd.to_numeric(D["Gap_%"], errors="coerce").values,  0, None)/100.0 + eps)
    ln_atr    = np.log(np.clip(pd.to_numeric(D["ATR_$"], errors="coerce").values,  eps, None))
    ln_pm     = np.log(np.clip(pd.to_numeric(D["PM_Vol_M"], errors="coerce").values, eps, None))
    ln_pm_dol = np.log(np.clip(pd.to_numeric(D["PM_$Vol_M$"], errors="coerce").values, eps, None))
    ln_fr     = np.log(np.clip(pd.to_numeric(D["FR_x"], errors="coerce").values,   eps, None))
    ln_float_pmmax = np.log(np.clip(pd.to_numeric((df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df["Float_M"])[idx_rows] if idx_rows is not None else (df["Float_PM_Max_M"] if "Float_PM_Max_M" in df.columns else df["Float_M"]), errors="coerce").values, eps, None))
    maxpullpm = pd.to_numeric(D.get("Max_Pull_PM_%", np.nan), errors="coerce").values
    ln_rvol   = np.log(np.clip(pd.to_numeric(D.get("RVOL_Max_PM_cum", np.nan), errors="coerce").values, eps, None))
    pmmc      = pd.to_numeric(D.get("PM$Vol/MC_%", np.nan), errors="coerce").values
    catalyst  = pd.to_numeric(D.get("Catalyst", np.nan), errors="coerce").fillna(0.0).clip(0,1).values

    mult = np.maximum(DV / PM, 1.0)
    y_ln = np.log(mult)

    feats = [
        ("ln_mcap_pmmax",  ln_mcap), ("ln_gapf", ln_gapf), ("ln_atr", ln_atr),
        ("ln_pm", ln_pm), ("ln_pm_dol", ln_pm_dol), ("ln_fr", ln_fr),
        ("catalyst", catalyst), ("ln_float_pmmax", ln_float_pmmax),
        ("maxpullpm", maxpullpm), ("ln_rvolmaxpm", ln_rvol), ("pm_dol_over_mc", pmmc),
    ]
    X = np.hstack([a.reshape(-1,1) for _,a in feats])
    mask = valid & np.isfinite(y_ln) & np.isfinite(X).all(axis=1)
    if mask.sum() < 50: return {}
    X = X[mask]; y_ln = y_ln[mask]; PMm = PM[mask]; DVv = DV[mask]

    # Winsorize heavy-tailed selected columns on train
    name_to_idx = {nm:i for i,(nm,_) in enumerate(feats)}
    def _winsor(col):
        arr = X[:, name_to_idx[col]]
        lo, hi = _compute_bounds(arr[np.isfinite(arr)])
        X[:, name_to_idx[col]] = _apply_bounds(arr, lo, hi)
        return (col, lo, hi)
    winsor_bounds = {}
    for nm in ("maxpullpm", "pm_dol_over_mc"):
        if nm in name_to_idx:
            col, lo, hi = _winsor(nm); winsor_bounds[col] = (float(lo) if np.isfinite(lo) else np.nan, float(hi) if np.isfinite(hi) else np.nan)

    # also winsor target multiplier
    mult_w = np.exp(y_ln)
    m_lo, m_hi = _compute_bounds(mult_w)
    y_ln = np.log(_apply_bounds(mult_w, m_lo, m_hi))

    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd==0] = 1.0
    Xs = (X - mu) / sd

    # LASSO CD with simple CV for lambda
    folds = _kfold_indices(len(y_ln), k=min(5, max(2, len(y_ln)//10)), seed=42)
    lam_grid = np.geomspace(0.001, 1.0, 26)
    cv_mse = []
    for lam in lam_grid:
        errs=[]
        for vi in range(len(folds)):
            te = folds[vi]; tr = np.hstack([folds[j] for j in range(len(folds)) if j!=vi])
            w = _lasso_cd_std(Xs[tr], y_ln[tr], lam=lam, max_iter=1400)
            yhat = Xs[te] @ w
            errs.append(np.mean((yhat - y_ln[te])**2))
        cv_mse.append(np.mean(errs))
    lam_best = float(lam_grid[int(np.argmin(cv_mse))])
    w_l1 = _lasso_cd_std(Xs, y_ln, lam=lam_best, max_iter=2000)
    sel = np.flatnonzero(np.abs(w_l1) > 1e-8)
    if sel.size == 0: return {}

    X_sel = X[:, sel]
    X_design = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
    coef_ols, *_ = np.linalg.lstsq(X_design, y_ln, rcond=None)
    b0 = float(coef_ols[0]); bet = coef_ols[1:].astype(float)

    # isotonic on holdout chunk if enough
    iso_bx = np.array([], dtype=float); iso_by = np.array([], dtype=float)
    split = max(10, int(X_sel.shape[0]*0.8))
    if X_sel.shape[0] - split >= 8:
        yhat_va_ln = (np.column_stack([np.ones(X_sel.shape[0]-split), X_sel[split:]]) @ coef_ols).astype(float)
        mult_pred = np.exp(yhat_va_ln)
        mult_true = np.maximum(DVv, 1e-6) / np.maximum(PMm, 1e-6)
        mult_va_true = mult_true[split:]
        finite = np.isfinite(mult_pred) & np.isfinite(mult_va_true)
        if finite.sum() >= 8 and np.unique(mult_pred[finite]).size >= 3:
            iso_bx, iso_by = _pav_isotonic(mult_pred[finite], mult_va_true[finite])

    return {
        "eps": eps, "terms": [feats[i][0] for i in sel], "b0": b0, "betas": bet, "sel_idx": sel.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(), "winsor_bounds": winsor_bounds,
        "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "feat_order": [nm for nm,_ in feats],
    }

def _predict_daily_for_rows(df_rows: pd.DataFrame, model: dict) -> np.ndarray:
    # returns PredVol_M vector (nan on failure)
    if not model or "betas" not in model: return np.full(len(df_rows), np.nan)
    eps = float(model.get("eps", 1e-6))
    feat_order = model["feat_order"]; winsor_bounds = model.get("winsor_bounds", {})
    sel = model.get("sel_idx", []); b0 = float(model["b0"]); bet = np.array(model["betas"], dtype=float)

    def safe_log(v):
        v = float(v) if v is not None else np.nan
        return np.log(np.clip(v, eps, None)) if np.isfinite(v) else np.nan

    out = []
    for _, row in df_rows.iterrows():
        ln_mcap  = safe_log(row.get("MC_PM_Max_M") if pd.notna(row.get("MC_PM_Max_M", np.nan)) else row.get("MarketCap_M$"))
        ln_gapf  = np.log(np.clip((row.get("Gap_%") or 0.0)/100.0 + eps, eps, None)) if row.get("Gap_%") is not None else np.nan
        ln_atr   = safe_log(row.get("ATR_$"))
        ln_pm    = safe_log(row.get("PM_Vol_M"))
        ln_pm_d  = safe_log(row.get("PM_$Vol_M$"))
        ln_fr    = safe_log(row.get("FR_x"))
        catalyst = 1.0 if float(row.get("Catalyst",0))>=0.5 else 0.0
        ln_float = safe_log(row.get("Float_PM_Max_M") if pd.notna(row.get("Float_PM_Max_M", np.nan)) else row.get("Float_M"))
        maxpull  = float(row.get("Max_Pull_PM_%")) if row.get("Max_Pull_PM_%") is not None else np.nan
        ln_rvol  = safe_log(row.get("RVOL_Max_PM_cum"))
        pmmc     = float(row.get("PM$Vol/MC_%")) if row.get("PM$Vol/MC_%") is not None else np.nan

        fmap = {"ln_mcap_pmmax":ln_mcap,"ln_gapf":ln_gapf,"ln_atr":ln_atr,"ln_pm":ln_pm,"ln_pm_dol":ln_pm_d,
                "ln_fr":ln_fr,"catalyst":catalyst,"ln_float_pmmax":ln_float,"maxpullpm":maxpull,
                "ln_rvolmaxpm":ln_rvol,"pm_dol_over_mc":pmmc}
        vec=[]
        ok=True
        for nm in feat_order:
            v = fmap.get(nm, np.nan)
            if not np.isfinite(v): ok=False; break
            lo, hi = winsor_bounds.get(nm, (np.nan, np.nan))
            if np.isfinite(lo) or np.isfinite(hi):
                v = float(np.clip(v, lo if np.isfinite(lo) else v, hi if np.isfinite(hi) else v))
            vec.append(v)
        if (not ok) or (not sel): out.append(np.nan); continue
        vec = np.array(vec, dtype=float)
        yhat_ln = b0 + float(np.dot(vec[sel], bet))
        raw_mult = math.exp(yhat_ln) if np.isfinite(yhat_ln) else np.nan
        if not np.isfinite(raw_mult): out.append(np.nan); continue
        iso_bx = np.array(model.get("iso_bx", []), dtype=float)
        iso_by = np.array(model.get("iso_by", []), dtype=float)
        cal_mult = float(_iso_predict(iso_bx, iso_by, np.array([raw_mult]))[0]) if (iso_bx.size>=2 and iso_by.size>=2) else float(raw_mult)
        cal_mult = max(cal_mult, 1.0)
        PM = float(row.get("PM_Vol_M") or np.nan)
        out.append(float(PM * cal_mult) if np.isfinite(PM) and PM>0 else np.nan)
    return np.array(out, dtype=float)

# ============================== Similarity helpers ==============================
def _elbow_kstar(sorted_vals, k_min=3, k_max=15, max_rank=30):
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
    bw = np.median(d[d>0]) if np.any(d>0) else (np.mean(d)+1e-6)
    bw = max(bw, 1e-6)
    w = np.exp(-(d/bw)**2)
    # tiny weight floor
    w[w < 1e-6] = 0.0
    return d, w

# ============================== Feature selection (drop-one + stability, per head) ==============================
def _score_nca_fold(X_tr, y_tr, X_va, y_va):
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr); Xva = scaler.transform(X_va)
    nca = NeighborhoodComponentsAnalysis(n_components=min(6, Xtr.shape[1]), random_state=42, max_iter=250)
    Xtr_n = nca.fit_transform(Xtr, y_tr); Xva_n = nca.transform(Xva)

    # simple kNN in embedded space with fixed k=7 for CV scoring
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=min(7, len(Xtr_n)))
    knn.fit(Xtr_n, y_tr)
    proba = knn.predict_proba(Xva_n)[:,1]
    try:
        auc = roc_auc_score(y_va, proba)
    except Exception:
        # fallback to accuracy if AUC undefined
        auc = np.mean(((proba>=0.5).astype(int) == y_va).astype(float))
    return float(auc)

def _score_leaf_fold(X_tr, y_tr, X_va, y_va):
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr); Xva = scaler.transform(X_va)
    params = dict(
        depth=6, learning_rate=0.08, loss_function="Logloss",
        iterations=1200, l2_leaf_reg=6, random_seed=42, verbose=False,
        bootstrap_type='No', random_strength=0.0,
        class_weights=[1.0, float((y_tr==0).sum()/max(1,(y_tr==1).sum()))]
    )
    cb = CatBoostClassifier(**params)
    # early stopping
    cb.fit(Xtr, y_tr, eval_set=(Xva, y_va), verbose=False, early_stopping_rounds=50)
    try:
        proba = cb.predict_proba(Xva)[:,1]
        auc = roc_auc_score(y_va, proba)
    except Exception:
        pred = cb.predict(Xva)
        auc = np.mean((pred.flatten().astype(int) == y_va).astype(float))
    return float(auc)

def select_features_drop_one(df, feat_list, head="nca", kfolds=5, stab_thr=0.60, cap=10, y_col="FT01"):
    # returns selected feature names
    D = df.dropna(subset=feat_list + [y_col]).copy()
    if D.shape[0] < 60:  # need some mass for selection to be meaningful
        return feat_list[:min(cap, len(feat_list))]
    X = D[feat_list].astype(float).values
    y = D[y_col].astype(int).values
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

    # baseline CV
    base_scores = []
    for tr, va in skf.split(X, y):
        if head=="nca": s = _score_nca_fold(X[tr], y[tr], X[va], y[va])
        else:           s = _score_leaf_fold(X[tr], y[tr], X[va], y[va])
        base_scores.append(s)
    base = np.mean(base_scores)

    contrib = {f: [] for f in feat_list}
    for j, f in enumerate(feat_list):
        scores=[]
        cols = [i for i in range(len(feat_list)) if i!=j]
        for tr, va in skf.split(X, y):
            if len(cols)==0:
                s = 0.0
            else:
                if head=="nca": s = _score_nca_fold(X[tr][:,cols], y[tr], X[va][:,cols], y[va])
                else:           s = _score_leaf_fold(X[tr][:,cols], y[tr], X[va][:,cols], y[va])
            scores.append(s)
        # positive if dropping hurts (i.e., feature useful)
        drops = np.array(scores) < np.array(base_scores)
        contrib[f] = drops.astype(float).tolist()

    # stability: keep features that helped in >= stab_thr folds
    keep = [f for f in feat_list if np.mean(contrib[f]) >= stab_thr]
    if len(keep) == 0:
        # fallback: keep top by average delta
        deltas = {f: (np.mean(base_scores) - np.mean([_ for _ in contrib[f]])) for f in feat_list}
        keep = [k for k,_ in sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)][:min(cap, len(feat_list))]
    if len(keep) > cap:
        keep = keep[:cap]
    return keep

# ============================== Heads: NCA + CatBoost Leaf-Embedding ==============================
def _build_nca_head(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    need = [y_col] + feat_names
    df_face = df.dropna(subset=need)
    if df_face.shape[0] < 60: return {}
    y = df_face[y_col].astype(int).values
    X = df_face[feat_names].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    nca = NeighborhoodComponentsAnalysis(n_components=min(6, Xs.shape[1]), random_state=42, max_iter=300)
    Xn = nca.fit_transform(Xs, y)
    guard = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    guard.fit(Xn)
    meta_cols = ["FT01"]
    for c in ("TickerDB","Ticker"):
        if c in df_face.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df_face.columns and f not in meta_cols: meta_cols.append(f)
    meta = df_face[meta_cols].reset_index(drop=True)
    return {"scaler":scaler, "nca":nca, "X_emb":Xn, "y":y, "df_meta":meta, "guard":guard, "feat_names":feat_names}

def _build_leaf_head(df: pd.DataFrame, feat_names: list[str], y_col="FT01"):
    need = [y_col] + feat_names
    df_face = df.dropna(subset=need)
    if df_face.shape[0] < 60: return {}
    y = df_face[y_col].astype(int).values
    X = df_face[feat_names].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    params = dict(
        depth=6, learning_rate=0.08, loss_function="Logloss",
        iterations=1200, l2_leaf_reg=6, random_seed=42, verbose=False,
        bootstrap_type='No', random_strength=0.0,
        class_weights=[1.0, float((y==0).sum()/max(1,(y==1).sum()))]
    )
    cb = CatBoostClassifier(**params)
    # use 15% eval split for early stopping
    n = len(y); cut = max(20, int(n*0.85))
    cb.fit(Xs[:cut], y[:cut], eval_set=(Xs[cut:], y[cut:]), verbose=False, early_stopping_rounds=50)
    # leaf embedding
    leaf_mat = cb.calc_leaf_indexes(Xs)
    emb = np.array(leaf_mat, dtype=int)
    meta_cols = ["FT01"]
    for c in ("TickerDB","Ticker"):
        if c in df_face.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df_face.columns and f not in meta_cols: meta_cols.append(f)
    meta = df_face[meta_cols].reset_index(drop=True)
    return {"scaler":scaler, "cb":cb, "emb":emb, "metric":"cosine", "y":y, "df_meta":meta, "feat_names":feat_names}

def _score_nca_query(model, row_dict):
    vec=[]
    for f in model["feat_names"]:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)
    xn = model["nca"].transform(xs)
    # outlier score (not shown, but could be used)
    _ = model["guard"].decision_function(xn.reshape(1,-1))[0]
    Xn = model["X_emb"]; y = model["y"]
    d = pairwise_distances(xn, Xn, metric="euclidean").ravel()
    order = np.argsort(d); d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=15, max_rank=30)
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0,"k":0}
    w_norm = w_top/(w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0
    return {"p1":p1, "k": int(k_star)}

def _score_leaf_query(model, row_dict):
    vec=[]
    for f in model["feat_names"]:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)
    leaf_row = model["cb"].calc_leaf_indexes(xs)
    emb_q = np.array(leaf_row, dtype=int)
    Emb = model["emb"]; metric = model["metric"]; y = model["y"]
    d = pairwise_distances(emb_q, Emb, metric=metric).ravel()
    order = np.argsort(d); d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=15, max_rank=30)
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0,"k":0}
    w_norm = w_top/(w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0
    return {"p1":p1, "k": int(k_star)}

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

            # Catalyst
            cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_safe_to_binary_float)

            # Derived: FR_x and PM$Vol/MC_%
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # Scale % fields if stored as fractions
            if "Gap_%" in df.columns:         df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "PM_Vol_%" in df.columns:      df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns: df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT labels
            df["FT01"] = pd.Series(df["GroupRaw"]).map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()

            # Ticker passthrough (nice-to-have)
            tcol = _pick(raw, ["ticker","symbol","name"])
            if tcol is not None: df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

            # Max Push Daily (%) fraction -> %
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            if pmh_col is not None:
                pmh_raw = pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")
                df["Max_Push_Daily_%"] = pmh_raw * 100.0
            else:
                df["Max_Push_Daily_%"] = np.nan

            # ---------- OOF PredVol_M to compute PM_Vol_% without leakage ----------
            # Build folds on rows where we can train the predictor
            idx_all = np.arange(len(df))
            has_pred_cols = df[["ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Daily_Vol_M","Gap_%"]].notna().all(axis=1)
            has_mcap = df[["MC_PM_Max_M","MarketCap_M$"]].notna().any(axis=1)
            has_float = df[["Float_PM_Max_M","Float_M"]].notna().any(axis=1)
            pred_mask = has_pred_cols & has_mcap & has_float
            oof = np.full(len(df), np.nan)
            y_ft = df["FT01"].values
            usable = np.where(pred_mask)[0]
            if len(usable) >= 60:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for tr_idx, va_idx in skf.split(usable, y_ft[usable]):
                    tr_rows = usable[tr_idx]; va_rows = usable[va_idx]
                    m = _fit_daily_predictor(df, idx_rows=tr_rows)
                    if not m: continue
                    preds = _predict_daily_for_rows(df.iloc[va_rows], m)
                    oof[va_rows] = preds
            # Set PM_Vol_% from OOF preds (train rows)
            df["PredVol_M_OOF"] = oof
            df["PM_Vol_%"] = np.where(np.isfinite(oof) & (oof>0),
                                      df["PM_Vol_M"] / oof * 100.0,
                                      df.get("PM_Vol_%", np.nan))

            # Fit full predictor for queries (on all eligible rows)
            ss.pred_full = _fit_daily_predictor(df)

            # ---------- Head-specific feature selection ----------
            train_for_heads = df.copy()
            nca_feats  = select_features_drop_one(train_for_heads, FEAT12, head="nca",  kfolds=5, stab_thr=0.60, cap=10, y_col="FT01")
            leaf_feats = select_features_drop_one(train_for_heads, FEAT12, head="leaf", kfolds=5, stab_thr=0.60, cap=10, y_col="FT01")

            # ---------- Build heads ----------
            ss.nca_model  = _build_nca_head(df,  nca_feats,  y_col="FT01") or {}
            ss.leaf_model = _build_leaf_head(df, leaf_feats, y_col="FT01") or {}
            if ss.nca_model:  ss.nca_model["sel_feats"]  = nca_feats
            if ss.leaf_model: ss.leaf_model["sel_feats"] = leaf_feats

            ss.base_df = df
            st.success(f"Loaded “{sel_sheet}”. Models ready. NCA uses {len(nca_feats)} feats; Leaf uses {len(leaf_feats)} feats.")
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
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
        "CatalystYN": catalyst_yn,
    }
    # Predict daily volume with full predictor, compute PM_Vol_% for this query
    pred = np.nan
    if ss.get("pred_full", {}):
        pred = _predict_daily_for_rows(pd.DataFrame([row]), ss.pred_full)[0]
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Similarity Table ==============================
st.markdown("### Similarity — FT=1% by head (NCA, Leaf, Average)")

if not (ss.get("nca_model") and ss.get("leaf_model")):
    st.info("Upload DB and click **Build models** first (need OOF % and both heads).")
    st.stop()

nca_model  = ss.nca_model
leaf_model = ss.leaf_model

# Build display rows
summaries, details = [], {}
for row in ss.rows:
    tkr = row.get("Ticker","—")
    nca = _score_nca_query(nca_model, row)
    leaf = _score_leaf_query(leaf_model, row)
    if not nca or not leaf:
        summaries.append({"Ticker": tkr, "NCA":0.0,"Leaf":0.0,"Avg":0.0, "kN":0,"kL":0})
        details[tkr] = [{"__group__":"Missing inputs for all 12 variables (or head selection)"}]
        continue
    nca_p = float(nca["p1"]); leaf_p = float(leaf["p1"]); avg_p = (nca_p + leaf_p)/2.0
    summaries.append({"Ticker": tkr, "NCA": nca_p, "Leaf": leaf_p, "Avg": avg_p, "kN": int(nca["k"]), "kL": int(leaf["k"])})
    # Compact variable summary (12 vars + PredVol_M)
    rows = [{"__group__": "Inputs summary"}]
    var_order = ["MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%","Max_Pull_PM_%","PM_Vol_M",
                 "PM_$Vol_M$","PM$Vol/MC_%","RVOL_Max_PM_cum","FR_x","PM_Vol_%","PredVol_M"]
    for v in var_order:
        val = row.get(v, None)
        rows.append({"Variable": v, "Value": (None if val is None or (isinstance(val, float) and not np.isfinite(val)) else float(val))})
    details[tkr] = rows

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": [
    {"Ticker": s["Ticker"], "NCA": s["NCA"], "Leaf": s["Leaf"], "Avg": s["Avg"], "kN": s["kN"], "kL": s["kL"]}
    for s in summaries
], "details": details})

html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:140px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:36px;text-align:center}
  .nca>span{background:#3b82f6}.leaf>span{background:#10b981}.avg>span{background:#f59e0b}
  #align td:nth-child(2),#align td:nth-child(3),#align td:nth-child(4){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .col-var{width:28%}.col-val{width:20%}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>NCA (FT=1)</th><th>Leaf (FT=1)</th><th>Average (FT=1)</th><th>kN</th><th>kL</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;
    function barCell(valRaw, cls, valInt){
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}%</div></div>`;
    }
    function formatVal(x){return (x==null||isNaN(x))?'':Number(x).toFixed(4);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No details.</div>';
      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="2">'+r.__group__+'</td></tr>';
        const v = (r.Value==null||isNaN(r.Value))? '' : formatVal(r.Value);
        return `<tr><td class="col-var">${r.Variable}</td><td class="col-val">${v}</td></tr>`;
      }).join('');
      return `<table class="child-table">
        <colgroup><col class="col-var"/><col class="col-val"/></colgroup>
        <thead><tr><th class="col-var">Variable</th><th class="col-val">Value</th></tr></thead>
        <tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[3,'desc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCell(row.NCA,'nca',Math.round(row.NCA))},
          {data:null, render:(row)=>barCell(row.Leaf,'leaf',Math.round(row.Leaf))},
          {data:null, render:(row)=>barCell(row.Avg,'avg',Math.round(row.Avg))},
          {data:'kN'},
          {data:'kL'}
        ]
      });
      $('#align tbody').on('click','tr',function(){
        const row=table.row(this);
        if(row.child.isShown()){ row.child.hide(); $(this).removeClass('shown'); }
        else { const t=row.data().Ticker; row.child(childTableHTML(t)).show(); $(this).addClass('shown'); }
      });
    });
  </script>
</body></html>
"""
html = html.replace("%%PAYLOAD%%", SAFE_JSON_DUMPS(payload))
components.html(html, height=620, scrolling=True)

# ============================== Delete Control (below table) ==============================
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
