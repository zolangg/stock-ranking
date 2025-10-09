# app.py — Premarket Ranking with Pred Model + NCA kernel-kNN + CatBoost Leaf-Embedding kNN (with Average bar)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])         # user-added rows
ss.setdefault("last", {})         # last added
ss.setdefault("lassoA", {})       # daily volume predictor (multiplier model)
ss.setdefault("base_df", pd.DataFrame())

ss.setdefault("nca_model", {})    # scaler, nca, X_emb, y, feat_names, df_meta, guard
ss.setdefault("leaf_model", {})   # scaler, cat_model, emb (train), y, feat_names, df_meta, head_name

# ============================== Core deps ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.ensemble import IsolationForest
except ModuleNotFoundError:
    st.error("Missing scikit-learn. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

# CatBoost ONLY (no RF)
try:
    from catboost import CatBoostClassifier
except Exception:
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
    try:
        return 1 if float(sv) >= 0.5 else 0
    except:
        return np.nan

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

# ============================== Variables ==============================
# 12 features (includes PM_$Vol_M$, FR_x, PM_Vol_%)
FEAT12 = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM_$Vol_M$","PM$Vol/MC_%",
    "RVOL_Max_PM_cum","FR_x","PM_Vol_%"
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

    nca = NeighborhoodComponentsAnalysis(
        n_components=min(Xs.shape[1], 8), random_state=42, max_iter=300
    )
    Xn = nca.fit_transform(Xs, y)

    guard = IsolationForest(n_estimators=200, random_state=42, contamination="auto").fit(Xn)

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

    model = CatBoostClassifier(
        depth=6, learning_rate=0.08, loss_function="Logloss",
        random_state=42, iterations=600, verbose=False, l2_leaf_reg=6,
        class_weights=[1.0, float((y==0).sum() / max(1,(y==1).sum()))]
    )
    model.fit(Xs, y)
    leaf_mat = np.array(model.calc_leaf_indexes(Xs), dtype=int)  # (n, n_trees)
    metric = "cosine"

    meta_cols = ["FT01"]
    for c in ("TickerDB","Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols: meta_cols.append(f)
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True)

    return {"scaler":scaler, "leaf_clf":model, "emb":leaf_mat, "metric":metric, "y":y,
            "feat_names":feat_names, "df_meta":meta, "head_name":"CatBoost-leaf-kNN"}

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
    bw = np.median(d[d>0]) if np.any(d>0) else (np.mean(d)+1e-6)
    bw = max(bw, 1e-6)
    w = np.exp(-(d/bw)**2)
    return d, w

def _nca_score(model, row_dict, feat_names):
    vec = []
    for f in feat_names:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)
    xn = model["nca"].transform(xs)

    Xn = model["X_emb"]; y = model["y"]
    d = pairwise_distances(xn, Xn, metric="euclidean").ravel()
    order = np.argsort(d)
    d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(25, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0, "k":0}

    y_top = y[idx]
    w_norm = w_top / (w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y_top==1).astype(float), w_norm)) * 100.0
    return {"p1": p1, "k": int(k_star)}

def _leaf_score(model, row_dict, feat_names):
    vec = []
    for f in feat_names:
        v = row_dict.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        vec.append(float(v))
    x = np.array(vec)[None,:]
    xs = model["scaler"].transform(x)

    leaf_row = np.array(model["leaf_clf"].calc_leaf_indexes(xs), dtype=int)  # (1, n_trees)
    Emb = model["emb"]; metric = model["metric"]

    d = pairwise_distances(leaf_row, Emb, metric=metric).ravel()
    order = np.argsort(d)
    d_sorted = d[order]
    k_star = _elbow_kstar(d_sorted, k_min=3, k_max=min(25, len(order)))
    d_top, w_top = _kernel_weights(d_sorted, k_star)
    idx = order[:k_star]
    if w_top.size == 0: return {"p1":0.0, "k":0}

    y = model["y"]
    w_norm = w_top / (w_top.sum() if w_top.sum()>0 else 1.0)
    p1 = float(np.dot((y[idx]==1).astype(float), w_norm)) * 100.0
    return {"p1": p1, "k": int(k_star)}

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
            def _to_binary_local(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1.0
                if sv in {"0","false","no","n","f"}: return 0.0
                try:
                    fv = float(sv); return 1.0 if fv >= 0.5 else 0.0
                except: return np.nan
            if cand_catalyst: df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

            # derived ratios
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # scale % fields stored as fractions
            if "Gap_%" in df.columns:            df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "PM_Vol_%" in df.columns:         df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns:    df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT labels
            def _to_binary(v):
                sv = str(v).strip().lower()
                if sv in {"1","true","yes","y","t"}: return 1
                if sv in {"0","false","no","n","f"}: return 0
                try:
                    return 1 if float(sv) >= 0.5 else 0
                except: return np.nan
            df["FT01"] = df["GroupRaw"].map(_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

            # Max Push Daily (%) fraction -> %
            pmh_col = _pick(raw, ["Max Push Daily (%)", "Max Push Daily %", "Max_Push_Daily_%"])
            if pmh_col is not None:
                pmh_raw = pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")
                df["Max_Push_Daily_%"] = pmh_raw * 100.0
            else:
                df["Max_Push_Daily_%"] = np.nan

            ss.base_df = df

            # train daily volume predictor
            ss.lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            # Build heads on FEAT12
            have = [f for f in FEAT12 if f in df.columns]
            if len(have) < len(FEAT12):
                st.warning(f"Missing features for heads: {sorted(set(FEAT12)-set(have))}")
            nca = _build_nca(df, have, y_col="FT01")
            leaf = _build_leaf_head(df, have, y_col="FT01")
            if not nca: st.warning("NCA head could not be built (need ≥30 complete rows).")
            if not leaf: st.warning("CatBoost-leaf head could not be built (need ≥30 complete rows).")
            ss.nca_model = nca or {}
            ss.leaf_model = leaf or {}

            st.success(f"Loaded “{sel_sheet}”. Models ready.")
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
    # predict daily volume
    pred = predict_daily_calibrated(row, ss.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    # PM_Vol_% based on prediction
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Similarity Alignment (bars + summary child rows) ==============================
st.markdown("### Alignment")

# guards
base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty:
    st.info("Upload DB and click **Build models** first.")
    st.stop()
if not ss.nca_model or not ss.leaf_model:
    st.info("Models not ready. Build models first.")
    st.stop()

# build summaries
summary_rows, detail_map = [], {}
feat_list = [f for f in FEAT12 if f]  # exactly 12 features

for row in ss.rows:
    stock = dict(row)
    tkr = stock.get("Ticker") or "—"

    # scores
    nca_res = _nca_score(ss.nca_model, stock, ss.nca_model.get("feat_names", feat_list)) or {"p1":0.0,"k":0}
    leaf_res = _leaf_score(ss.leaf_model, stock, ss.leaf_model.get("feat_names", feat_list)) or {"p1":0.0,"k":0}

    nca_p  = float(np.clip(nca_res["p1"],  0, 100))
    leaf_p = float(np.clip(leaf_res["p1"], 0, 100))
    avg_p  = (nca_p + leaf_p) / 2.0

    summary_rows.append({
        "Ticker": tkr,
        "NCA_val_raw":   nca_p,
        "Leaf_val_raw":  leaf_p,
        "Avg_val_raw":   avg_p,
        "NCA_val_int":   int(round(nca_p)),
        "Leaf_val_int":  int(round(leaf_p)),
        "Avg_val_int":   int(round(avg_p)),
    })

    # summary child rows (12 features + PredVol_M)
    drows = [{"__group__": "Inputs (12) + Predicted Daily Volume"}]
    for v in feat_list + ["PredVol_M"]:
        val = stock.get(v, None)
        drows.append({"Variable": v, "Value": None if val is None or (isinstance(val,float) and not np.isfinite(val)) else float(val)})
    detail_map[tkr] = drows

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": summary_rows, "details": detail_map})

html = """
<!DOCTYPE html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
<style>
  body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,"Helvetica Neue",sans-serif}
  table.dataTable tbody tr{cursor:pointer}
  .bar-wrap{display:flex;justify-content:center;align-items:center;gap:6px}
  .bar{height:12px;width:120px;border-radius:8px;background:#eee;position:relative;overflow:hidden}
  .bar>span{position:absolute;left:0;top:0;bottom:0;width:0%}
  .bar-label{font-size:11px;white-space:nowrap;color:#374151;min-width:28px;text-align:center}
  .blue>span{background:#3b82f6}
  .purple>span{background:#8b5cf6}
  .slate>span{background:#64748b}
  #align td:nth-child(2),#align th:nth-child(2),
  #align td:nth-child(3),#align th:nth-child(3),
  #align td:nth-child(4),#align th:nth-child(4){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .col-var{width:30%}.col-val{width:20%}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th>NCA (FT=1)</th><th>Leaf (FT=1)</th><th>Average (FT=1)</th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCell(valRaw, cls, valInt){
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function formatVal(x){return (x==null||isNaN(x))?'':Number(x).toFixed(3);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No inputs.</div>';

      const cells = rows.map(r=>{
        if(r.__group__) return '<tr class="group-row"><td colspan="2">'+r.__group__+'</td></tr>';
        return `<tr><td class="col-var">${r.Variable}</td><td class="col-val">${formatVal(r.Value)}</td></tr>`;
      }).join('');

      return `<table class="child-table">
        <colgroup><col class="col-var"/><col class="col-val"/></colgroup>
        <thead><tr><th class="col-var">Variable</th><th class="col-val">Value</th></tr></thead>
        <tbody>${cells}</tbody></table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCell(row.NCA_val_raw,'blue',row.NCA_val_int)},
          {data:null, render:(row)=>barCell(row.Leaf_val_raw,'purple',row.Leaf_val_int)},
          {data:null, render:(row)=>barCell(row.Avg_val_raw,'slate',row.Avg_val_int)}
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

# ============================== Delete Control (below table; no title) ==============================
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
