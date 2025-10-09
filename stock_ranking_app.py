# app.py — Premarket Stock Ranking (Pred model + RF leaf-similarity; bars + child rows)
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Session ==============================
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("lassoA", {})           # daily volume prediction model (multiplier)
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("rf_model", {})         # scaler, rf, leaf_train, X_scaled, y, feat_names, df_meta

# ============================== Dependencies ==============================
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    st.error("scikit-learn not installed. Add `scikit-learn==1.5.2` to requirements.txt and redeploy.")
    st.stop()

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

# ============================== Feature sets ==============================
# Base 9 + derived (FR_x, PM_Vol_%) so RF uses the nuanced features too
FACE_VARS = [
    "MC_PM_Max_M","Float_PM_Max_M","Catalyst","ATR_$","Gap_%",
    "Max_Pull_PM_%","PM_Vol_M","PM$Vol/MC_%","RVOL_Max_PM_cum",
    "FR_x","PM_Vol_%"
]

# ============================== LASSO (prediction model) ==============================
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

# ============================== RF model ==============================
def _row_to_face(row: dict, feat_names: list[str]) -> np.ndarray | None:
    vals = []
    for f in feat_names:
        v = row.get(f, None)
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return None
        try: vals.append(float(v))
        except: return None
    return np.array(vals, dtype=float)

def _build_rf(df: pd.DataFrame, feat_names: list[str]):
    need = ["FT01"] + feat_names
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"DB missing required columns for RF: {miss}")
        return {}

    df_face = df[need].dropna()
    if df_face.shape[0] < 30:
        st.error(f"Need ≥30 rows without NaNs in {need}. Have {df_face.shape[0]}.")
        return {}

    y = df_face["FT01"].astype(int).values
    X = df_face[feat_names].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1,
        class_weight="balanced_subsample", max_features="sqrt"
    )
    rf.fit(Xs, y)
    leaf_train = rf.apply(Xs)

    # meta for neighbors (ticker + FT + the RF features for display)
    meta_cols = ["FT01"]
    for c in ("TickerDB", "Ticker"):
        if c in df.columns: meta_cols.append(c)
    for f in feat_names:
        if f in df.columns and f not in meta_cols:
            meta_cols.append(f)
    meta = df.loc[df_face.index, meta_cols].reset_index(drop=True)

    return {"scaler":scaler,"rf":rf,"leaf_train":leaf_train,"X_scaled":Xs,"y":y,"feat_names":feat_names,"df_meta":meta}

def _rf_proximity(model, x_vec):
    xs = model["scaler"].transform(x_vec.reshape(1,-1)).ravel()
    leaf_q = model["rf"].apply(xs.reshape(1,-1)).ravel()
    prox = (model["leaf_train"] == leaf_q).mean(axis=1)  # (n_train,)
    order = np.argsort(-prox)
    return prox[order], order

def _elbow_kstar(prox_sorted, max_rank=30, k_min=3, k_max=25):
    if prox_sorted.size <= k_min: return max(1, prox_sorted.size)
    upto = min(max_rank, prox_sorted.size - 1)
    if upto < 2: return min(k_max, max(k_min, prox_sorted.size))
    gaps = prox_sorted[:upto] - prox_sorted[1:upto+1]
    k_star = int(np.argmax(gaps) + 1)
    return max(k_min, min(k_max, k_star))

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

            # map fields (base + for prediction)
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

            # derived basic ratios (FR_x, PM$Vol/MC_%)
            float_basis = "Float_PM_Max_M" if "Float_PM_Max_M" in df.columns and df["Float_PM_Max_M"].notna().any() else "Float_M"
            if {"PM_Vol_M", float_basis}.issubset(df.columns):
                df["FR_x"] = (df["PM_Vol_M"] / df[float_basis]).replace([np.inf,-np.inf], np.nan)
            mcap_basis = "MC_PM_Max_M" if "MC_PM_Max_M" in df.columns and df["MC_PM_Max_M"].notna().any() else "MarketCap_M$"
            if {"PM_$Vol_M$", mcap_basis}.issubset(df.columns):
                df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df[mcap_basis] * 100.0).replace([np.inf,-np.inf], np.nan)

            # scale % fields (DB stores fractions)
            if "Gap_%" in df.columns:         df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
            if "PM_Vol_%" in df.columns:      df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0
            if "Max_Pull_PM_%" in df.columns: df["Max_Pull_PM_%"] = pd.to_numeric(df["Max_Pull_PM_%"], errors="coerce") * 100.0

            # FT groups
            df["FT01"] = df["GroupRaw"].map(_safe_to_binary)
            df = df[df["FT01"].isin([0,1])].copy()
            df["GroupFT"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

            # optional passthrough ticker
            tcol = _pick(raw, ["ticker","symbol","name"])
            if tcol is not None:
                df["TickerDB"] = raw[tcol].astype(str).str.upper().str.strip()

            # ===== Train prediction model on DB and compute PredVol_M & PM_Vol_% if missing =====
            lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}
            ss.lassoA = lassoA

            def _predict_predvol_for_df(df_in: pd.DataFrame, model: dict) -> pd.Series:
                out = []
                for i, r in df_in.iterrows():
                    row = r.to_dict()
                    # ensure FR_x and PM$Vol/MC_% exist for prediction features
                    if "FR_x" not in row or not np.isfinite(row["FR_x"]):
                        fbase = "Float_PM_Max_M" if ("Float_PM_Max_M" in df_in.columns and pd.notna(r.get("Float_PM_Max_M"))) else "Float_M"
                        row["FR_x"] = (r.get("PM_Vol_M", np.nan) / r.get(fbase, np.nan)) if (pd.notna(r.get("PM_Vol_M")) and pd.notna(r.get(fbase))) else np.nan
                    if "PM$Vol/MC_%" not in row or not np.isfinite(row["PM$Vol/MC_%"]):
                        mbase = "MC_PM_Max_M" if ("MC_PM_Max_M" in df_in.columns and pd.notna(r.get("MC_PM_Max_M"))) else "MarketCap_M$"
                        row["PM$Vol/MC_%"] = (r.get("PM_$Vol_M$", np.nan) / r.get(mbase, np.nan) * 100.0) if (pd.notna(r.get("PM_$Vol_M$")) and pd.notna(r.get(mbase)) and r.get(mbase, 0)>0) else np.nan
                    pv = predict_daily_calibrated(row, model)
                    out.append(pv)
                return pd.to_numeric(pd.Series(out, index=df_in.index), errors="coerce")

            df["PredVol_M"] = _predict_predvol_for_df(df, lassoA)
            if "PM_Vol_%" not in df.columns or df["PM_Vol_%"].isna().all():
                denom = df["PredVol_M"]
                df["PM_Vol_%"] = (df["PM_Vol_M"] / denom * 100.0).where(denom > 0)

            # save base
            ss.base_df = df

            # ===== Build RF similarity model on FACE_VARS =====
            available_feats = [f for f in FACE_VARS if f in df.columns]
            missing = [f for f in FACE_VARS if f not in df.columns]
            if missing:
                st.warning(f"Some RF features missing and will be dropped: {missing}")
            ss.rf_model = _build_rf(df, available_feats)
            if not ss.rf_model: st.stop()

            st.success(f"Loaded “{sel_sheet}”. Pred model + RF ready with {ss.rf_model['X_scaled'].shape[0]} rows.")
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
        rvol_pm_cum = st.number_input("Premarket Max RVOL (cum)", 0.0, step=0.01, format="%.2f")
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
    # Predict daily volume and compute PM_Vol_% for the query row
    pred = predict_daily_calibrated(row, ss.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan
    denom = row["PredVol_M"]
    row["PM_Vol_%"] = (row["PM_Vol_M"] / denom) * 100.0 if np.isfinite(denom) and denom > 0 else np.nan

    ss.rows.append(row); ss.last = row
    st.success(f"Saved {ticker}."); do_rerun()

# ============================== Alignment (RF similarity) ==============================
st.markdown("### Alignment")

base_df = ss.get("base_df", pd.DataFrame()).copy()
if base_df.empty or not ss.rf_model:
    if ss.rows:
        st.info("Upload DB and click **Build models** first.")
    else:
        st.info("Upload DB (and/or add at least one stock) to compute alignment.")
    st.stop()

gA, gB = "FT=1", "FT=0"
summary_rows, detail_map = [], {}
feat_names = ss.rf_model.get("feat_names", FACE_VARS)

for row in ss.rows:
    stock = dict(row); tkr = stock.get("Ticker") or "—"
    x = _row_to_face(stock, feat_names)
    if x is None:
        summary_rows.append({"Ticker": tkr, "A_val_raw":0.0,"B_val_raw":0.0,"A_val_int":0,"B_val_int":0,
                             "A_label":gA,"B_label":gB})
        detail_map[tkr] = [{"__group__":"Missing inputs for similarity (need all RF features)."}]
        continue

    prox, order = _rf_proximity(ss.rf_model, x)
    if prox.size == 0:
        summary_rows.append({"Ticker": tkr, "A_val_raw":0.0,"B_val_raw":0.0,"A_val_int":0,"B_val_int":0,
                             "A_label":gA,"B_label":gB})
        detail_map[tkr] = [{"__group__":"No proximities computed."}]
        continue

    K = _elbow_kstar(prox, max_rank=30, k_min=3, k_max=min(25, prox.size))
    sel = order[:K]; y = ss.rf_model["y"][sel]
    cntA = int((y==1).sum()); cntB = int((y==0).sum()); tot = max(1, cntA+cntB)
    pA = 100.0*cntA/tot; pB = 100.0 - pA

    summary_rows.append({
        "Ticker": tkr,
        "A_val_raw": pA, "B_val_raw": pB,
        "A_val_int": int(round(pA)), "B_val_int": int(round(pB)),
        "A_label": gA, "B_label": gB,
    })

    # neighbor rows (top-K by proximity): show FT, Proximity, Ticker, and all RF features used
    meta = ss.rf_model.get("df_meta", pd.DataFrame())
    rows = [{"__group__": f"Top-{int(K)} neighbors by RF leaf proximity"}]
    if K > 0:
        for rnk, idx in enumerate(sel, 1):
            rec = {"#": rnk, "FT": int(ss.rf_model["y"][idx]), "Proximity": float(prox[rnk-1])}
            if not meta.empty and 0 <= idx < len(meta):
                m = meta.iloc[idx]
                rec["Ticker"] = m.get("TickerDB", m.get("Ticker", ""))
                for f in feat_names:
                    val = m.get(f, np.nan)
                    rec[f] = float(val) if (val is not None and np.isfinite(val)) else None
            rows.append(rec)
    detail_map[tkr] = rows

# ---------------- HTML/JS render ----------------
import streamlit.components.v1 as components

def _round_rec(o):
    if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
    if isinstance(o, list): return [_round_rec(v) for v in o]
    if isinstance(o, float): return float(np.round(o, 6))
    return o

payload = _round_rec({"rows": summary_rows, "details": detail_map, "gA": gA, "gB": gB, "feat_names": feat_names})

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
  .blue>span{background:#3b82f6}.red>span{background:#ef4444}
  #align td:nth-child(2),#align th:nth-child(2),#align td:nth-child(3),#align th:nth-child(3){text-align:center}
  .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:auto}
  .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .child-table th:first-child,.child-table td:first-child{text-align:left}
  tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
  .w-col{width:84px}
</style></head><body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead><tr><th>Ticker</th><th id="hdrA"></th><th id="hdrB"></th></tr></thead>
  </table>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;
    document.getElementById('hdrA').textContent = data.gA + " (blue)";
    document.getElementById('hdrB').textContent = data.gB + " (red)";

    function barCellLabeled(valRaw,label,valInt){
      const cls=(label==='FT=1')?'blue':'red';
      const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,Number(valRaw)));
      const text=(valInt==null||isNaN(valInt))?Math.round(w):Number(valInt);
      return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
    }
    function fmt(x){return (x==null||isNaN(x))?'':Number(x).toFixed(2);}
    function fmtP(x){return (x==null||isNaN(x))?'':Number(x).toFixed(4);}

    function childTableHTML(ticker){
      const rows=(data.details||{})[ticker]||[];
      if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No neighbors.</div>';

      // dynamic header for RF feature columns
      const fn = data.feat_names || [];
      const headF = fn.map(v=>`<th>${v}</th>`).join('');

      const cells = rows.map(r=>{
        if(r.__group__) return `<tr class="group-row"><td colspan="${4 + fn.length}">${r.__group__}</td></tr>`;
        const ft = (r.FT===1)?'FT=1':'FT=0';
        const tick = (r.Ticker||'');
        const feats = fn.map(v=>{
          const val = r[v];
          return `<td>${fmt(val)}</td>`;
        }).join('');
        return `<tr>
          <td class="w-col">${r["#"]||""}</td>
          <td>${ft}</td>
          <td class="w-col">${fmtP(r.Proximity)}</td>
          <td>${tick}</td>
          ${feats}
        </tr>`;
      }).join('');

      return `<table class="child-table">
        <thead><tr>
          <th class="w-col">#</th><th>FT</th><th class="w-col">Proximity</th><th>Ticker</th>${headF}
        </tr></thead>
        <tbody>${cells}</tbody>
      </table>`;
    }

    $(function(){
      const table=$('#align').DataTable({
        data: data.rows||[], responsive:true, paging:false, info:false, searching:false, order:[[0,'asc']],
        columns:[
          {data:'Ticker'},
          {data:null, render:(row)=>barCellLabeled(row.A_val_raw,row.A_label,row.A_val_int)},
          {data:null, render:(row)=>barCellLabeled(row.B_val_raw,row.B_label,row.B_val_int)}
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
