# ============================== Part 1 — Imports, Session, Helpers, Constants, Upload ==============================
import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, io, math, base64
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime

# Optional ML deps (graceful)
try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except Exception:
    _CATBOOST_OK = False

st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ---------------- Session alias & defaults ----------------
ss = st.session_state
ss.setdefault("rows", [])
ss.setdefault("last", {})
ss.setdefault("base_df", pd.DataFrame())
ss.setdefault("var_core", [])
ss.setdefault("var_moderate", [])
ss.setdefault("lassoA", {})
ss.setdefault("nca_model", {})
ss.setdefault("cat_model", {})
ss.setdefault("del_selection", [])
ss.setdefault("__delete_msg", None)
ss.setdefault("__catboost_warned", False)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---------------- String helpers ----------------
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
        if "," in ss_ and "." not in ss_: ss_ = ss_.replace(",", ".")
        else: ss_ = ss_.replace(",", "")
        return float(ss_)
    except Exception:
        return np.nan

def SAFE_JSON_DUMPS(obj) -> str:
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.ndarray,)): return o.tolist()
            return super().default(o)
    s = json.dumps(obj, cls=NpEncoder, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</script>", "<\\/script>")

# ---------------- Robust stats & calibration helpers ----------------
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

# ---------------- Seasonality helpers ----------------
def _pick_date_col(df: pd.DataFrame) -> str | None:
    cands = ["Date","Trade Date","TradeDate","Session Date","Session","Datetime","Timestamp"]
    hit = _pick(df, cands)
    if hit: return hit
    for c in df.columns:
        if "date" in str(c).lower(): return c
    return None

def _coerce_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")

def _winsor(s: pd.Series, lo_q=0.01, hi_q=0.99):
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().empty: return x
    lo, hi = np.nanquantile(x, lo_q), np.nanquantile(x, hi_q)
    return x.clip(lo, hi)

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float,float]:
    if n <= 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def _smooth_series(x_ord: np.ndarray, y_vals: np.ndarray, mode: str, sg_window=7, sg_poly=2, loess_span=0.25):
    y = y_vals.astype(float)
    if mode == "None" or len(y) < 3:
        return y
    if mode == "Savitzky–Golay":
        try:
            from scipy.signal import savgol_filter
            win = int(sg_window)
            if win > len(y): win = len(y) - (1 - len(y)%2)
            if win < 3: return y
            if sg_poly >= win: sg_poly = max(1, win-1)
            return savgol_filter(y, window_length=int(win), polyorder=int(sg_poly), mode="interp")
        except Exception:
            return y
    if mode == "LOESS":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            x = x_ord.astype(float)
            x0 = (x - x.min()) / max(1e-9, (x.max() - x.min()))
            yy = lowess(y, x0, frac=float(loess_span), return_sorted=False)
            return np.asarray(yy, dtype=float)
        except Exception:
            return y
    return y

# ---------------- Variables (global) ----------------
VAR_CORE = ["Gap_%","FR_x","PM$Vol/MC_%","Catalyst","PM_Vol_%","Max_Pull_PM_%","RVOL_Max_PM_cum"]
VAR_MODERATE = ["MC_PM_Max_M","Float_PM_Max_M","PM_Vol_M","PM_$Vol_M$","ATR_$","Daily_Vol_M","MarketCap_M$","Float_M"]
VAR_ALL = VAR_CORE + VAR_MODERATE

ALLOWED_LIVE_FEATURES = ["MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"]
EXCLUDE_FOR_NCA = {"PredVol_M","PM_Vol_%","Daily_Vol_M"}

# ---------------- Upload section (always visible) ----------------
def render_upload_section():
    st.subheader("Upload Database")
    uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
    build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")
    return uploaded, build_btn

uploaded, build_btn = render_upload_section()

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

# ============================== Part 2 — Build logic, modeling core, prediction, Add Stock ==============================
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
    ln_gapf   = np.log(np.clip(pd.to_numeric(df["Gap_%"], errors="coerce").values,  0, None) / 100.0 + eps)
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

    iso_bx = np.array(ss.get("lassoA", {}).get("iso_bx", []), dtype=float)
    iso_by = np.array(ss.get("lassoA", {}).get("iso_by", []), dtype=float)
    cal_mult = float(_iso_predict(iso_bx, iso_by, np.array([raw_mult]))[0]) if (iso_bx.size>=2 and iso_by.size>=2) else float(raw_mult)
    cal_mult = max(cal_mult, 1.0)

    PM = float(row.get("PM_Vol_M") or np.nan)
    if not np.isfinite(PM) or PM <= 0: return np.nan
    return float(PM * cal_mult)

# ---------------- Build block ----------------
if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            file_bytes = uploaded.getvalue()
            _ = _hash_bytes(file_bytes)
            raw, sel_sheet, _all = _load_sheet(file_bytes)

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
            if pmh_col is not None:
                pmh_raw = pd.to_numeric(raw[pmh_col].map(_to_float), errors="coerce")
                df["Max_Push_Daily_%"] = pmh_raw * 100.0
            else:
                df["Max_Push_Daily_%"] = np.nan

            # store
            ss["base_df"] = df
            ss["var_core"] = [v for v in VAR_CORE if v in df.columns]
            ss["var_moderate"] = [v for v in VAR_MODERATE if v in df.columns]
            ss["lassoA"] = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            st.success(f"Loaded “{sel_sheet}”. Base ready.")
            do_rerun()
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ---------------- Add Stock ----------------
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

# ============================== Part 3 — Tabs & Alignment ==============================
tab_align, tab_season = st.tabs(["Alignment", "Seasonality"])

with tab_align:
    st.markdown("---")
    st.subheader("Alignment")

    # Controls
    col_mode, col_gain = st.columns([2.8, 1.0])
    with col_mode:
        mode = st.radio(
            "",
            ["FT vs Fail (Gain% cutoff on FT=1 only)", "Gain% vs Rest"],
            horizontal=True, key="cmp_mode", label_visibility="collapsed",
        )
    with col_gain:
        gain_choices = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        gain_min = st.selectbox("", gain_choices,
                                index=gain_choices.index(100) if 100 in gain_choices else 0,
                                key="gain_min_pct", help="Threshold on Max Push Daily (%).",
                                label_visibility="collapsed")

    base_df = ss.get("base_df", pd.DataFrame()).copy()
    if base_df.empty:
        if ss.rows:
            st.info("Upload DB and click **Build model stocks** to compute group centers first.")
        else:
            st.info("Upload DB (and/or add at least one stock) to compute alignment.")
        st.stop()

    if "Max_Push_Daily_%" not in base_df.columns:
        st.error("Column “Max Push Daily (%)” not found in DB (expected as Max_Push_Daily_% after load)."); st.stop()
    if "FT01" not in base_df.columns:
        st.error("FT01 column not found (expected after load)."); st.stop()

    df_cmp = base_df.copy()
    thr = float(gain_min)

    if mode == "Gain% vs Rest":
        df_cmp["__Group__"] = np.where(pd.to_numeric(df_cmp["Max_Push_Daily_%"], errors="coerce") >= thr, f"≥{int(thr)}%", "Rest")
        gA, gB = f"≥{int(thr)}%", "Rest"
        status_line = f"Gain% split at ≥ {int(thr)}%"
    else:
        a_mask = (df_cmp["FT01"] == 1) & (pd.to_numeric(df_cmp["Max_Push_Daily_%"], errors="coerce") >= thr)
        b_mask = (df_cmp["FT01"] == 0)
        df_cmp = df_cmp[a_mask | b_mask].copy()
        df_cmp["__Group__"] = np.where(df_cmp["FT01"] == 1, f"FT=1 ≥{int(thr)}%", "FT=0 (all)")
        gA, gB = f"FT=1 ≥{int(thr)}%", "FT=0 (all)"
        status_line = f"A: FT=1 with Gain% ≥ {int(thr)}% • B: all FT=0 (no cutoff)"
    st.caption(status_line)

    var_core = ss.get("var_core", [])
    var_mod  = ss.get("var_moderate", [])
    var_all  = var_core + var_mod

    def _mad_local(series: pd.Series) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return np.nan
        med = float(np.median(s))
        return float(np.median(np.abs(s - med)))

    def _summaries_median_and_mad(df_in: pd.DataFrame, var_all: list[str], group_col: str):
        avail = [v for v in var_all if v in df_in.columns]
        if not avail:
            empty = pd.DataFrame()
            return {"med_tbl": empty, "mad_tbl": empty}
        g = df_in.groupby(group_col, observed=True)[avail]
        med_tbl = g.median(numeric_only=True).T
        mad_tbl = df_in.groupby(group_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad_local)).T
        return {"med_tbl": med_tbl, "mad_tbl": mad_tbl}

    summ = _summaries_median_and_mad(df_cmp, var_all, "__Group__")
    med_tbl = summ["med_tbl"]; mad_tbl = summ["mad_tbl"] * 1.4826

    if med_tbl.empty or med_tbl.shape[1] < 2:
        st.info("Not enough data to form two groups with the current mode/threshold. Adjust settings."); st.stop()

    cols = list(med_tbl.columns)
    if (gA in cols) and (gB in cols):
        med_tbl = med_tbl[[gA, gB]]
    else:
        top2 = df_cmp["__Group__"].value_counts().index[:2].tolist()
        if len(top2) < 2:
            st.info("One of the groups is empty. Adjust Gain% threshold."); st.stop()
        gA, gB = top2[0], top2[1]
        med_tbl = med_tbl[[gA, gB]]
    mad_tbl = mad_tbl.reindex(index=med_tbl.index)[[gA, gB]]

    # ------------- NCA / LDA -------------
    def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
        df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
        feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
        if not feats: return {}
        Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
        y = (df2["__Group__"].values == gA_label).astype(int)

        mask = np.isfinite(Xdf.values).all(axis=1)
        Xdf = Xdf.loc[mask]; y = y[mask]
        if Xdf.shape[0] < 20 or np.unique(y).size < 2:
            return {}

        X = Xdf.values
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
            # Fisher LDA fallback
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

        zf = z[np.isfinite(z)]; yf = y[np.isfinite(z)]
        iso_bx, iso_by = np.array([]), np.array([])
        platt_params = None
        if zf.size >= 8 and np.unique(zf).size >= 3:
            bx, by = _pav_isotonic(zf, yf.astype(float))
            if len(bx) >= 2:
                iso_bx, iso_by = np.array(bx), np.array(by)
        if iso_bx.size < 2:
            z0 = zf[yf==0]; z1 = zf[yf==1]
            if z0.size and z1.size:
                m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                platt_params = (m, k)

        return {"ok": True, "kind": used, "feats": feats,
                "mu": mu.tolist(), "sd": sd.tolist(),
                "w_vec": (w_vec.tolist() if w_vec is not None else None),
                "components": (components.tolist() if components is not None else None),
                "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
                "platt": platt_params, "gA": gA_label, "gB": gB_label}

    def _nca_predict_proba(row: dict, model: dict) -> float:
        if not model or not model.get("ok"): return np.nan
        feats = model["feats"]
        x = []
        for f in feats:
            v = pd.to_numeric(row.get(f), errors="coerce")
            if not np.isfinite(v): return np.nan
            x.append(float(v))
        x = np.array(x, dtype=float)
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

    # ------------- CatBoost -------------
    def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
        if not _CATBOOST_OK:
            if not ss.get("__catboost_warned", False):
                st.info("CatBoost is not installed. Run `pip install catboost` to enable CatBoost probabilities.")
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
        if n < 40 or np.unique(y).size < 2:
            return {}
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

        params = dict(loss_function="Logloss", eval_metric="Logloss",
                      iterations=200, learning_rate=0.04, depth=3, l2_leaf_reg=6,
                      bootstrap_type="Bayesian", bagging_temperature=0.5,
                      auto_class_weights="Balanced", random_seed=42,
                      allow_writing_files=False, verbose=False)
        if eval_ok: params.update(dict(od_type="Iter", od_wait=40))
        else: params.update(dict(od_type="None"))
        model = CatBoostClassifier(**params)

        try:
            if eval_ok: model.fit(Xtr, ytr, eval_set=(Xva, yva))
            else: model.fit(Xtr, ytr)
        except Exception:
            try:
                model = CatBoostClassifier(**{**params, "od_type": "None"})
                model.fit(X_all, y_all); eval_ok = False
            except Exception:
                return {}

        iso_bx = np.array([]); iso_by = np.array([]); platt = None
        if eval_ok:
            try:
                p_raw = model.predict_proba(Xva)[:, 1].astype(float)
                if np.unique(p_raw).size >= 3 and _has_both(yva):
                    bx, by = _pav_isotonic(p_raw, yva.astype(float))
                    if len(bx) >= 2:
                        iso_bx, iso_by = np.array(bx), np.array(by)
                if iso_bx.size < 2:
                    z0 = p_raw[yva==0]; z1 = p_raw[yva==1]
                    if z0.size and z1.size:
                        m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                        s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                        m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                        platt = (m, k)
            except Exception:
                pass
        else:
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
                tr2, va2 = next(sss2.split(X_all, y_all))
                pr = model.predict_proba(X_all[va2])[:, 1].astype(float); yv2 = y_all[va2]
                if np.unique(pr).size >= 3 and _has_both(yv2):
                    bx, by = _pav_isotonic(pr, yv2.astype(float))
                    if len(bx) >= 2:
                        iso_bx, iso_by = np.array(bx), np.array(by)
                if iso_bx.size < 2:
                    m0, m1 = float(np.mean(pr[yv2==0])), float(np.mean(pr[yv2==1]))
                    s0, s1 = float(np.std(pr[yv2==0])+1e-9), float(np.std(pr[yv2==1])+1e-9)
                    m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                    platt = (m, k)
            except Exception:
                pass
        return {"ok": True, "feats": feats, "gA": gA_label, "gB": gB_label,
                "cb": model, "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(), "platt": platt}

    def _cat_predict_proba(row: dict, model: dict) -> float:
        if not model or not model.get("ok"): return np.nan
        feats = model["feats"]; x = []
        for f in feats:
            v = pd.to_numeric(row.get(f), errors="coerce")
            if not np.isfinite(v): return np.nan
            x.append(float(v))
        x = np.array(x, dtype=float).reshape(1, -1)
        try:
            cb = model.get("cb")
            if cb is None: return np.nan
            z = float(cb.predict_proba(x)[0, 1])
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

    # Train NCA + Catboost for this split
    features_for_models = VAR_ALL[:]
    ss.nca_model = _train_nca_or_lda(df_cmp, gA, gB, features_for_models) or {}
    ss.cat_model = _train_catboost_once(df_cmp, gA, gB, features_for_models) or {}

    # ---------- Alignment scores for added rows ----------
    def _compute_alignment_counts_weighted(stock_row: dict, centers_tbl: pd.DataFrame, var_core: list[str], var_mod: list[str],
                                           w_core: float = 1.0, w_mod: float = 0.5, tie_mode: str = "split") -> dict:
        if centers_tbl is None or centers_tbl.empty or len(centers_tbl.columns) != 2:
            return {}
        gA_, gB_ = list(centers_tbl.columns)
        counts = {gA_: 0.0, gB_: 0.0}
        def _vote_one(var: str, weight: float):
            if var not in centers_tbl.index: return
            xv = pd.to_numeric(stock_row.get(var), errors="coerce")
            if not np.isfinite(xv): return
            vA = float(centers_tbl.at[var, gA_]); vB = float(centers_tbl.at[var, gB_])
            if np.isnan(vA) or np.isnan(vB): return
            dA = abs(xv - vA); dB = abs(xv - vB)
            if dA < dB: counts[gA_] += weight
            elif dB < dA: counts[gB_] += weight
            else:
                if tie_mode == "split": counts[gA_] += weight*0.5; counts[gB_] += weight*0.5

        for v in var_core: _vote_one(v, w_core)
        for v in var_mod:  _vote_one(v, w_mod)

        total = counts[gA_] + counts[gB_]
        a_raw = 100.0 * counts[gA_] / total if total > 0 else 0.0
        b_raw = 100.0 - a_raw
        return {"A": a_raw, "B": b_raw, "A_label": gA_, "B_label": gB_}

    centers_tbl = med_tbl.copy()
    rows_out = []
    for row in ss.rows:
        counts = _compute_alignment_counts_weighted(row, centers_tbl, var_core, var_mod, 1.0, 0.5, "split")
        if not counts: continue
        pN = _nca_predict_proba(row, ss.get("nca_model", {}))
        pC = _cat_predict_proba(row, ss.get("cat_model", {}))
        rows_out.append({
            "Ticker": row.get("Ticker","—"),
            f"{gA} (%) — Median centers": int(round(counts["A"])),
            f"{gB} (%) — Median centers": int(round(counts["B"])),
            f"NCA: P({gA}) (%)": (int(round(float(pN)*100)) if np.isfinite(pN) else None),
            f"CatBoost: P({gA}) (%)": (int(round(float(pC)*100)) if np.isfinite(pC) else None),
        })

    if rows_out:
        st.dataframe(pd.DataFrame(rows_out), use_container_width=True)
    else:
        st.info("Add at least one stock to compute alignment table.")

    # ---------- Radar ----------
    st.markdown("### Radar")
    if not ss.rows:
        st.info("Add at least one stock to see the radar chart.")
    else:
        def _available_live_axes():
            axes = [f for f in ALLOWED_LIVE_FEATURES if f in med_tbl.index and f not in EXCLUDE_FOR_NCA]
            order_hint = ["Gap_%","FR_x","PM$Vol/MC_%","Catalyst","Max_Pull_PM_%","RVOL_Max_PM_cum",
                          "MC_PM_Max_M","Float_PM_Max_M","PM_Vol_M","PM_$Vol_M$","ATR_$"]
            ordered_axes = [a for a in order_hint if a in axes]
            remaining = [a for a in axes if a not in ordered_axes]
            return ordered_axes + remaining
        axes_all = _available_live_axes()
        if not axes_all:
            st.info("No live features available for radar plotting with current split.")
        else:
            c1, c2 = st.columns([1.5, 2])
            with c1:
                feat_mode = st.radio("Features", ["Core", "All live"], index=0, key="radar_feat_mode")
                show_A = st.checkbox(f"Show {gA} center", value=True)
                show_B = st.checkbox(f"Show {gB} center", value=True)
                all_tks = []
                seen=set()
                for r in ss.rows:
                    t=(r.get("Ticker") or "—")
                    if t not in seen: all_tks.append(t); seen.add(t)
                sel = st.multiselect("Stocks", options=all_tks, default=all_tks[:5])
            with c2:
                core_live = [f for f in VAR_CORE if f in axes_all]
                axes = core_live if feat_mode=="Core" else axes_all
                centerA = {f: (float(med_tbl.at[f, gA]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gA])) else np.nan) for f in axes}
                centerB = {f: (float(med_tbl.at[f, gB]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gB])) else np.nan) for f in axes}

                N = len(axes)
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                angles_close = np.concatenate([angles, [angles[0]]])

                def _norm_minmax(values, a_center, b_center):
                    out = {}; eps=1e-9
                    for f, x in values.items():
                        a = a_center.get(f, np.nan); b = b_center.get(f, np.nan)
                        lo = np.nanmin([a,b]); hi = np.nanmax([a,b])
                        span = hi-lo
                        if not np.isfinite(span) or span<=eps:
                            lo, span = 0.0, 1.0
                        v = (x - lo) / (span + eps)
                        out[f] = float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else np.nan
                    return out

                rA = np.array([_norm_minmax(centerA, centerA, centerB)[f] for f in axes], dtype=float)
                rB = np.array([_norm_minmax(centerB, centerA, centerB)[f] for f in axes], dtype=float)
                rA_close = np.concatenate([rA, [rA[0]]]); rB_close = np.concatenate([rB, [rB[0]]])

                fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6.5, 6.5))
                ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
                ax.set_thetagrids(angles * 180/np.pi, labels=[f.replace('$', r'\$') for f in axes])
                ax.set_ylim(0, 1); ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["0.25","0.5","0.75","1.0"])
                if show_A: ax.plot(angles_close, np.concatenate([rA,[rA[0]]]), color="#3b82f6", linewidth=2, label=f"{gA} center")
                if show_B: ax.plot(angles_close, np.concatenate([rB,[rB[0]]]), color="#ef4444", linewidth=2, label=f"{gB} center")

                color_cycle = ["#06b6d4","#6366f1","#f59e0b","#10b981","#a3e635","#fb7185","#14b8a6"]
                idx=0
                for t in sel:
                    row = next((r for r in ss.rows if (r.get("Ticker") or "—")==t), None)
                    if not row: continue
                    vals = {f: float(pd.to_numeric(row.get(f), errors="coerce")) if np.isfinite(pd.to_numeric(row.get(f), errors="coerce")) else np.nan for f in axes}
                    normS = _norm_minmax(vals, centerA, centerB)
                    rS = np.array([normS.get(f, 0.5) for f in axes], dtype=float)
                    rS_close = np.concatenate([rS, [rS[0]]])
                    color = color_cycle[idx % len(color_cycle)]; idx+=1
                    ax.plot(angles_close, rS_close, color=color, linewidth=2, label=t)
                    ax.fill(angles_close, rS_close, color=color, alpha=0.18)
                ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05), frameon=False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

    # ---------- Distributions across Gain% cutoffs ----------
    st.markdown("### Distributions across Gain% cutoffs")
    if not ss.rows:
        st.info("Add at least one stock to see distributions.")
    else:
        # selected tickers
        all_tickers = []
        seen=set()
        for r in ss.rows:
            t=(r.get("Ticker") or "—")
            if t not in seen: all_tickers.append(t); seen.add(t)
        if "dist_stock_sel" not in st.session_state:
            st.session_state["dist_stock_sel"] = all_tickers[:]
        cols = st.columns([4,1])
        with cols[1]:
            if st.button("Clear selection"):
                st.session_state["dist_stock_sel"] = []
        with cols[0]:
            stocks_selected = st.multiselect("Stocks to include", options=all_tickers,
                                             default=st.session_state["dist_stock_sel"],
                                             key="dist_stock_sel")
        rows_for_dist = [r for r in ss.rows if (r.get("Ticker") or "—") in stocks_selected]
        if not rows_for_dist:
            st.info("No stocks selected.")
        else:
            gain_choices = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
            def _make_split(df_base: pd.DataFrame, thr_val: float, mode_val: str):
                df_tmp = df_base.copy()
                if mode_val == "Gain% vs Rest":
                    df_tmp["__Group__"] = np.where(pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val, f"≥{int(thr_val)}%", "Rest")
                    gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
                else:
                    a_mask = (df_tmp["FT01"] == 1) & (pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val)
                    b_mask = (df_tmp["FT01"] == 0)
                    df_tmp = df_tmp[a_mask | b_mask].copy()
                    df_tmp["__Group__"] = np.where(df_tmp["FT01"] == 1, f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)")
                    gA_, gB_ = f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)"
                return df_tmp, gA_, gB_

            def _summaries(df_in: pd.DataFrame, vars_all: list[str], grp_col: str):
                avail = [v for v in vars_all if v in df_in.columns]
                if not avail: return pd.DataFrame(), pd.DataFrame()
                g = df_in.groupby(grp_col, observed=True)[avail]
                med_tbl_ = g.median(numeric_only=True).T
                mad_tbl_ = df_in.groupby(grp_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad)).T * 1.4826
                return med_tbl_, mad_tbl_

            thr_labels = []; series_A_med, series_B_med, series_N_med, series_C_med = [], [], [], []

            for thr_val in gain_choices:
                df_split, gA2, gB2 = _make_split(base_df, float(thr_val), mode)
                med_tbl2, _ = _summaries(df_split, VAR_ALL, "__Group__")
                if med_tbl2.empty or med_tbl2.shape[1] < 2: continue
                cols2 = list(med_tbl2.columns)
                if (gA2 in cols2) and (gB2 in cols2):
                    med_tbl2 = med_tbl2[[gA2, gB2]]
                else:
                    top2 = df_split["__Group__"].value_counts().index[:2].tolist()
                    if len(top2) < 2: continue
                    gA2, gB2 = top2[0], top2[1]
                    med_tbl2 = med_tbl2[[gA2, gB2]]

                def _align_pct(row):
                    A = B = 0.0
                    for v in ss.var_core:
                        if v in med_tbl2.index and v in row and np.isfinite(pd.to_numeric(row.get(v), errors="coerce")):
                            x=float(row.get(v)); vA=float(med_tbl2.at[v,gA2]); vB=float(med_tbl2.at[v,gB2])
                            if abs(x-vA) < abs(x-vB): A += 1.0
                            elif abs(x-vB) < abs(x-vA): B += 1.0
                            else: A += 0.5; B += 0.5
                    for v in ss.var_moderate:
                        if v in med_tbl2.index and v in row and np.isfinite(pd.to_numeric(row.get(v), errors="coerce")):
                            x=float(row.get(v)); vA=float(med_tbl2.at[v,gA2]); vB=float(med_tbl2.at[v,gB2])
                            if abs(x-vA) < abs(x-vB): A += 0.5
                            elif abs(x-vB) < abs(x-vA): B += 0.5
                            else: A += 0.25; B += 0.25
                    tot=A+B
                    return (100.0*A/tot if tot>0 else np.nan, 100.0*B/tot if tot>0 else np.nan)

                A_vals=[]; B_vals=[]; N_vals=[]; C_vals=[]
                nca_model2 = _train_nca_or_lda(df_split, gA2, gB2, VAR_ALL) or {}
                cat_model2 = _train_catboost_once(df_split, gA2, gB2, VAR_ALL) or {}
                for r in rows_for_dist:
                    a,b = _align_pct(r)
                    A_vals.append(a); B_vals.append(b)
                    pN = _nca_predict_proba(r, nca_model2); pC = _cat_predict_proba(r, cat_model2)
                    N_vals.append((float(pN)*100.0) if np.isfinite(pN) else np.nan)
                    C_vals.append((float(pC)*100.0) if np.isfinite(pC) else np.nan)

                thr_labels.append(int(thr_val))
                series_A_med.append(float(np.nanmedian(A_vals)) if len(A_vals) else np.nan)
                series_B_med.append(float(np.nanmedian(B_vals)) if len(B_vals) else np.nan)
                series_N_med.append(float(np.nanmedian(N_vals)) if len(N_vals) else np.nan)
                series_C_med.append(float(np.nanmedian(C_vals)) if len(C_vals) else np.nan)

            if not thr_labels:
                st.info("Not enough data across cutoffs to form two groups — broaden your DB or change mode.")
            else:
                labA = f"{gA} (Median centers)"
                labB = f"{gB} (Median centers)"
                labN = f"NCA: P({gA})"
                labC = f"CatBoost: P({gA})"
                dist_df = pd.DataFrame({
                    "GainCutoff_%": thr_labels,
                    labA: series_A_med,
                    labB: series_B_med,
                    labN: series_N_med,
                    labC: series_C_med,
                })
                df_long = dist_df.melt(id_vars="GainCutoff_%", var_name="Series", value_name="Value")
                color_domain = [labA, labB, labN, labC]
                color_range  = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]
                chart = (
                    alt.Chart(df_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("GainCutoff_%:O", title="Gain% cutoff"),
                        y=alt.Y("Value:Q", title="Median across selected stocks (%)", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title="")),
                        xOffset="Series:N",
                        tooltip=["GainCutoff_%:O","Series:N",alt.Tooltip("Value:Q", format=".1f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)
                # Exports
                png_bytes = None
                try:
                    pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
                    series_names = list(pivot.columns)
                    color_map = {labA:"#3b82f6", labB:"#ef4444", labN:"#10b981", labC:"#8b5cf6"}
                    colors = [color_map.get(s, "#999999") for s in series_names]
                    thresholds = pivot.index.tolist()
                    n_groups = len(thresholds); n_series = len(series_names)
                    x = np.arange(n_groups); width = 0.8 / max(n_series, 1)
                    fig, ax = plt.subplots(figsize=(max(6, n_groups*0.6), 4))
                    for i, s in enumerate(series_names):
                        vals = pivot[s].values.astype(float)
                        ax.bar(x + i*width - (n_series-1)*width/2, vals, width=width, label=s, color=colors[i])
                    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in thresholds])
                    ax.set_ylim(0, 100); ax.set_xlabel("Gain% cutoff"); ax.set_ylabel("Median across selected stocks (%)")
                    ax.legend(loc="upper left", frameon=False)
                    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight"); plt.close(fig)
                    png_bytes = buf.getvalue()
                except Exception:
                    png_bytes = None
                st.markdown("#### Export distributions")
                c1,c2 = st.columns(2)
                with c1:
                    if png_bytes:
                        st.download_button("Download PNG", data=png_bytes, file_name=f"distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                           mime="image/png", use_container_width=True)
                with c2:
                    spec = chart.to_dict()
                    html_tpl = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Distribution</title>
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
                    st.download_button("Download HTML (interactive)", data=html_tpl.encode("utf-8"),
                                       file_name="distribution_chart.html", mime="text/html", use_container_width=True)

    # ---------- Export alignment summary (CSV + Markdown) ----------
    if rows_out:
        df_summary = pd.DataFrame(rows_out)
        def _df_to_markdown_simple(df: pd.DataFrame, float_fmt=".0f") -> str:
            def _fmt(x):
                if x is None: return ""
                if isinstance(x, float):
                    if math.isnan(x) or math.isinf(x): return ""
                    return format(x, float_fmt)
                return str(x)
            cols = list(df.columns)
            header = "| " + " | ".join(cols) + " |"
            sep    = "| " + " | ".join("---" for _ in cols) + " |"
            lines = [header, sep]
            for _, row in df.iterrows():
                cells = [_fmt(v) for v in row.tolist()]
                lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)

        st.markdown("#### Export alignment summary")
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("Download CSV", data=df_summary.to_csv(index=False).encode("utf-8"),
                               file_name="alignment_summary.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("Download Markdown", data=_df_to_markdown_simple(df_summary).encode("utf-8"),
                               file_name="alignment_summary.md", mime="text/markdown", use_container_width=True)

# ============================== Part 4 — Seasonality (Seasonality tab) ==============================
with tab_season:
    st.markdown("---")
    st.subheader("Seasonality")

    base_df = ss.get("base_df", pd.DataFrame()).copy()
    if base_df.empty:
        st.info("Upload DB and click **Build model stocks** to analyze seasonality.")
        st.stop()

    # Date handling
    date_col = _pick_date_col(base_df)
    if not date_col:
        st.error("No Date column detected (e.g., 'Date'). Add one to your DB to use Seasonality."); st.stop()

    base_df["__DATE__"] = _coerce_date_col(base_df, date_col)
    if not pd.to_datetime(base_df["__DATE__"], errors="coerce").notna().any():
        st.error(f"Could not parse any valid dates from column '{date_col}'."); st.stop()

    df_time = base_df.dropna(subset=["__DATE__"]).copy()
    df_time["Year"] = df_time["__DATE__"].dt.year
    df_time["Month"] = df_time["__DATE__"].dt.month
    df_time["Week"] = df_time["__DATE__"].dt.isocalendar().week.astype(int)
    df_time["DOW"] = df_time["__DATE__"].dt.dayofweek    # 0=Mon
    df_time["DOM"] = df_time["__DATE__"].dt.day

    # Controls
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        bucket = st.selectbox("Seasonality bucket", ["Year","Month","Week","DOW","DOM"], index=1)
    with c2:
        target_kind = st.selectbox("Metric type", ["Numeric", "Rate (FT%)"], index=0,
                                   help="Choose 'Rate (FT%)' for proportion of FT=1; else aggregate numeric variable.")
    with c3:
        smooth_mode = st.selectbox("Smoothing", ["None","Savitzky–Golay","LOESS"], index=0)
    with c4:
        winsorize = st.checkbox("Winsorize 1–99% (numeric)", value=True)

    # Target selection
    numeric_candidates = [c for c in base_df.columns if c not in {"__DATE__"} and pd.api.types.is_numeric_dtype(base_df[c])]
    rate_col = "FT" if "FT" in base_df.columns else ("FT01" if "FT01" in base_df.columns else None)

    if target_kind == "Rate (FT%)":
        if not rate_col:
            st.warning("No FT/FT01 column found to compute FT%."); st.stop()
        target = rate_col
        st.caption("Computing FT% per bucket with Wilson 95% CIs.")
    else:
        priority = ["Max_Push_Daily_%","Gap_%","PM_Vol_%","Daily_Vol_M","ATR_$","FR_x","PM$Vol/MC_%"]
        ordered = [c for c in priority if c in numeric_candidates] + [c for c in numeric_candidates if c not in priority]
        if not ordered:
            st.warning("No numeric variables found for Seasonality."); st.stop()
        target = st.selectbox("Numeric variable", ordered, index=0)
        st.caption("Median per bucket (winsorized if enabled), with optional smoothing overlay.")

    # Aggregate & visualize
    if target_kind == "Rate (FT%)":
        grp = df_time.groupby(bucket, observed=True)
        n = grp[target].count()
        k = grp[target].sum()
        p = (k / n).astype(float)
        lo, hi = zip(*[ _wilson_ci(int(kv), int(nv)) for kv, nv in zip(k.fillna(0).astype(int), n.fillna(0).astype(int)) ])
        out = pd.DataFrame({bucket: n.index, "count": n.values, "FT%": (p.values*100.0), "lo": np.array(lo)*100.0, "hi": np.array(hi)*100.0})
        out = out.sort_values(by=bucket)

        base = alt.Chart(out).mark_bar().encode(
            x=f"{bucket}:O",
            y=alt.Y("FT%:Q", title="FT rate (%)", scale=alt.Scale(domain=[0,100])),
            tooltip=[bucket, alt.Tooltip("FT%:Q", format=".1f"), "count:Q", alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")]
        )
        rule = alt.Chart(out).mark_rule().encode(
            x=f"{bucket}:O",
            y="lo:Q", y2="hi:Q",
            tooltip=[bucket, alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")]
        )
        st.altair_chart((base + rule).properties(height=340), use_container_width=True)
        st.dataframe(out, use_container_width=True)

    else:
        ser = df_time[target]
        if winsorize:
            ser = _winsor(ser, 0.01, 0.99)
        dfp = df_time[[bucket]].copy()
        dfp["val"] = pd.to_numeric(ser, errors="coerce")
        agg = dfp.groupby(bucket, observed=True)["val"].median().reset_index()
        agg = agg.sort_values(by=bucket)

        x_ord = agg[bucket].astype(float).values
        y_vals = agg["val"].astype(float).values
        y_smooth = _smooth_series(x_ord, y_vals, smooth_mode)

        base = alt.Chart(agg).mark_bar().encode(
            x=f"{bucket}:O",
            y=alt.Y("val:Q", title=f"Median {target}"),
            tooltip=[bucket, alt.Tooltip("val:Q", format=".2f")]
        )
        line = alt.Chart(pd.DataFrame({bucket: agg[bucket], "sm": y_smooth})).mark_line().encode(
            x=f"{bucket}:O",
            y=alt.Y("sm:Q", title=f"Smoothed {target}"),
            tooltip=[bucket, alt.Tooltip("sm:Q", format=".2f")]
        )
        st.altair_chart((base + line).resolve_scale(y='independent').properties(height=340), use_container_width=True)
        st.dataframe(agg.rename(columns={"val": f"Median({target})"}), use_container_width=True)

# ============================== Part 5 — Footer / Small Utilities ==============================
st.markdown("---")
st.caption("Premarket Stock Analysis • Alignment + Seasonality • © Your Team")

# Nothing else needed — app is ready.
