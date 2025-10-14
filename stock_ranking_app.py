# app.py — Premarket Stock Ranking
# (Median-only centers; Gain% filter; 3σ coloring; delete UI callback-safe; NCA bar w/ live features; distributions w/ clear labels)
# + CatBoost P(A) (purple) trained once per split; rendered as 4th column & series
# + Robust components.html() try-chain
# + Horizontal scroll & fixed widths so columns always visible
# + Light efficiency passes without changing behavior

import streamlit as st
import pandas as pd
import numpy as np
import re, json, hashlib, io, math
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from datetime import datetime

# CatBoost import (graceful if unavailable)
try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except Exception:
    _CATBOOST_OK = False

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Analysis", layout="wide")
st.title("Premarket Stock Analysis")

# ============================== Helpers ==============================
_norm_cache = {}
def _norm(s: str) -> str:
    if s in _norm_cache: return _norm_cache[s]
    v = re.sub(r"\s+", " ", str(s).strip().lower())
    v = v.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")
    _norm_cache[s] = v
    return v

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

# ====== Seasonality helpers ======
def _pick_date_col(df: pd.DataFrame) -> str | None:
    cands = ["Date","Trade Date","TradeDate","Session Date","Session","Datetime","Timestamp"]
    hit = _pick(df, cands)
    if hit: return hit
    for c in df.columns:
        if "date" in str(c).lower(): return c
    return None

def _coerce_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")

def _mad_sigma(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    med = np.median(s)
    return 1.4826 * np.median(np.abs(s - med))

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float,float]:
    # 95% Wilson interval by default
    if n <= 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

# Optional winsorization
def _winsor(s: pd.Series, lo_q=0.01, hi_q=0.99):
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = np.nanquantile(x, lo_q), np.nanquantile(x, hi_q)
    return x.clip(lo, hi)

# Smoothing (Savitzky–Golay / LOESS) with safe fallbacks
def _smooth_series(x_ord: np.ndarray, y_vals: np.ndarray, mode: str, sg_window=7, sg_poly=2, loess_span=0.25):
    y = y_vals.astype(float)
    if mode == "None" or len(y) < 3: 
        return y
    # try SciPy SG
    if mode == "Savitzky–Golay":
        try:
            from scipy.signal import savgol_filter
            win = min(len(y) - (1 - len(y)%2), sg_window)  # odd <= len(y)
            if win < 3: return y
            if sg_poly >= win: sg_poly = max(1, win-1)
            return savgol_filter(y, window_length=win, polyorder=sg_poly, mode="wrap")
        except Exception:
            return y
    # try statsmodels LOESS
    if mode == "LOESS":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            # normalize x to [0,1] for numerical stability
            x = x_ord.astype(float)
            x0 = (x - x.min()) / max(1e-9, (x.max() - x.min()))
            yy = lowess(y, x0, frac=float(loess_span), return_sorted=False)
            return np.asarray(yy, dtype=float)
        except Exception:
            return y
    return y

def _fmt_pct(x):
    return "" if not np.isfinite(x) else f"{x:.1f}%"

def _fmt_num(x):
    return "" if not np.isfinite(x) else f"{x:.2f}"

# ============================== Upload Section Helper ==============================
def render_upload_section():
    st.subheader("Upload Database")

    uploaded = st.file_uploader(
        "Upload .xlsx with your DB", 
        type=["xlsx"], 
        key="db_upl"
    )
    build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

    return uploaded, build_btn

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

            st.session_state["base_df"] = df
            ss.var_core = [v for v in VAR_CORE if v in df.columns]
            ss.var_moderate = [v for v in VAR_MODERATE if v in df.columns]

            # train model once (on full base)
            ss.lassoA = train_ratio_winsor_iso(df, lo_q=0.01, hi_q=0.99) or {}

            st.success(f"Loaded “{sel_sheet}”. Base ready.")
            do_rerun()
        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)



tab_align, tab_season = st.tabs(["Alignment", "Seasonality"])

with tab_align:   
    # ============================== Session ==============================
    ss = st.session_state
    ss.setdefault("rows", [])
    ss.setdefault("last", {})
    ss.setdefault("lassoA", {})
    ss.setdefault("base_df", pd.DataFrame())
    ss.setdefault("var_core", [])
    ss.setdefault("var_moderate", [])
    ss.setdefault("nca_model", {})
    ss.setdefault("cat_model", {})           # CatBoost model per current split
    ss.setdefault("del_selection", [])       # for delete UI
    ss.setdefault("__delete_msg", None)      # flash msg
    ss.setdefault("__catboost_warned", False)
    
    
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
    
    # NCA/CatBoost: only “live” features from Add Stock (exclude PredVol_M / PM_Vol_% / Daily_Vol_M)
    ALLOWED_LIVE_FEATURES = [
        "MC_PM_Max_M","Float_PM_Max_M","Gap_%","ATR_$","PM_Vol_M","PM_$Vol_M$",
        "FR_x","PM$Vol/MC_%","Max_Pull_PM_%","RVOL_Max_PM_cum","Catalyst"
    ]
    EXCLUDE_FOR_NCA = {"PredVol_M","PM_Vol_%","Daily_Vol_M"}
    
    # ============================== LASSO (unchanged core) ==============================
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
    
    # ============================== Predict (daily) ==============================
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
    st.markdown("---")
    st.subheader("Alignment")
    
    # --- mode + Gain% ---
    col_mode, col_gain = st.columns([2.8, 1.0])
    with col_mode:
        mode = st.radio(
            "",
            [
                "FT vs Fail (Gain% cutoff on FT=1 only)",
                "Gain% vs Rest",
            ],
            horizontal=True,
            key="cmp_mode",
            label_visibility="collapsed",
        )
    with col_gain:
        gain_choices = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        gain_min = st.selectbox(
            "",
            gain_choices,
            index=gain_choices.index(100) if 100 in gain_choices else 0,
            key="gain_min_pct",
            help="Threshold on Max Push Daily (%).",
            label_visibility="collapsed",
        )
    
    # ============================== Delete (top-right, no expander) ==============================
    # Build unique tickers preserving order
    tickers = [r.get("Ticker") for r in ss.rows if r.get("Ticker")]
    unique_tickers, _seen = [], set()
    for t in tickers:
        if t and t not in _seen:
            unique_tickers.append(t); _seen.add(t)
    
    def _handle_delete():
        sel = st.session_state.get("del_selection", [])
        if sel:
            ss.rows = [r for r in ss.rows if r.get("Ticker") not in set(sel)]
            st.session_state["del_selection"] = []
            st.session_state["__delete_msg"] = f"Deleted: {', '.join(sel)}"
        else:
            st.session_state["__delete_msg"] = "No tickers selected."
    
    cdel1, cdel2 = st.columns([4, 1])
    with cdel1:
        _ = st.multiselect(
            "",
            options=unique_tickers,
            default=ss.get("del_selection", []),
            key="del_selection",
            placeholder="Select tickers…",
            label_visibility="collapsed",
        )
    with cdel2:
        st.button(
            "Delete",
            use_container_width=True,
            key="delete_btn",
            disabled=not bool(ss.get("del_selection")),
            on_click=_handle_delete,
        )
    
    # flash message
    _msg = st.session_state.pop("__delete_msg", None)
    if _msg:
        (st.success if _msg.startswith("Deleted:") else st.info)(_msg)
    
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
        df_cmp["__Group__"] = np.where(pd.to_numeric(df_cmp["Max_Push_Daily_%"], errors="coerce") >= thr, f"≥{int(thr)}%", "Rest")
        gA, gB = f"≥{int(thr)}%", "Rest"
        status_line = f"Gain% split at ≥ {int(thr)}%"
    else:  # "FT vs Fail (Gain% cutoff on FT=1 only)"
        a_mask = (df_cmp["FT01"] == 1) & (pd.to_numeric(df_cmp["Max_Push_Daily_%"], errors="coerce") >= thr)
        b_mask = (df_cmp["FT01"] == 0)
        df_cmp = df_cmp[a_mask | b_mask].copy()
        df_cmp["__Group__"] = np.where(df_cmp["FT01"] == 1, f"FT=1 ≥{int(thr)}%", "FT=0 (all)")
        gA, gB = f"FT=1 ≥{int(thr)}%", "FT=0 (all)"
        status_line = f"A: FT=1 with Gain% ≥ {int(thr)}% • B: all FT=0 (no cutoff)"
    
    st.caption(status_line)
    
    # ---------- summaries (median centers + MAD→σ for 3σ highlighting) ----------
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
    med_tbl = summ["med_tbl"]; mad_tbl = summ["mad_tbl"] * 1.4826  # MAD → σ
    
    # ensure exactly two groups exist
    if med_tbl.empty or med_tbl.shape[1] < 2:
        st.info("Not enough data to form two groups with the current mode/threshold. Adjust settings.")
        st.stop()
    
    cols = list(med_tbl.columns)
    if (gA in cols) and (gB in cols):
        med_tbl = med_tbl[[gA, gB]]
    else:
        top2 = df_cmp["__Group__"].value_counts().index[:2].tolist()
        if len(top2) < 2:
            st.info("One of the groups is empty. Adjust Gain% threshold.")
            st.stop()
        gA, gB = top2[0], top2[1]
        med_tbl = med_tbl[[gA, gB]]
    
    mad_tbl = mad_tbl.reindex(index=med_tbl.index)[[gA, gB]]
    
    # ============================== NCA training (live features only) ==============================
    def _train_nca_or_lda(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
        df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
    
        feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
        if not feats:
            return {}
    
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
        if not feats:
            return {}
    
        X = df2[feats].apply(pd.to_numeric, errors="coerce").values
        y = (df2["__Group__"].values == gA_label).astype(int)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]; y = y[mask]
        if X.shape[0] < 20 or np.unique(y).size < 2:
            return {}
    
        # standardize
        mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd==0] = 1.0
        Xs = (X - mu) / sd
    
        used = "lda"; w_vec = None; components = None
        try:
            from sklearn.neighbors import NeighborhoodComponentsAnalysis
            nca = NeighborhoodComponentsAnalysis(n_components=1, random_state=42, max_iter=400)
            z = nca.fit_transform(Xs, y).ravel()
            used = "nca"
            # FIX: Save the learned transformation components
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
    
        # Orient so larger → A
        if np.nanmean(z[y==1]) < np.nanmean(z[y==0]):
            z = -z
            if w_vec is not None: w_vec = -w_vec
            if components is not None: components = -components
    
        # Calibration: isotonic preferred; Platt fallback
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
                m = 0.5*(m0+m1)
                k = 2.0 / (0.5*(s0+s1) + 1e-6)
                platt_params = (m, k)
    
        return {
            "ok": True, "kind": used, "feats": feats,
            "mu": mu.tolist(), "sd": sd.tolist(),
            "w_vec": (w_vec.tolist() if w_vec is not None else None),
            "components": (components.tolist() if components is not None else None), # FIX: Add components to model dict
            "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
            "platt": (platt_params if platt_params is not None else None),
            "gA": gA_label, "gB": gB_label,
        }
    
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
    
        z = np.nan
        if model["kind"] == "lda":
            w = np.array(model.get("w_vec"), dtype=float)
            if w is None or not np.isfinite(w).all(): return np.nan
            z = float(xs @ w)
        else: # kind == "nca"
            # FIX: Use the saved transformation components for prediction
            comp = model.get("components")
            if comp is None: return np.nan
            w = np.array(comp, dtype=float).ravel()
            if w.size != xs.size: return np.nan # Guardrail
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
    
    # ============================== CatBoost training (same live features; once per split) ==============================
    def _train_catboost_once(df_groups: pd.DataFrame, gA_label: str, gB_label: str, features: list[str]) -> dict:
        """Train CatBoost on live features for this split with strong guards:
           - numeric-only, finite, float32 features
           - stratified split with both classes in train/val when possible
           - safe fallback when val is degenerate
           - probability calibration (isotonic → Platt)
        """
        if not _CATBOOST_OK:
            if not ss.get("__catboost_warned", False):
                st.info("CatBoost is not installed. Run `pip install catboost` to enable the purple CatBoost column/series.")
                ss["__catboost_warned"] = True
            return {}
    
        # 1) Select rows & features (applying same exclusion as NCA)
        df2 = df_groups[df_groups["__Group__"].isin([gA_label, gB_label])].copy()
        feats = [f for f in features if f in df2.columns and f in ALLOWED_LIVE_FEATURES and f not in EXCLUDE_FOR_NCA]
        if not feats:
            return {}
    
        Xdf = df2[feats].apply(pd.to_numeric, errors="coerce")
        y = (df2["__Group__"].values == gA_label).astype(int)
    
        # 2) Drop any rows with non-finite values
        mask_finite = np.isfinite(Xdf.values).all(axis=1)
        Xdf = Xdf.loc[mask_finite]
        y = y[mask_finite]
    
        # Need enough data and two classes
        n = len(y)
        if n < 40 or np.unique(y).size < 2:
            return {}
    
        # Cast to float32 for CatBoost stability on numeric features
        X_all = Xdf.values.astype(np.float32, copy=False)
        y_all = y.astype(np.int32, copy=False)
    
        # 3) Stratified split (try to ensure both classes in val)
        from sklearn.model_selection import StratifiedShuffleSplit
        test_size = 0.2 if n >= 100 else max(0.15, min(0.25, 20 / n))  # 15–25% val; ensure some samples on small sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        tr_idx, va_idx = next(sss.split(X_all, y_all))
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y_all[tr_idx], y_all[va_idx]
    
        # Check class coverage in train/val
        def _has_both_classes(arr):
            u = np.unique(arr)
            return (u.size == 2)
    
        eval_ok = (len(yva) >= 8) and _has_both_classes(yva) and _has_both_classes(ytr)
    
        # 4) Build model (enable early stopping only if eval_set is valid)
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
        if eval_ok:
            params.update(dict(od_type="Iter", od_wait=40))
        else:
            # No reliable eval → disable early stopping
            params.update(dict(od_type="None"))
    
        model = CatBoostClassifier(**params)
    
        # 5) Fit with safe fallbacks
        try:
            if eval_ok:
                model.fit(Xtr, ytr, eval_set=(Xva, yva))
            else:
                model.fit(Xtr, ytr)
        except Exception:
            # Retry: train on all data without early stopping
            try:
                model = CatBoostClassifier(**{**params, "od_type": "None"})
                model.fit(X_all, y_all)
                # If we trained on all data, we won't have a clean val for isotonic. We'll use Platt later.
                eval_ok = False
            except Exception:
                return {}
    
        # 6) Calibration
        iso_bx = np.array([]); iso_by = np.array([]); platt = None
        if eval_ok:
            try:
                p_raw = model.predict_proba(Xva)[:, 1].astype(float)
                # Need variety & both classes for isotonic
                if np.unique(p_raw).size >= 3 and _has_both_classes(yva):
                    bx, by = _pav_isotonic(p_raw, yva.astype(float))
                    if len(bx) >= 2:
                        iso_bx, iso_by = np.array(bx), np.array(by)
                if iso_bx.size < 2:
                    # Platt from validation
                    z0 = p_raw[yva==0]; z1 = p_raw[yva==1]
                    if z0.size and z1.size:
                        m0, m1 = float(np.mean(z0)), float(np.mean(z1))
                        s0, s1 = float(np.std(z0)+1e-9), float(np.std(z1)+1e-9)
                        m = 0.5*(m0+m1)
                        k = 2.0 / (0.5*(s0+s1) + 1e-6)
                        platt = (m, k)
            except Exception:
                # Fallback: simple Platt using train split
                try:
                    p_raw = model.predict_proba(Xtr)[:, 1].astype(float)
                    if _has_both_classes(ytr):
                        m0, m1 = float(np.mean(p_raw[ytr==0])), float(np.mean(p_raw[ytr==1]))
                        s0, s1 = float(np.std(p_raw[ytr==0])+1e-9), float(np.std(p_raw[ytr==1])+1e-9)
                        m = 0.5*(m0+m1); k = 2.0 / (0.5*(s0+s1) + 1e-6)
                        platt = (m, k)
                except Exception:
                    pass
        else:
            # No eval set: Platt from a small internal split on train if possible
            try:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
                tr2, va2 = next(sss2.split(X_all, y_all))
                pr = model.predict_proba(X_all[va2])[:, 1].astype(float)
                yv2 = y_all[va2]
                if np.unique(pr).size >= 3 and _has_both_classes(yv2):
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
    
        return {
            "ok": True,
            "feats": feats,
            "gA": gA_label, "gB": gB_label,
            "cb": model,
            "iso_bx": iso_bx.tolist(), "iso_by": iso_by.tolist(),
            "platt": platt
        }
    
    def _cat_predict_proba(row: dict, model: dict) -> float:
        if not model or not model.get("ok"): return np.nan
        feats = model["feats"]
        x = []
        for f in feats:
            v = pd.to_numeric(row.get(f), errors="coerce")
            if not np.isfinite(v): return np.nan
            x.append(float(v))
        x = np.array(x, dtype=float).reshape(1, -1)
    
        try:
            cb = model.get("cb")
            if cb is None: return np.nan
            z = float(cb.predict_proba(x)[0, 1])  # raw prob of class A
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
    
    # Train NCA + CatBoost once for the current split
    features_for_models = VAR_ALL[:]  # filtered in trainer to live features
    ss.nca_model = _train_nca_or_lda(df_cmp, gA, gB, features_for_models) or {}
    ss.cat_model = _train_catboost_once(df_cmp, gA, gB, features_for_models) or {}
    
    # ---------- alignment computation for entered rows ----------
    def _compute_alignment_counts_weighted(
        stock_row: dict,
        centers_tbl: pd.DataFrame,
        var_core: list[str],
        var_mod: list[str],
        w_core: float = 1.0,
        w_mod: float = 0.5,
        tie_mode: str = "split",
    ) -> dict:
        if centers_tbl is None or centers_tbl.empty or len(centers_tbl.columns) != 2:
            return {}
        gA_, gB_ = list(centers_tbl.columns)
        counts = {gA_: 0.0, gB_: 0.0}
        core_pts = {gA_: 0.0, gB_: 0.0}
        mod_pts  = {gA_: 0.0, gB_: 0.0}
        idx_set = set(centers_tbl.index)
    
        def _vote_one(var: str, weight: float, bucket: dict):
            if var not in idx_set: return
            xv = pd.to_numeric(stock_row.get(var), errors="coerce")
            if not np.isfinite(xv): return
            vA = float(centers_tbl.at[var, gA_]); vB = float(centers_tbl.at[var, gB_])
            if np.isnan(vA) or np.isnan(vB): return
            dA = abs(xv - vA); dB = abs(xv - vB)
            if dA < dB:
                counts[gA_] += weight; bucket[gA_] += weight
            elif dB < dA:
                counts[gB_] += weight; bucket[gB_] += weight
            else:
                if tie_mode == "split":
                    counts[gA_] += weight*0.5; counts[gB_] += weight*0.5
                    bucket[gA_] += weight*0.5; bucket[gB_] += weight*0.5
    
        for v in var_core: _vote_one(v, w_core, core_pts)
        for v in var_mod:  _vote_one(v, w_mod,  mod_pts)
    
        total = counts[gA_] + counts[gB_]
        a_raw = 100.0 * counts[gA_] / total if total > 0 else 0.0
        b_raw = 100.0 - a_raw
        a_int = int(round(a_raw)); b_int = 100 - a_int
    
        return {
            gA_: counts[gA_], gB_: counts[gB_],
            "A_pts": counts[gA_], "B_pts": counts[gB_],
            "A_core": core_pts[gA_], "B_core": core_pts[gB_],
            "A_mod":  mod_pts[gA_],  "B_mod":  mod_pts[gB_],
            "A_pct_raw": a_raw, "B_pct_raw": b_raw,
            "A_pct_int": a_int, "B_pct_int": b_int,
            "A_label": gA_, "B_label": gB_,
        }
    
    centers_tbl = med_tbl.copy()
    disp_tbl = mad_tbl.copy()
    
    summary_rows, detail_map = [], {}
    detail_order = [("Core variables", var_core),
                    ("Moderate variables", var_mod + (["PredVol_M"] if "PredVol_M" not in var_mod else []))]
    
    mt_index = set(centers_tbl.index); dt_index = set(disp_tbl.index)
    
    for row in ss.rows:
        stock = dict(row); tkr = stock.get("Ticker") or "—"
        counts = _compute_alignment_counts_weighted(
            stock_row=stock, centers_tbl=centers_tbl, var_core=var_core, var_mod=var_mod,
            w_core=1.0, w_mod=0.5, tie_mode="split",
        )
        if not counts: continue
    
        # NCA probability (A)
        pA = _nca_predict_proba(stock, ss.get("nca_model", {}))
        nca_raw = float(pA)*100.0 if np.isfinite(pA) else np.nan
        nca_int = int(round(nca_raw)) if np.isfinite(nca_raw) else None
    
        # CatBoost probability (A)
        pC = _cat_predict_proba(stock, ss.get("cat_model", {}))
        cat_raw = float(pC)*100.0 if np.isfinite(pC) else np.nan
        cat_int = int(round(cat_raw)) if np.isfinite(cat_raw) else None
    
        summary_rows.append({
            "Ticker": tkr,
            "A_val_raw": counts.get("A_pct_raw", 0.0),
            "B_val_raw": counts.get("B_pct_raw", 0.0),
            "A_val_int": counts.get("A_pct_int", 0),
            "B_val_int": counts.get("B_pct_int", 0),
            "A_label": counts.get("A_label", gA),
            "B_label": counts.get("B_label", gB),
            "A_pts": counts.get("A_pts", 0.0),
            "B_pts": counts.get("B_pts", 0.0),
            "A_core": counts.get("A_core", 0.0),
            "B_core": counts.get("B_core", 0.0),
            "A_mod": counts.get("A_mod", 0.0),
            "B_mod": counts.get("B_mod", 0.0),
            "NCA_raw": nca_raw,
            "NCA_int": nca_int,
            "CAT_raw": cat_raw,
            "CAT_int": cat_int,
        })
    
        drows_grouped = []
        for grp_label, grp_vars in detail_order:
            drows_grouped.append({"__group__": grp_label})
            for v in grp_vars:
                if v == "Daily_Vol_M": continue
                va = pd.to_numeric(stock.get(v), errors="coerce")
    
                med_var = "Daily_Vol_M" if v in ("PredVol_M",) else v
                vA = centers_tbl.at[med_var, gA] if med_var in mt_index else np.nan
                vB = centers_tbl.at[med_var, gB] if med_var in mt_index else np.nan
    
                sA = float(disp_tbl.at[med_var, gA]) if (med_var in dt_index and pd.notna(disp_tbl.at[med_var, gA])) else np.nan
                sB = float(disp_tbl.at[med_var, gB]) if (med_var in dt_index and pd.notna(disp_tbl.at[med_var, gB])) else np.nan
    
                if v not in ("PredVol_M",) and pd.isna(va) and pd.isna(vA) and pd.isna(vB): continue
    
                dA = None if (pd.isna(va) or pd.isna(vA)) else float(va - vA)
                dB = None if (pd.isna(va) or pd.isna(vB)) else float(va - vB)
    
                drows_grouped.append({
                    "Variable": v,
                    "Value": None if pd.isna(va) else float(va),
                    "A":   None if pd.isna(vA) else float(vA),
                    "B":   None if pd.isna(vB) else float(vB),
                    "sA":  None if not np.isfinite(sA) else float(sA),
                    "sB":  None if not np.isfinite(sB) else float(sB),
                    "d_vs_A": None if dA is None else dA,
                    "d_vs_B": None if dB is None else dB,
                    "is_core": (v in var_core),
                })
        detail_map[tkr] = drows_grouped
    
    # ---------------- HTML/JS render ----------------
    import streamlit.components.v1 as components
    
    def _round_rec(o):
        if isinstance(o, dict): return {k:_round_rec(v) for k,v in o.items()}
        if isinstance(o, list): return [_round_rec(v) for v in o]
        if isinstance(o, float): return float(np.round(o, 6))
        return o
    
    payload = _round_rec({"rows": summary_rows, "details": detail_map, "gA": gA, "gB": gB})
    
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
      .blue>span{background:#3b82f6}.red>span{background:#ef4444}.green>span{background:#10b981}.purple>span{background:#8b5cf6}
      #align td:nth-child(2),#align th:nth-child(2),
      #align td:nth-child(3),#align th:nth-child(3),
      #align td:nth-child(4),#align th:nth-child(4),
      #align td:nth-child(5),#align th:nth-child(5){text-align:center}
      .child-table{width:100%;border-collapse:collapse;margin:2px 0 2px 24px;table-layout:fixed}
      .child-table th,.child-table td{font-size:11px;padding:3px 6px;border-bottom:1px solid #e5e7eb;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .child-table th:first-child,.child-table td:first-child{text-align:left}
      tr.group-row td{background:#f3f4f6!important;color:#374151;font-weight:600}
      .col-var{width:18%}.col-val{width:12%}.col-a{width:18%}.col-b{width:18%}.col-da{width:17%}.col-db{width:17%}
      .pos{color:#059669}.neg{color:#dc2626}
      .sig-hi{background:rgba(250,204,21,0.18)!important}
      .sig-lo{background:rgba(239,68,68,0.18)!important}
    
      /* Ensure horizontal scroll with fixed min-width so columns never hide */
      #align { min-width: 1060px; } /* tweak if you change column widths below */
      #align-wrap { overflow:auto; width:100%; }
    </style></head><body>
      <div id="align-wrap">
        <table id="align" class="display nowrap stripe" style="width:100%">
          <thead><tr><th>Ticker</th><th id="hdrA"></th><th id="hdrB"></th><th id="hdrN"></th><th id="hdrC"></th></tr></thead>
        </table>
      </div>
      <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
      <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
      <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
      <script>
        const data = %%PAYLOAD%%;
        document.getElementById('hdrA').textContent = data.gA;
        document.getElementById('hdrB').textContent = data.gB;
        document.getElementById('hdrN').textContent = 'NCA: P(' + data.gA + ')';
        document.getElementById('hdrC').textContent = 'CatBoost: P(' + data.gA + ')';
    
        function barCellLabeled(valRaw,label,valInt,clsOverride){
          const cls=clsOverride || 'blue';
          const w=(valRaw==null||isNaN(valRaw))?0:Math.max(0,Math.min(100,valRaw));
          const text=(valInt==null||isNaN(valInt))?Math.round(w):valInt;
          return `<div class="bar-wrap"><div class="bar ${cls}"><span style="width:${w}%"></span></div><div class="bar-label">${text}</div></div>`;
        }
        function formatVal(x){return (x==null||isNaN(x))?'':Number(x).toFixed(2);}
    
        function childTableHTML(ticker){
          const rows=(data.details||{})[ticker]||[];
          if(!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No variable overlaps for this stock.</div>';
    
          const cells = rows.map(r=>{
            if(r.__group__) return '<tr class="group-row"><td colspan="6">'+r.__group__+'</td></tr>';
    
            const v=formatVal(r.Value), a=formatVal(r.A), b=formatVal(r.B);
    
            const rawDa=(r.d_vs_A==null||isNaN(r.d_vs_A))?null:Number(r.d_vs_A);
            const rawDb=(r.d_vs_B==null||isNaN(r.d_vs_B))?null:Number(r.d_vs_B);
            const da=(rawDa==null)?'':formatVal(Math.abs(rawDa));
            const db=(rawDb==null)?'':formatVal(Math.abs(rawDb));
            const ca=(rawDa==null)?'':(rawDa>=0?'pos':'neg');
            const cb=(rawDb==null)?'':(rawDb>=0?'pos':'neg');
    
            // 3σ significance vs closer center
            const val  = (r.Value==null || isNaN(r.Value)) ? null : Number(r.Value);
            const cA   = (r.A==null || isNaN(r.A)) ? null : Number(r.A);
            const cB   = (r.B==null || isNaN(r.B)) ? null : Number(r.B);
            const sA   = (r.sA==null || isNaN(r.sA)) ? null : Number(r.sA);
            const sB   = (r.sB==null || isNaN(r.sB)) ? null : Number(r.sB);
    
            let sigClass = '';
            if (val!=null && cA!=null && cB!=null) {
              const dAabs = Math.abs(val - cA);
              const dBabs = Math.abs(val - cB);
              const closer = (dAabs <= dBabs) ? 'A' : 'B';
              const center = closer === 'A' ? cA : cB;
              const sigma  = closer === 'A' ? sA : sB;
    
              if (sigma!=null && sigma>0) {
                const z = (val - center) / sigma;
                if (z >= 3)       sigClass = 'sig-hi';
                else if (z <= -3) sigClass = 'sig-lo';
              }
            }
    
            return `<tr class="${sigClass}">
              <td class="col-var">${r.Variable}</td>
              <td class="col-val">${v}</td>
              <td class="col-a">${a}</td>
              <td class="col-b">${b}</td>
              <td class="col-da ${ca}">${da}</td>
              <td class="col-db ${cb}">${db}</td>
            </tr>`;
          }).join('');
    
          return `<table class="child-table">
            <colgroup><col class="col-var"/><col class="col-val"/><col class="col-a"/><col class="col-b"/><col class="col-da"/><col class="col-db"/></colgroup>
            <thead><tr>
              <th class="col-var">Variable</th>
              <th class="col-val">Value</th>
              <th class="col-a">${data.gA} center</th>
              <th class="col-b">${data.gB} center</th>
              <th class="col-da">Δ vs ${data.gA}</th>
              <th class="col-db">Δ vs ${data.gB}</th>
            </tr></thead>
            <tbody>${cells}</tbody></table>`;
        }
    
        $(function(){
          const table=$('#align').DataTable({
            data: data.rows||[],
            paging:false, info:false, searching:false, order:[[0,'asc']],
            responsive:false,   // keep all columns visible
            scrollX:true,       // enable horizontal scroll inside wrapper
            autoWidth:false,    // we set widths manually
            columns:[
              {data:'Ticker', width:'140px'},
              {data:null, render:(row)=>barCellLabeled(row.A_val_raw,row.A_label,row.A_val_int,'blue'),   width:'220px'},
              {data:null, render:(row)=>barCellLabeled(row.B_val_raw,row.B_label,row.B_val_int,'red'),    width:'220px'},
              {data:null, render:(row)=>barCellLabeled(row.NCA_raw,'NCA',row.NCA_int,'green'),            width:'220px'},
              {data:null, render:(row)=>barCellLabeled(row.CAT_raw,'CatBoost',row.CAT_int,'purple'),      width:'220px'}
            ]
          });
          table.columns.adjust().draw();
          $(window).on('resize', ()=>table.columns.adjust());
    
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
    
    # Force refresh of the HTML table when split/rows change
    cache_bust = f"{gA}|{gB}|{int(thr)}|{len(summary_rows)}"
    
    # FIX: Replaced the faulty, complex try-except block with a single, reliable call.
    # This ensures that the scrolling functionality is correctly enabled in modern Streamlit versions.
    components.html(html, height=800, scrolling=True)
    
    
    # ============================== Alignment exports (CSV full + Markdown compact) ==============================
    if summary_rows:
        # ---------- Markdown (compact summary) ----------
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
    
        df_align_md = pd.DataFrame(summary_rows)[
            ["Ticker", "A_label", "A_val_int", "B_label", "B_val_int", "NCA_int", "CAT_int"]
        ].rename(
            columns={
                "A_label": "A group",
                "A_val_int": "A (%) — Median centers",
                "B_label": "B group",
                "B_val_int": "B (%) — Median centers",
                "NCA_int": "NCA (%)",
                "CAT_int": "CatBoost (%)",
            }
        )
    
        # ---------- CSV (full with child rows) ----------
        full_rows = []
        sum_by_ticker = {s["Ticker"]: s for s in summary_rows}
    
        for tkr, rows in detail_map.items():
            s = sum_by_ticker.get(tkr, {})
            section = ""
            for r in rows:
                if r.get("__group__"):
                    section = r["__group__"]
                    full_rows.append({
                        "Ticker": tkr,
                        "Section": section,
                        "Variable": "",
                        "Value": "",
                        "A center": "",
                        "B center": "",
                        "Δ vs A": "",
                        "Δ vs B": "",
                        "σ(A)": "",
                        "σ(B)": "",
                        "Is core": "",
                        "A group": s.get("A_label", ""),
                        "B group": s.get("B_label", ""),
                        "A (%) — Median centers": s.get("A_val_int", ""),
                        "B (%) — Median centers": s.get("B_val_int", ""),
                        "NCA (%)": s.get("NCA_int", ""),
                        "CatBoost (%)": s.get("CAT_int", ""),
                    })
                    continue
    
                full_rows.append({
                    "Ticker": tkr,
                    "Section": section,
                    "Variable": r.get("Variable", ""),
                    "Value": ("" if pd.isna(r.get("Value")) else r.get("Value")),
                    "A center": ("" if pd.isna(r.get("A")) else r.get("A")),
                    "B center": ("" if pd.isna(r.get("B")) else r.get("B")),
                    "Δ vs A": ("" if (r.get("d_vs_A") is None or pd.isna(r.get("d_vs_A"))) else r.get("d_vs_A")),
                    "Δ vs B": ("" if (r.get("d_vs_B") is None or pd.isna(r.get("d_vs_B"))) else r.get("d_vs_B")),
                    "σ(A)": ("" if (r.get("sA") is None or pd.isna(r.get("sA"))) else r.get("sA")),
                    "σ(B)": ("" if (r.get("sB") is None or pd.isna(r.get("sB"))) else r.get("sB")),
                    "Is core": bool(r.get("is_core", False)),
                    "A group": s.get("A_label", ""),
                    "B group": s.get("B_label", ""),
                    "A (%) — Median centers": s.get("A_val_int", ""),
                    "B (%) — Median centers": s.get("B_val_int", ""),
                    "NCA (%)": s.get("NCA_int", ""),
                    "CatBoost (%)": s.get("CAT_int", ""),
                })
    
        df_align_csv_full = pd.DataFrame(full_rows)
    
        def _fmt_num(x, fmt=".2f"):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) or x == "":
                return ""
            try:
                return format(float(x), fmt)
            except Exception:
                return ""
    
        def _fmt_int(x):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) or x == "":
                return ""
            try:
                return f"{int(round(float(x)))}"
            except Exception:
                return ""
    
        # Columns to format
        two_dec_cols = ["Value", "A center", "B center", "Δ vs A", "Δ vs B"]
        sigma_cols   = ["σ(A)", "σ(B)"]
        pct_cols     = ["A (%) — Median centers", "B (%) — Median centers", "NCA (%)", "CatBoost (%)"]
    
        for col in two_dec_cols + sigma_cols:
            if col in df_align_csv_full.columns:
                df_align_csv_full[col] = df_align_csv_full[col].apply(lambda v: _fmt_num(v, ".2f"))
        for col in pct_cols:
            if col in df_align_csv_full.columns:
                df_align_csv_full[col] = df_align_csv_full[col].apply(_fmt_int)
    
        col_order = [
            "Ticker", "Section", "Variable",
            "Value", "A center", "B center", "Δ vs A", "Δ vs B", "σ(A)", "σ(B)", "Is core",
            "A group", "B group",
            "A (%) — Median centers", "B (%) — Median centers", "NCA (%)", "CatBoost (%)",
        ]
        df_align_csv_pretty = df_align_csv_full[[c for c in col_order if c in df_align_csv_full.columns]]
    
        st.markdown("##### Export alignment")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download CSV (full, with child rows)",
                data=df_align_csv_pretty.to_csv(index=False).encode("utf-8"),
                file_name="alignment_full_with_children.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_align_csv_full",
            )
        with c2:
            st.download_button(
                "Download Markdown (summary only)",
                data=_df_to_markdown_simple(df_align_md, float_fmt=".0f").encode("utf-8"),
                file_name="alignment_summary.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_align_md_summary",
            )
    
    # ============================== Radar — centers vs stocks (Matplotlib, with toggles) ==============================
    st.markdown("---")
    st.subheader("Radar")
    
    if not ss.rows:
        st.info("Add at least one stock to see the radar chart.")
    else:
        def _available_live_axes_mpl():
            # Correctly use ALLOWED_LIVE_FEATURES and EXCLUDE_FOR_NCA for axis selection
            axes = [f for f in ALLOWED_LIVE_FEATURES if f in med_tbl.index and f not in EXCLUDE_FOR_NCA]
            order_hint = [
                "Gap_%","FR_x","PM$Vol/MC_%","Catalyst","Max_Pull_PM_%","RVOL_Max_PM_cum",
                "MC_PM_Max_M","Float_PM_Max_M","PM_Vol_M","PM_$Vol_M$","ATR_$"
            ]
            # Keep order while filtering
            ordered_axes = [a for a in order_hint if a in axes]
            # Add any remaining valid axes that weren't in the hint
            remaining_axes = [a for a in axes if a not in ordered_axes]
            return ordered_axes + remaining_axes
    
        def _catboost_topk_features_mpl(model: dict, k: int = 6) -> list[str]:
            if not (model and model.get("ok") and model.get("cb")):
                return []
            try:
                cb = model["cb"]
                feats = model["feats"] # These are already filtered correctly
                imps = np.array(cb.get_feature_importance(), dtype=float)
                order = np.argsort(imps)[::-1]
                # Ensure features are also in the median table to be plottable
                return [feats[i] for i in order[:k] if feats[i] in med_tbl.index]
            except Exception:
                return []
    
        def _norm_minmax_between_centers_mpl(values: dict, a_center: dict, b_center: dict) -> dict:
            out = {}
            eps = 1e-9
            for f, x in values.items():
                a = a_center.get(f, np.nan)
                b = b_center.get(f, np.nan)
                lo = np.nanmin([a, b])
                hi = np.nanmax([a, b])
                span = hi - lo
                if not np.isfinite(span) or span <= eps:
                    try:
                        sA = float(disp_tbl.at[f, gA]) if f in disp_tbl.index else np.nan
                        sB = float(disp_tbl.at[f, gB]) if f in disp_tbl.index else np.nan
                    except Exception:
                        sA, sB = np.nan, np.nan
                    s = np.nanmax([sA, sB])
                    mid = np.nanmean([a, b])
                    if np.isfinite(s) and s > 0:
                        lo, hi = mid - 3*s, mid + 3*s
                        span = hi - lo
                    else:
                        lo, span = 0.0, 1.0
                v = (x - lo) / (span + eps)
                out[f] = float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else np.nan
            return out
    
        axes_all = _available_live_axes_mpl()
        if not axes_all:
            st.info("No live features available for radar plotting with current split.")
        else:
            # ---------- Controls
            c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.1, 1.3])
            with c1:
                feat_mode = st.radio(
                    "Features",
                    ["Core", "All live", "Top-6 CatBoost importances"],
                    index=0,
                    key="radar_feat_mode_mpl",
                    help="Pick which axes to plot.",
                    horizontal=False,
                )
            with c2:
                show_A = st.checkbox(f"Show {gA} center", value=True, key="radar_show_A")
                show_B = st.checkbox(f"Show {gB} center", value=True, key="radar_show_B")
                show_cb = st.checkbox("Show CatBoost polygon", value=True, key="radar_show_cb")
                show_nca = st.checkbox("Show NCA polygon", value=False, key="radar_show_nca")
            with c3:
                # quick selectors for stocks
                all_tickers_radar = [(r.get("Ticker") or "—") for r in ss.rows]
                _seen_radar = set()
                all_tickers_radar = [t for t in all_tickers_radar if not (t in _seen_radar or _seen_radar.add(t))]
                if "radar_stock_sel_mpl" not in st.session_state:
                    st.session_state["radar_stock_sel_mpl"] = all_tickers_radar[:5]
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Select none", key="radar_none"):
                        st.session_state["radar_stock_sel_mpl"] = []
                        do_rerun()
                with col_btn2:
                    if st.button("Select all", key="radar_all"):
                        st.session_state["radar_stock_sel_mpl"] = all_tickers_radar[:]
                        do_rerun()
            with c4:
                stocks_overlay = st.multiselect(
                    "Stocks to overlay",
                    options=all_tickers_radar,
                    default=st.session_state.get("radar_stock_sel_mpl", []),
                    key="radar_stock_sel_mpl",
                    help="Turn individual stock polygons on/off.",
                )
    
            # ---------- Axes selection
            core_live_features = [f for f in VAR_CORE if f in axes_all]
            if feat_mode == "Core":
                axes = core_live_features or axes_all[:6]
            elif feat_mode == "All live":
                axes = axes_all
            else:
                axes = _catboost_topk_features_mpl(ss.get("cat_model", {}), k=6) or core_live_features or axes_all[:6]
    
            # ---------- Centers dicts
            centerA = {f: (float(med_tbl.at[f, gA]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gA])) else np.nan) for f in axes}
            centerB = {f: (float(med_tbl.at[f, gB]) if (f in med_tbl.index and pd.notna(med_tbl.at[f, gB])) else np.nan) for f in axes}
    
            # ---------- Angles
            N = len(axes)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            angles_close = np.concatenate([angles, [angles[0]]])
    
            # ---------- Normalize centers on chosen axes
            normA = _norm_minmax_between_centers_mpl(centerA, centerA, centerB)
            normB = _norm_minmax_between_centers_mpl(centerB, centerA, centerB)
            rA = np.array([normA.get(f, 0.5) for f in axes], dtype=float)
            rB = np.array([normB.get(f, 0.5) for f in axes], dtype=float)
            rA_close = np.concatenate([rA, [rA[0]]])
            rB_close = np.concatenate([rB, [rB[0]]])
    
            # ---------- Figure
            fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6.5, 6.5))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            # ---------- Which group has higher median per feature
            higher_group = {}
            for f in axes:
                try:
                    valA = float(med_tbl.at[f, gA]) if f in med_tbl.index else np.nan
                    valB = float(med_tbl.at[f, gB]) if f in med_tbl.index else np.nan
                    if np.isfinite(valA) and np.isfinite(valB):
                        higher_group[f] = gA if valA > valB else gB
                    else:
                        higher_group[f] = ""
                except Exception:
                    higher_group[f] = ""
    
            # FIX: Resolve SyntaxError by separating string replacement from f-string formatting.
            # This also correctly escapes '$' for Matplotlib labels.
            clean_labels = [f.replace('$', r'\$') + f"\n({higher_group.get(f, '')})↑" for f in axes]
            ax.set_thetagrids(angles * 180/np.pi, labels=clean_labels)
    
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25","0.5","0.75","1.0"])
    
            # ---------- Centers (optional)
            if show_A:
                ax.plot(angles_close, rA_close, color="#3b82f6", linewidth=2, label=f"{gA} center")
            if show_B:
                ax.plot(angles_close, rB_close, color="#ef4444", linewidth=2, label=f"{gB} center")
    
            # ---------- NCA polygon (optional; smarter importances)
            if show_nca and ss.get("nca_model", {}).get("ok"):
                try:
                    ncam = ss["nca_model"]
                    feats_nca = ncam["feats"]
    
                    # Build a clean matrix X and labels y from the current split for these features
                    df2 = df_cmp[df_cmp["__Group__"].isin([gA, gB])].copy()
                    Xdf = df2[feats_nca].apply(pd.to_numeric, errors="coerce")
                    ybin = (df2["__Group__"].values == gA).astype(int)
                    mask = np.isfinite(Xdf.values).all(axis=1)
                    X = Xdf.values[mask]
                    y = ybin[mask]
                    if X.shape[0] >= 20 and np.unique(y).size == 2:
                        # standardize using model's mu/sd so alignment matches training
                        mu = np.array(ncam["mu"], dtype=float)
                        sd = np.array(ncam["sd"], dtype=float); sd[sd==0] = 1.0
                        Xs = (X - mu) / sd
    
                        if ncam["kind"] == "lda" and ncam.get("w_vec") is not None:
                            # LDA: absolute weights as importances
                            w = np.array(ncam["w_vec"], dtype=float)
                            imp_map = {f:0.0 for f in axes}
                            for f, wi in zip(feats_nca, w):
                                if f in imp_map:
                                    imp_map[f] = abs(float(wi))
                        else:
                            # True NCA path: derive proxy importances from abs corr with the 1-D embedding
                            try:
                                from sklearn.neighbors import NeighborhoodComponentsAnalysis
                                # Refit a tiny NCA on Xs,y to get the 1-D z for correlations (cheap; only for display)
                                nca_tmp = NeighborhoodComponentsAnalysis(n_components=1, random_state=0, max_iter=200)
                                z = nca_tmp.fit_transform(Xs, y).ravel()
                                # abs Pearson corr(feature_j, z)
                                imp = []
                                for j, f in enumerate(feats_nca):
                                    xj = Xs[:, j]
                                    # handle constant columns
                                    if np.std(xj) < 1e-9:
                                        imp.append(0.0)
                                    else:
                                        r = np.corrcoef(xj, z)[0, 1]
                                        imp.append(abs(float(r)) if np.isfinite(r) else 0.0)
                                imp_map = {f:0.0 for f in axes}
                                for f, v in zip(feats_nca, imp):
                                    if f in imp_map:
                                        imp_map[f] = v
                            except Exception:
                                # fallback: equal weights
                                imp_map = {f:1.0 for f in axes}
    
                        arr = np.array([imp_map.get(f, 0.0) for f in axes], dtype=float)
                        # normalize to [0,1] for plotting
                        if np.isfinite(arr).any():
                            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
                            norm_imp = (arr - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(arr)
                            rN_close = np.concatenate([norm_imp, [norm_imp[0]]])
                            ax.plot(angles_close, rN_close, color="#10b981", linewidth=2, label="NCA (scaled)")
                            ax.fill(angles_close, rN_close, color="#10b981", alpha=0.18)
                except Exception:
                    pass
    
            # ---------- CatBoost polygon (optional)
            if show_cb and ss.get("cat_model", {}).get("ok"):
                try:
                    cb = ss["cat_model"]["cb"]
                    feats_cb = ss["cat_model"]["feats"]
                    imp_vals = np.array(cb.get_feature_importance(), dtype=float)
                    imp_map = {f:0.0 for f in axes}
                    for f, v in zip(feats_cb, imp_vals):
                        if f in imp_map: imp_map[f] = float(v)
                    arr = np.array([imp_map.get(f, 0.0) for f in axes], dtype=float)
                    if np.isfinite(arr).any():
                        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
                        norm_imp = (arr - mn) / (mx - mn + 1e-9) if mx > mn else np.zeros_like(arr)
                        rC_close = np.concatenate([norm_imp, [norm_imp[0]]])
                        ax.plot(angles_close, rC_close, color="#8b5cf6", linewidth=2, label="CatBoost (scaled)")
                        ax.fill(angles_close, rC_close, color="#8b5cf6", alpha=0.20)
                except Exception:
                    pass
            elif show_cb and not ss.get("cat_model", {}).get("ok"):
                st.caption("CatBoost polygon unavailable for this split.")
    
            # ---------- Stocks (multi-toggle)
            stock_color_cycle = ["#06b6d4", "#6366f1", "#f59e0b", "#10b981", "#a3e635", "#fb7185", "#14b8a6"]
            stock_lookup = { (r.get("Ticker") or "—"): r for r in ss.rows }
    
            for i, tkr in enumerate(stocks_overlay):
                row = stock_lookup.get(tkr)
                if not row:
                    continue
                vals = {}
                for f in axes:
                    v = pd.to_numeric(row.get(f), errors="coerce")
                    vals[f] = float(v) if np.isfinite(v) else np.nan
                normS = _norm_minmax_between_centers_mpl(vals, centerA, centerB)
                rS = np.array([normS.get(f, 0.5) for f in axes], dtype=float)
                rS_close = np.concatenate([rS, [rS[0]]])
                color = stock_color_cycle[i % len(stock_color_cycle)]
                ax.plot(angles_close, rS_close, color=color, linewidth=2, label=tkr)
                ax.fill(angles_close, rS_close, color=color, alpha=0.20)
    
            ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.05), frameon=False)
            fig.tight_layout()
    
            st.pyplot(fig, use_container_width=True)
            st.caption("Normalized per feature using A/B centers (0 = closer to lower center, 1 = closer to higher).")
    
            # ---- Export radar: PNG + standalone HTML (base64) ----
            import base64
    
            radar_png_bytes = None
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                radar_png_bytes = buf.getvalue()
            except Exception:
                radar_png_bytes = None
    
            st.markdown("##### Export radar")
            col_png, col_html = st.columns(2)
    
            with col_png:
                if radar_png_bytes:
                    st.download_button(
                        "Download PNG (radar)",
                        data=radar_png_bytes,
                        file_name=f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key="dl_radar_png",
                    )
                else:
                    st.caption("PNG export failed — could not capture the figure.")
    
            with col_html:
                if radar_png_bytes:
                    b64 = base64.b64encode(radar_png_bytes).decode("ascii")
                    html_doc = f"""<!doctype html>
            <html><head><meta charset="utf-8"><title>Radar</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body {{ margin: 0; padding: 1rem; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
              .wrap {{ display: flex; justify-content: center; }}
              img {{ max-width: 100%; height: auto; }}
            </style>
            </head><body>
              <div class="wrap">
                <img alt="Radar" src="data:image/png;base64,{b64}">
              </div>
            </body></html>"""
                    st.download_button(
                        "Download HTML (radar)",
                        data=html_doc.encode("utf-8"),
                        file_name="radar.html",
                        mime="text/html",
                        use_container_width=True,
                        key="dl_radar_html",
                    )
                else:
                    st.caption("HTML export unavailable — no PNG image to embed.")
    
    # ============================== Distributions across Gain% cutoffs ==============================
    st.markdown("---")
    st.subheader("Distributions")
    
    if not ss.rows:
        st.info("Add at least one stock to see distributions across cutoffs.")
    else:
        # unique tickers, preserve first-seen order
        all_tickers = [(r.get("Ticker") or "—") for r in ss.rows]
        _seen = set()
        all_tickers = [t for t in all_tickers if not (t in _seen or _seen.add(t))]
    
        # --- init state BEFORE widgets ---
        if "dist_stock_sel" not in st.session_state:
            st.session_state["dist_stock_sel"] = all_tickers[:]  # default = all
    
        def _clear_dist_sel():
            st.session_state["dist_stock_sel"] = []
    
        csel1, csel2 = st.columns([4, 1])
    
        with csel2:
            st.button("Clear", use_container_width=True, key="dist_sel_clear", on_click=_clear_dist_sel)
    
        with csel1:
            stocks_selected = st.multiselect(
                "",
                options=all_tickers,
                default=st.session_state["dist_stock_sel"],
                key="dist_stock_sel",
                label_visibility="collapsed",
                help="Which added stocks are used to compute the distribution at each cutoff.",
            )
    
        rows_for_dist = [r for r in ss.rows if (r.get("Ticker") or "—") in stocks_selected]
    
        if not rows_for_dist:
            st.info("No stocks selected.")
        else:
            def _make_split(df_base: pd.DataFrame, thr_val: float, mode_val: str):
                df_tmp = df_base.copy()
                if mode_val == "Gain% vs Rest":
                    df_tmp["__Group__"] = np.where(
                        pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val,
                        f"≥{int(thr_val)}%", "Rest"
                    )
                    gA_, gB_ = f"≥{int(thr_val)}%", "Rest"
                else: # "FT vs Fail (Gain% cutoff on FT=1 only)"
                    a_mask = (df_tmp["FT01"] == 1) & (pd.to_numeric(df_tmp["Max_Push_Daily_%"], errors="coerce") >= thr_val)
                    b_mask = (df_tmp["FT01"] == 0)
                    df_tmp = df_tmp[a_mask | b_mask].copy()
                    df_tmp["__Group__"] = np.where(df_tmp["FT01"] == 1, f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)")
                    gA_, gB_ = f"FT=1 ≥{int(thr_val)}%", "FT=0 (all)"
                return df_tmp, gA_, gB_
    
            def _summaries(df_in: pd.DataFrame, vars_all: list[str], grp_col: str):
                avail = [v for v in vars_all if v in df_in.columns]
                if not avail:
                    return pd.DataFrame(), pd.DataFrame()
                g = df_in.groupby(grp_col, observed=True)[avail]
                med_tbl_ = g.median(numeric_only=True).T
                mad_tbl_ = df_in.groupby(grp_col, observed=True)[avail].apply(lambda gg: gg.apply(_mad_local)).T * 1.4826
                return med_tbl_, mad_tbl_
    
            thr_labels = []
            series_A_med, series_B_med, series_N_med, series_C_med = [], [], [], []
    
            for thr_val in gain_choices:
                df_split, gA2, gB2 = _make_split(base_df, float(thr_val), mode)
                med_tbl2, _ = _summaries(df_split, var_all, "__Group__")
                if med_tbl2.empty or med_tbl2.shape[1] < 2:
                    continue
    
                cols2 = list(med_tbl2.columns)
                if (gA2 in cols2) and (gB2 in cols2):
                    med_tbl2 = med_tbl2[[gA2, gB2]]
                else:
                    top2 = df_split["__Group__"].value_counts().index[:2].tolist()
                    if len(top2) < 2:
                        continue
                    gA2, gB2 = top2[0], top2[1]
                    med_tbl2 = med_tbl2[[gA2, gB2]]
    
                As, Bs, Ns, Cs = [], [], [], []
                nca_model2 = _train_nca_or_lda(df_split, gA2, gB2, var_all) or {}
                cat_model2 = _train_catboost_once(df_split, gA2, gB2, var_all) or {}
    
                for row in rows_for_dist:
                    counts2 = _compute_alignment_counts_weighted(
                        stock_row=row,
                        centers_tbl=med_tbl2,
                        var_core=ss.var_core,
                        var_mod=ss.var_moderate,
                        w_core=1.0, w_mod=0.5, tie_mode="split",
                    )
                    a = counts2.get("A_pct_raw", np.nan) if counts2 else np.nan
                    b = counts2.get("B_pct_raw", np.nan) if counts2 else np.nan
    
                    pA = _nca_predict_proba(row, nca_model2)
                    pC = _cat_predict_proba(row, cat_model2)  # ← use the per-cutoff CatBoost
    
                    Ns.append((float(pA)*100.0) if np.isfinite(pA) else np.nan)
                    Cs.append((float(pC)*100.0) if np.isfinite(pC) else np.nan)
                    As.append(a); Bs.append(b)
    
                thr_labels.append(int(thr_val))
                series_A_med.append(float(np.nanmedian(As)) if len(As) else np.nan)
                series_B_med.append(float(np.nanmedian(Bs)) if len(Bs) else np.nan)
                series_N_med.append(float(np.nanmedian(Ns)) if len(Ns) else np.nan)
                series_C_med.append(float(np.nanmedian(Cs)) if len(Cs) else np.nan)
    
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
                color_range  = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]  # blue, red, green, purple
    
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
    
                # ============================== Distribution chart export (PNG via Matplotlib) ==============================
                png_bytes = None
                try:
                    pivot = df_long.pivot(index="GainCutoff_%", columns="Series", values="Value").sort_index()
                    series_names = list(pivot.columns)
    
                    color_map = {
                        f"{gA} (Median centers)": "#3b82f6",   # blue
                        f"{gB} (Median centers)": "#ef4444",   # red
                        f"NCA: P({gA})": "#10b981",            # green
                        f"CatBoost: P({gA})": "#8b5cf6",       # purple
                    }
                    colors = [color_map.get(s, "#999999") for s in series_names]
    
                    thresholds = pivot.index.tolist()
                    n_groups = len(thresholds)
                    n_series = len(series_names)
                    x = np.arange(n_groups)
                    width = 0.8 / max(n_series, 1)
    
                    fig, ax = plt.subplots(figsize=(max(6, n_groups*0.6), 4))
                    for i, s in enumerate(series_names):
                        vals = pivot[s].values.astype(float)
                        ax.bar(x + i*width - (n_series-1)*width/2, vals, width=width, label=s, color=colors[i])
    
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(t) for t in thresholds])
                    ax.set_ylim(0, 100)
                    ax.set_xlabel("Gain% cutoff")
                    ax.set_ylabel("Median across selected stocks (%)")
                    ax.legend(loc="upper left", frameon=False)
    
                    buf = io.BytesIO()
                    fig.tight_layout()
                    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
                    plt.close(fig)
                    png_bytes = buf.getvalue()
                except Exception:
                    png_bytes = None
    
                # ============================== Distribution chart export (HTML + PNG side-by-side) ==============================
                st.markdown("##### Export distribution chart")
                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    if png_bytes:
                        st.download_button(
                            "Download PNG (distribution)",
                            data=png_bytes,
                            file_name=f"distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True,
                            key="dl_dist_png_matplotlib",
                        )
                    else:
                        st.caption("PNG export fallback failed.")
    
                with dl_c2:
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
                    st.download_button(
                        "Download HTML (interactive distribution)",
                        data=html_tpl.encode("utf-8"),
                        file_name="distribution_chart.html",
                        mime="text/html",
                        use_container_width=True,
                        key="dl_dist_html",
                    )

with tab_season:
    st.subheader("Seasonality")

    base_df = ss.get("base_df", pd.DataFrame()).copy()
    if base_df.empty:
        st.info("Upload DB and click **Build** to enable Seasonality.")
        st.stop()

    # ---- Date prep
    date_col = _pick_date_col(base_df)
    if not date_col:
        st.warning("No Date column detected (e.g., 'Date'). Add one to your DB to use Seasonality.")
        st.stop()

    base_df["__DATE__"] = _coerce_date_col(base_df, date_col)
    base_df = base_df[base_df["__DATE__"].notna()].copy()
    if base_df.empty:
        st.warning("Could not parse any valid dates from the detected date column.")
        st.stop()

    # Derived time keys
    base_df["__Month__"] = base_df["__DATE__"].dt.month
    base_df["__DOW__"]   = base_df["__DATE__"].dt.dayofweek  # 0=Mon
    base_df["__Week__"]  = base_df["__DATE__"].dt.isocalendar().week.astype(int)
    base_df["__Qtr__"]   = base_df["__DATE__"].dt.quarter
    base_df["__Year__"]  = base_df["__DATE__"].dt.year

    # ----- Variable sets (Tier 1, 2, 4)
    VAR_TIER1 = [
        # Liquidity/participation
        "PM Vol (M)","PM $Vol (M)","Daily Vol (M)","Daily $Vol (M)",
        "RVOL Max PM (cum)","RVOL @ BO","RVOL Push Sum (avg)","FR @ BO","Sum Push FR","Sum Pull FR",
        # Momentum & FT
        "FT","Max Push Daily (%)","Max Push (%)","Max Pull (%)","Max Fade Daily (%)",
        # Volatility/Risk
        "Daily ATR","Gap %","Max Pull PM (%)",
        # Size/Float
        "Float PM Max (M)","MC PM Max (M)",
        # Catalyst freq
        "Catalyst",
    ]
    VAR_TIER2 = [
        # Attempts / geometry
        "BO Attempts","FT Attempts","Fail Attempts","PMH Tests pre","PMH Tests post","Legs",
        # Timing
        "Time → FT (min)","Time → Fail (min)","Time BO (min from 9:30)",
        "Time → LOD (min)","Time → HOD (min)","Hold ≥ PMH (min)",
        # Micro structure (shares / $)
        "Vol @ BO (M)","Max Push Vol (M)","Max Pull Vol (M)","Sum Push Vol (M)","Sum Pull Vol (M)",
        "$Vol @ BO (M)","Max Push $Vol (M)","Max Pull $Vol (M)","Sum Push $Vol (M)","Sum Pull $Vol (M)",
        # Percent allocations
        "PM Vol (%)","PM $Vol (%)","Vol @ BO (%)","$Vol @ BO (%)",
        "Max Push Vol (%)","Max Pull Vol (%)","Sum Push Vol (%)","Sum Pull Vol (%)",
        "Max Push $Vol (%)","Max Pull $Vol (%)","Sum Push $Vol (%)","Sum Pull $Vol (%)",
        # MpB%
        "MpB% PM (→ 9:30)","MpB% @ BO","MpB% Day (RTH)",
    ]
    VAR_TIER4 = [
        "RVOL Push Max (avg)","RVOL Pull Max (avg)",
        "RVOL Push Sum (avg)","RVOL Pull Sum (avg)",
        "Max Push FR","Max Pull FR","Sum Push FR","Sum Pull FR","FR @ BO"
    ]

    # Deduplicate & keep order that exists in df
    def _present(vars_list):
        return [v for v in vars_list if v in base_df.columns]
    VARS_ALL = []
    for grp in (VAR_TIER1, VAR_TIER2, VAR_TIER4):
        for v in grp:
            if v in base_df.columns and v not in VARS_ALL:
                VARS_ALL.append(v)

    # ----- Controls
    c1,c2,c3,c4 = st.columns([1.5,1.2,1.4,1.7])
    with c1:
        var_sel = st.selectbox("Variable", options=VARS_ALL, index=0)
    with c2:
        axis_sel = st.selectbox("Seasonality axis", options=["Month","Quarter","Week-of-Year","Day-of-Week","Month×DOW"], index=0)
    with c3:
        agg_sel = st.selectbox("Aggregation", options=["Median","Mean","Binary Rate (if 0/1)"], index=0,
                               help="Binary rate will be used if the variable is 0/1 (e.g., FT, Catalyst).")
    with c4:
        smooth_sel = st.selectbox("Smoothing", options=["None","Savitzky–Golay","LOESS"], index=1)

    c5,c6,c7,c8 = st.columns([1.1,1.1,1.2,1.6])
    with c5:
        log_toggle = st.checkbox("Log scale", value=("Vol" in var_sel or "$Vol" in var_sel or "RVOL" in var_sel or "FR" in var_sel))
    with c6:
        winsor = st.checkbox("Winsorize 1–99%", value=("Vol" in var_sel or "$Vol" in var_sel or "RVOL" in var_sel or "FR" in var_sel))
    with c7:
        show_ci = st.checkbox("Show CI / ribbon", value=True)
    with c8:
        min_n = st.number_input("Min n per bucket", min_value=1, max_value=200, value=12, step=1)

    # Splits / Filters
    with st.expander("Splits & Filters", expanded=False):
        split_opt = st.selectbox("Split by", options=["None","Catalyst","FT"], index=0)
        c9,c10,c11 = st.columns(3)
        with c9:
            fl_min = st.number_input("Float ≥ (M)", value=0.0, step=0.5, format="%.2f")
            fl_max = st.number_input("Float ≤ (M)", value=1e9, step=1.0, format="%.2f")
        with c10:
            gap_min = st.number_input("Gap ≥ (%)", value=0.0, step=5.0, format="%.1f")
            gap_max = st.number_input("Gap ≤ (%)", value=1000.0, step=5.0, format="%.1f")
        with c11:
            atr_min = st.number_input("ATR ≥ ($)", value=0.0, step=0.05, format="%.2f")
            atr_max = st.number_input("ATR ≤ ($)", value=10.0, step=0.1, format="%.2f")

    # Apply filters
    filt = np.ones(len(base_df), dtype=bool)
    if "Float PM Max (M)" in base_df.columns:
        s = pd.to_numeric(base_df["Float PM Max (M)"], errors="coerce")
        filt &= (s >= fl_min) & (s <= fl_max)
    if "Gap %" in base_df.columns:
        s = pd.to_numeric(base_df["Gap %"], errors="coerce")
        filt &= (s >= gap_min) & (s <= gap_max)
    if "Daily ATR" in base_df.columns:
        s = pd.to_numeric(base_df["Daily ATR"], errors="coerce")
        filt &= (s >= atr_min) & (s <= atr_max)

    dfS = base_df.loc[filt].copy()
    if dfS.empty:
        st.info("No rows after filters.")
        st.stop()

    # Determine type
    is_binary = var_sel in ("FT","Catalyst") or (
        pd.to_numeric(dfS[var_sel], errors="coerce").dropna().isin([0,1]).all()
    )
    is_percent = "%" in var_sel
    y_raw = pd.to_numeric(dfS[var_sel], errors="coerce")

    # Optional winsor
    if winsor and not is_binary:
        y_proc = _winsor(y_raw)
    else:
        y_proc = y_raw

    # Build grouping keys
    if axis_sel == "Month":
        key = "__Month__"; order = list(range(1,13)); key_label = "Month"
        dfS["_KEY_"] = dfS[key]
    elif axis_sel == "Quarter":
        key = "__Qtr__"; order = [1,2,3,4]; key_label = "Quarter"
        dfS["_KEY_"] = dfS[key]
    elif axis_sel == "Week-of-Year":
        key = "__Week__"; order = sorted(dfS["__Week__"].unique()); key_label = "ISO Week"
        dfS["_KEY_"] = dfS[key]
    elif axis_sel == "Day-of-Week":
        key = "__DOW__"; order = [0,1,2,3,4,5,6]; key_label = "Day of Week"
        dfS["_KEY_"] = dfS[key]
    else:  # Month×DOW
        key = "__Month__"; key2 = "__DOW__"; key_label = "Month×DOW"
        dfS["_KEY_"] = dfS["__Month__"]
        dfS["_KEY2_"] = dfS["__DOW__"]

    # Split dimension
    split_col = None
    if split_opt in ("Catalyst","FT") and split_opt in dfS.columns:
        split_col = split_opt

    # Aggregate
    def _agg_block(sub: pd.DataFrame):
        suby = pd.to_numeric(sub[var_sel], errors="coerce")
        n = suby.notna().sum()
        if is_binary:
            k = (suby == 1).sum()
            p = k / n if n > 0 else np.nan
            lo, hi = _wilson_ci(k, n) if show_ci else (np.nan, np.nan)
            return pd.Series({"value": p, "n": n, "low": lo, "high": hi})
        else:
            if agg_sel == "Mean":
                val = float(np.nanmean(suby))
            else:
                val = float(np.nanmedian(suby))
            sig = _mad_sigma(suby) if show_ci else np.nan
            return pd.Series({"value": val, "n": n, "sigma": sig})

    if axis_sel != "Month×DOW":
        grp_cols = ["_KEY_"] + ([split_col] if split_col else [])
        g = dfS.groupby(grp_cols, observed=True)
        agg = g.apply(_agg_block).reset_index()
        # enforce order
        cat_type = pd.CategoricalDtype(categories=order, ordered=True)
        agg["_KEY_"] = agg["_KEY_"].astype(cat_type)
        agg = agg.sort_values(["_KEY_"] + ([split_col] if split_col else []))
    else:
        grp_cols = ["_KEY_","_KEY2_"] + ([split_col] if split_col else [])
        g = dfS.groupby(grp_cols, observed=True)
        agg = g.apply(_agg_block).reset_index()

    # Plot
    import altair as alt
    alt.data_transformers.disable_max_rows()

    title = f"{var_sel} — {axis_sel}"
    if axis_sel == "Month×DOW":
        # Heatmap of value
        if is_binary:
            scale = alt.Scale(domain=[0,1], clamp=True)
            color_title = "Rate"
        else:
            scale = alt.Scale(scheme="blues")
            color_title = "Value"
        base = alt.Chart(agg).transform_calculate(
            Month="datum._KEY_", DOW="datum._KEY2_"
        )
        heat = base.mark_rect().encode(
            x=alt.X("Month:O", title="Month"),
            y=alt.Y("DOW:O", title="Day of Week"),
            color=alt.Color("value:Q", title=color_title, scale=scale),
            tooltip=["Month:O","DOW:O", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("n:Q", format="d")]
        ).properties(height=280)
        if split_col:
            heat = heat.facet(column=alt.Column(f"{split_col}:N", title=split_col))
        st.altair_chart(heat, use_container_width=True)

    else:
        # 1D seasonal series (bars for binary, line for numeric)
        x_field = alt.X("_KEY_:O", title=key_label)
        color = alt.Color(f"{split_col}:N", title="", legend=alt.Legend(orient="top")) if split_col else alt.value("#3b82f6")

        if is_binary:
            # Bars with Wilson CI
            if split_col:
                chart = alt.Chart(agg).mark_bar().encode(
                    x=x_field, y=alt.Y("value:Q", title="Rate"),
                    color=color,
                    tooltip=[alt.Tooltip("_KEY_:O", title=key_label),
                             f"{split_col}:N",
                             alt.Tooltip("value:Q", title="Rate", format=".3f"),
                             alt.Tooltip("n:Q", format="d")]
                )
                if show_ci:
                    ci = alt.Chart(agg).mark_rule().encode(
                        x=x_field, color=color,
                        y=alt.Y("low:Q"), y2="high:Q",
                        tooltip=[alt.Tooltip("low:Q", format=".3f"), alt.Tooltip("high:Q", format=".3f")]
                    )
                    st.altair_chart((chart + ci).properties(height=320), use_container_width=True)
                else:
                    st.altair_chart(chart.properties(height=320), use_container_width=True)
            else:
                chart = alt.Chart(agg).mark_bar(color="#3b82f6").encode(
                    x=x_field, y=alt.Y("value:Q", title="Rate"),
                    tooltip=[alt.Tooltip("_KEY_:O", title=key_label),
                             alt.Tooltip("value:Q", title="Rate", format=".3f"),
                             alt.Tooltip("n:Q", format="d")]
                )
                ci = alt.Chart(agg).mark_rule(color="#111827").encode(
                    x=x_field, y=alt.Y("low:Q"), y2="high:Q"
                ) if show_ci else None
                st.altair_chart((chart + (ci or alt.Layer())).properties(height=320), use_container_width=True)

        else:
            # Line with optional smoothing and MADσ ribbon
            df_line = agg.dropna(subset=["value"]).copy()
            if df_line.empty:
                st.info("No numeric values to plot after filtering.")
            else:
                # smoothing
                x_ord = np.arange(len(df_line))
                y_vals = df_line["value"].values.astype(float)
                y_smooth = _smooth_series(x_ord, y_vals, smooth_sel)

                df_line["_smooth_"] = y_smooth
                baseL = alt.Chart(df_line)
                line_raw = baseL.mark_line(strokeDash=[4,3], opacity=0.7).encode(
                    x=x_field, y=alt.Y("value:Q", title=var_sel), color=color,
                    tooltip=[alt.Tooltip("_KEY_:O", title=key_label), alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("n:Q", format="d")]
                )
                line_s = baseL.mark_line().encode(
                    x=x_field, y=alt.Y("_smooth_:Q", title=var_sel), color=color,
                    tooltip=[alt.Tooltip("_KEY_:O", title=key_label), alt.Tooltip("_smooth_:Q", title="Smoothed", format=".3f"), alt.Tooltip("n:Q", format="d")]
                )

                layers = line_raw + line_s

                if show_ci and "sigma" in df_line.columns and df_line["sigma"].notna().any():
                    df_line["low"]  = df_line["value"] - df_line["sigma"]
                    df_line["high"] = df_line["value"] + df_line["sigma"]
                    band = alt.Chart(df_line).mark_area(opacity=0.15).encode(
                        x=x_field,
                        y=alt.Y("low:Q"),
                        y2=alt.Y2("high:Q"),
                        color=color
                    )
                    layers = band + layers

                if log_toggle:
                    layers = layers.encode(y=alt.Y("_smooth_:Q", scale=alt.Scale(type="log"), title=var_sel))

                st.altair_chart(layers.properties(height=340), use_container_width=True)

    # ====== Export table for the current view
    st.markdown("##### Export current seasonality table")
    # Normalize numeric display for export
    out = agg.copy()
    if is_binary:
        out["Rate_%"] = (out["value"] * 100.0).round(2)
        out["Low_%"]  = (out["low"] * 100.0).round(2)
        out["High_%"] = (out["high"] * 100.0).round(2)
        cols = [c for c in ["_KEY_", split_col, "Rate_%", "Low_%", "High_%", "n"] if c in out.columns]
        out = out[cols].rename(columns={"_KEY_": key_label})
    else:
        out["Value"] = out["value"].round(4)
        if "sigma" in out.columns: out["MAD_sigma"] = out["sigma"].round(4)
        cols = [c for c in ["_KEY_", split_col, "Value", "MAD_sigma", "n"] if c in out.columns]
        out = out[cols].rename(columns={"_KEY_": key_label})

    csv_bytes = out.to_csv(index=False).encode("utf-8")

    def _to_md(df: pd.DataFrame):
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep  = "| " + " | ".join(["---"]*len(cols)) + " |"
        lines = [head, sep]
        for _, r in df.iterrows():
            vals = [str(v) for v in r.tolist()]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines).encode("utf-8")

    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download CSV", data=csv_bytes, file_name="seasonality_view.csv", mime="text/csv", use_container_width=True)
    with colB:
        st.download_button("Download Markdown", data=_to_md(out), file_name="seasonality_view.md", mime="text/markdown", use_container_width=True)
