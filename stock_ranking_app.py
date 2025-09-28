# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json

# ============================== Page ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ============================== Helpers ==============================
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("%","").replace("$","").replace("(","").replace(")","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    cols_lc = {c: c.strip().lower() for c in cols}

    # 1) RAW case-insensitive exact match (preserves $)
    for cand in candidates:
        lc = cand.strip().lower()
        for c in cols:
            if cols_lc[c] == lc:
                return c

    # 2) Normalized exact
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c

    # 3) Normalized contains
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]:
                return c
    return None

def _to_float(s):
    if pd.isna(s): return np.nan
    try:
        ss = str(s).strip().replace(" ", "")
        if "," in ss and "." not in ss:  # EU decimals
            ss = ss.replace(",", ".")
        else:
            ss = ss.replace(",", "")
        return float(ss)
    except Exception:
        return np.nan

def _mad(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []      # manual rows ONLY
if "last" not in st.session_state: st.session_state.last = {}      # last manual row
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl":..., "mad_tbl":..., "var_core":[...], "var_moderate":[...]}
if "lassoA" not in st.session_state: st.session_state.lassoA = {}  # {"features":[...], "coef": array, "intercept": float, "eps":1e-6}

# ============================== Core & Moderate sets ==============================
# Core: add PM_Vol_% (percent of predicted daily volume).
VAR_CORE = ["Gap_%", "RVOL", "FR_x", "PM$Vol/MC_%", "Catalyst", "PM_Vol_%"]
# Moderate: keep PM_Vol_M and include Daily_Vol_M (for medians only; hidden in child row)
VAR_MODERATE = ["MarketCap_M$", "Float_M", "PM_Vol_M", "PM_$Vol_M$", "ATR_$", "Daily_Vol_M"]
VAR_ALL = VAR_CORE + VAR_MODERATE

# ============================== Simple LASSO (coordinate descent) ==============================
def _standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

def _lasso_cd(X, y, lam=0.05, max_iter=200, tol=1e-6):
    # X, y: standardized (no NaNs)
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            r_j = y - (X @ w) + X[:, j] * w[j]
            rho = np.dot(X[:, j], r_j) / n
            if rho < -lam/2:
                w[j] = (rho + lam/2)
            elif rho > lam/2:
                w[j] = (rho - lam/2)
            else:
                w[j] = 0.0
        if np.linalg.norm(w - w_old, ord=2) < tol:
            break
    return w

def train_lasso_on_db(df: pd.DataFrame) -> dict:
    """
    Train LASSO -> OLS on DB rows to predict ln(Daily_Vol_M).
    Features: ln_mcap, ln_atr, ln_pm, ln_pm_dol, ln_FR, catalyst (0/1)
    """
    eps = 1e-6
    need = {"MarketCap_M$","ATR_$","PM_Vol_M","PM_$Vol_M$","FR_x","Catalyst","Daily_Vol_M"}
    if not need.issubset(df.columns):
        return {}

    # Feature engineering (log)
    ln_mcap   = np.log(np.clip(df["MarketCap_M$"].astype(float).values, eps, None))
    ln_atr    = np.log(np.clip(df["ATR_$"].astype(float).values, eps, None))
    ln_pm     = np.log(np.clip(df["PM_Vol_M"].astype(float).values, eps, None))
    ln_pm_dol = np.log(np.clip(df["PM_$Vol_M$"].astype(float).values, eps, None))
    ln_fr     = np.log(np.clip(df["FR_x"].astype(float).values, eps, None))
    catalyst  = np.clip(pd.to_numeric(df["Catalyst"], errors="coerce").fillna(0).values, 0, 1)

    y_ln = np.log(np.clip(pd.to_numeric(df["Daily_Vol_M"], errors="coerce").values, eps, None))

    feats = np.column_stack([ln_mcap, ln_atr, ln_pm, ln_pm_dol, ln_fr, catalyst])
    mask = np.isfinite(feats).all(axis=1) & np.isfinite(y_ln)
    X = feats[mask]
    y = y_ln[mask]
    if X.shape[0] < 25:
        return {}

    Xs, mu, sd = _standardize(X)
    w_l1 = _lasso_cd(Xs, y, lam=0.05, max_iter=500)

    sel = np.flatnonzero(np.abs(w_l1) > 1e-8)
    if sel.size == 0:
        return {}

    # Refit OLS on selected (still standardized)
    Xs_sel = Xs[:, sel]
    beta_std, *_ = np.linalg.lstsq(Xs_sel, y, rcond=None)

    # Map back to original scale
    # y = b0 + sum_j beta_j * (x_j - mu_j)/sd_j  => intercept:
    b = np.zeros(X.shape[1])
    b[sel] = beta_std
    intercept = float(np.mean(y) - np.sum(b * (-mu / sd)))  # alternative robust intercept
    # Better compute exact intercept from means
    intercept = float(np.mean(y) - np.dot(b, (-mu / sd)))

    features = ["ln_mcap","ln_atr","ln_pm","ln_pm_dol","ln_fr","catalyst"]
    return {"features": features, "coef": b, "intercept": intercept, "mu": mu, "sd": sd, "sel_idx": sel, "eps": eps}

def predict_predvol_m(row: dict, lasso: dict) -> float:
    """Predict PredVol_M for one manual row using trained LASSO→OLS (if available)."""
    if not lasso or not lasso.get("coef") is not None:
        return np.nan
    eps = float(lasso.get("eps", 1e-6))
    # Build feature vector in the same order
    mc = float(row.get("MarketCap_M$", np.nan))
    atr= float(row.get("ATR_$", np.nan))
    pm = float(row.get("PM_Vol_M", np.nan))
    pmd= float(row.get("PM_$Vol_M$", np.nan))
    fr = float(row.get("FR_x", np.nan))
    cat= float(row.get("Catalyst", 0.0))

    ln_mcap   = np.log(np.clip(mc,  eps, None)) if np.isfinite(mc)  else np.nan
    ln_atr    = np.log(np.clip(atr, eps, None)) if np.isfinite(atr) else np.nan
    ln_pm     = np.log(np.clip(pm,  eps, None)) if np.isfinite(pm)  else np.nan
    ln_pm_dol = np.log(np.clip(pmd, eps, None)) if np.isfinite(pmd) else np.nan
    ln_fr     = np.log(np.clip(fr,  eps, None)) if np.isfinite(fr)  else np.nan
    catalyst  = 1.0 if (str(row.get("CatalystYN","No")).lower()=="yes" or float(row.get("Catalyst",0))>=0.5) else 0.0

    x = np.array([ln_mcap, ln_atr, ln_pm, ln_pm_dol, ln_fr, catalyst], dtype=float)
    if not np.isfinite(x).all():
        return np.nan

    # Project to standardized space that the OLS was fit on
    sel = lasso["sel_idx"]
    mu, sd = lasso["mu"], lasso["sd"]
    x_std = (x - mu) / sd
    # keep only selected columns in coef (others are zero)
    b = lasso["coef"]
    yhat_ln = float(lasso["intercept"] + np.dot(b, x_std))
    pred = np.exp(yhat_ln)
    if not np.isfinite(pred):
        return np.nan
    return max(pred, 0.0)

# ============================== Upload DB → Build Medians & Train LASSO ==============================
st.subheader("Upload Database")

uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

if build_btn:
    if not uploaded:
        st.error("Please upload an Excel workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            # choose first non-legend sheet
            sheet_candidates = [s for s in xls.sheet_names if _norm(s) not in {"legend","readme"}]
            sheet = sheet_candidates[0] if sheet_candidates else xls.sheet_names[0]
            raw = pd.read_excel(xls, sheet)

            # --- auto-detect group column ---
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
                # fallback: any binary-like col
                for c in raw.columns:
                    vals = pd.Series(raw[c]).dropna().astype(str).str.lower()
                    if vals.isin(["0","1","true","false","yes","no"]).all():
                        col_group = c
                        break

            if col_group is None:
                st.error("Could not detect FT column (0/1). Please ensure your sheet has an FT or binary column.")
            else:
                df = pd.DataFrame()
                df["GroupRaw"] = raw[col_group]

                def add_num(dfout, name, src_candidates):
                    src = _pick(raw, src_candidates)
                    if src:
                        dfout[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                # Map numeric columns
                add_num(df, "MarketCap_M$", ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap_m"])
                add_num(df, "Float_M",      ["float m","public float (m)","float_m","float (m)","float m shares","float_m_shares"])
                add_num(df, "Gap_%",        ["gap %","gap%","premarket gap","gap","gap_percent"])
                add_num(df, "ATR_$",        ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
                add_num(df, "RVOL",         ["rvol","relative volume","rvol @ bo","rvol at bo","rvol_bo"])
                add_num(df, "PM_Vol_M",     ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol_m"])
                add_num(df, "PM_$Vol_M$",   ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm_dollarvol_m","pm $vol"])
                add_num(df, "PM_Vol_%",     ["pm vol (%)","pm_vol_%","pm vol percent","pm volume (%)","pm_vol_percent"])
                add_num(df, "Daily_Vol_M",  ["daily vol (m)","daily_vol_m","day volume (m)","dvol_m"])

                # --- Catalyst (binary: Yes/No/1/0/True/False) from DB, if present ---
                cand_catalyst = _pick(raw, ["catalyst","catalyst?","has catalyst","news catalyst","catalyst_yn","cat"])
                def _to_binary_local(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1.0
                    if sv in {"0","false","no","n","f"}: return 0.0
                    try:
                        fv = float(sv)
                        return 1.0 if fv >= 0.5 else 0.0
                    except:
                        return np.nan
                if cand_catalyst:
                    df["Catalyst"] = raw[cand_catalyst].map(_to_binary_local)

                # Derived metrics
                if {"PM_Vol_M","Float_M"}.issubset(df.columns):
                    df["FR_x"] = (df["PM_Vol_M"] / df["Float_M"]).replace([np.inf,-np.inf], np.nan)
                if {"PM_$Vol_M$","MarketCap_M$"}.issubset(df.columns):
                    df["PM$Vol/MC_%"] = (df["PM_$Vol_M$"] / df["MarketCap_M$"] * 100.0).replace([np.inf,-np.inf], np.nan)

                # IMPORTANT: normalize percent-like fields stored as fractions
                if "Gap_%" in df.columns:
                    df["Gap_%"] = pd.to_numeric(df["Gap_%"], errors="coerce") * 100.0
                if "PM_Vol_%" in df.columns:
                    df["PM_Vol_%"] = pd.to_numeric(df["PM_Vol_%"], errors="coerce") * 100.0

                # Normalize to binary for FT groups
                def _to_binary(v):
                    sv = str(v).strip().lower()
                    if sv in {"1","true","yes","y","t"}: return 1
                    if sv in {"0","false","no","n","f"}: return 0
                    try:
                        fv = float(sv)
                        return 1 if fv >= 0.5 else 0
                    except:
                        return np.nan

                df["FT01"] = df["GroupRaw"].map(_to_binary)
                df = df[df["FT01"].isin([0,1])]
                if df.empty or df["FT01"].nunique() < 2:
                    st.error("Could not find both FT=1 and FT=0 rows in the DB. Please check the group column.")
                else:
                    df["Group"] = df["FT01"].map({1:"FT=1", 0:"FT=0"})

                    # Limit medians/MADs to available vars
                    var_core = [v for v in VAR_CORE if v in df.columns]
                    var_mod  = [v for v in VAR_MODERATE if v in df.columns]
                    var_all  = var_core + var_mod

                    # medians per group
                    gmed = df.groupby("Group")[var_all].median(numeric_only=True).T  # rows=variables, cols=FT=0/FT=1
                    # robust spread: MAD per group
                    gmads = df.groupby("Group")[var_all].apply(lambda g: g.apply(_mad)).T

                    st.session_state.models = {
                        "models_tbl": gmed,
                        "mad_tbl": gmads,
                        "var_core": var_core,
                        "var_moderate": var_mod
                    }

                    # ===== Train LASSO→OLS on DB (Daily_Vol_M) ONLY to use for manual predictions =====
                    # Use only rows that have all needed for training
                    lasso_model = train_lasso_on_db(df)
                    st.session_state.lassoA = lasso_model or {}

                    st.success(f"Built medians and trained PredVol model. Medians columns = {list(gmed.columns)}")
                    do_rerun()

        except Exception as e:
            st.error("Loading/processing failed.")
            st.exception(e)

# ============================== Medians tables (grouped) ==============================
models_data = st.session_state.models
if models_data and isinstance(models_data, dict) and not models_data.get("models_tbl", pd.DataFrame()).empty:
    with st.expander("Model Medians (FT=1 vs FT=0) — grouped", expanded=False):
        med_tbl: pd.DataFrame = models_data["models_tbl"]
        mad_tbl: pd.DataFrame = models_data.get("mad_tbl", pd.DataFrame())
        var_core = models_data.get("var_core", [])
        var_mod  = models_data.get("var_moderate", [])

        # User control for what we call "significant"
        sig_thresh = st.slider("Significance threshold (σ)", 0.0, 5.0, 3.0, 0.1,
                               help="Highlight rows where |FT=1 − FT=0| / (MAD₁ + MAD₀) ≥ σ")
        st.session_state["sig_thresh"] = float(sig_thresh)

        def show_grouped_table(title, vars_list):
            if not vars_list:
                st.info(f"No variables available for {title}.")
                return
            sub_med = med_tbl.loc[[v for v in vars_list if v in med_tbl.index]].copy()
            if not mad_tbl.empty and {"FT=1","FT=0"}.issubset(mad_tbl.columns):
                eps = 1e-9
                diff = (sub_med["FT=1"] - sub_med["FT=0"]).abs()
                spread = (mad_tbl.loc[diff.index, "FT=1"].fillna(0.0) + mad_tbl.loc[diff.index, "FT=0"].fillna(0.0))
                sig = diff / (spread.replace(0.0, np.nan) + eps)
                sig_flag = sig >= st.session_state["sig_thresh"]

                def _style_sig(col: pd.Series):
                    return ["background-color: #fde68a; font-weight: 600;" if sig_flag.get(idx, False) else "" 
                            for idx in col.index]

                st.markdown(f"**{title}**")
                styled = (sub_med
                          .style
                          .apply(_style_sig, subset=["FT=1"])
                          .apply(_style_sig, subset=["FT=0"])
                          .format("{:.2f}"))
                st.dataframe(styled, use_container_width=True)
            else:
                st.markdown(f"**{title}**")
                cfg = {
                    "FT=1": st.column_config.NumberColumn("FT=1 (median)", format="%.2f"),
                    "FT=0": st.column_config.NumberColumn("FT=0 (median)", format="%.2f"),
                }
                st.dataframe(sub_med, use_container_width=True, column_config=cfg, hide_index=False)

        show_grouped_table("Core variables", var_core)
        show_grouped_table("Moderate variables", var_mod)

# ============================== ➕ Manual Input ==============================
st.markdown("---")
st.subheader("Add Stock")

with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

    with c1:
        ticker  = st.text_input("Ticker", "").strip().upper()
        mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
        float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
        gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f")  # real percent e.g. 100 = +100%

    with c2:
        atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
        rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")
        pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
        pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")

    with c3:
        catalyst_yn = st.selectbox("Catalyst?", ["No", "Yes"], index=0)

    submitted = st.form_submit_button("Add to Table", use_container_width=True)

if submitted and ticker:
    # Derived metrics
    fr = (pm_vol / float_m) if float_m > 0 else 0.0
    pmmc = (pm_dol / mc_m * 100.0) if mc_m > 0 else 0.0

    row = {
        "Ticker": ticker,
        "MarketCap_M$": mc_m,
        "Float_M": float_m,
        "Gap_%": gap_pct,         # manual already percent (no scaling)
        "ATR_$": atr_usd,
        "RVOL": rvol,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,
        "FR_x": fr,
        "PM$Vol/MC_%": pmmc,
        "CatalystYN": catalyst_yn,
        "Catalyst": 1.0 if catalyst_yn == "Yes" else 0.0,
    }

    # Predict Daily Volume (M) for this stock (using LASSO→OLS trained on DB)
    pred = predict_predvol_m(row, st.session_state.get("lassoA", {}))
    row["PredVol_M"] = float(pred) if np.isfinite(pred) else np.nan

    # Compute PM_Vol_% = PM_Vol_M / PredVol_M × 100 (if PredVol_M available)
    if np.isfinite(row.get("PredVol_M", np.nan)) and row["PredVol_M"] > 0:
        row["PM_Vol_%"] = (row["PM_Vol_M"] / row["PredVol_M"]) * 100.0
    else:
        row["PM_Vol_%"] = np.nan

    st.session_state.rows.append(row)
    st.session_state.last = row
    st.success(f"Saved {ticker}.")
    do_rerun()

# ============================== Toolbar: Delete + Multiselect (same row) ==============================
tickers = [r.get("Ticker","").strip().upper() for r in st.session_state.rows if r.get("Ticker","")]
tcol_btn, tcol_sel = st.columns([1, 3])
with tcol_btn:
    if st.button("Delete selected", use_container_width=True, type="primary"):
        sel = st.session_state.get("to_delete", [])
        if not sel:
            st.info("No rows selected.")
        else:
            before = len(st.session_state.rows)
            sel_set = set([s.strip().upper() for s in sel])
            st.session_state.rows = [
                r for r in st.session_state.rows
                if str(r.get("Ticker","")).strip().upper() not in sel_set
            ]
            removed = before - len(st.session_state.rows)
            if removed > 0:
                st.success(f"Removed {removed} row(s): {', '.join(sorted(sel_set))}")
            else:
                st.info("No rows removed.")
            do_rerun()
with tcol_sel:
    st.multiselect(
        label="",
        options=tickers,
        default=[],
        key="to_delete",
        placeholder="Select tickers to delete…",
        label_visibility="collapsed"
    )

# ============================== Alignment (DataTables child-rows) ==============================
st.markdown("### Alignment")

def _compute_alignment_counts_core(stock_row: dict, models_tbl: pd.DataFrame, var_core: list[str]) -> dict:
    if models_tbl is None or models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns):
        return {}
    groups = ["FT=1","FT=0"]
    common = [v for v in var_core if (v in stock_row) and (v in models_tbl.index)]
    counts = {g: 0 for g in groups}; used = 0
    for v in common:
        xv = pd.to_numeric(stock_row.get(v), errors="coerce")
        if not np.isfinite(xv):
            continue
        med = models_tbl.loc[v, groups].astype(float).dropna()
        if med.empty:
            continue
        nearest = (med - xv).abs().idxmin()
        counts[nearest] += 1
        used += 1
    counts["N_Vars_Used"] = used
    return counts

models_tbl = (st.session_state.get("models") or {}).get("models_tbl", pd.DataFrame())
mad_tbl = (st.session_state.get("models") or {}).get("mad_tbl", pd.DataFrame())
var_core = (st.session_state.get("models") or {}).get("var_core", [])
var_mod  = (st.session_state.get("models") or {}).get("var_moderate", [])
SIG_THR = float(st.session_state.get("sig_thresh", 2.0))

if st.session_state.rows and not models_tbl.empty and {"FT=1","FT=0"}.issubset(models_tbl.columns):
    # Build summary using CORE variables only
    summary_rows, detail_map = [], {}

    # Order for details: core group first, then moderate (including PredVol_M proxying Daily_Vol_M medians)
    detail_order = [("Core variables", var_core), ("Moderate variables", var_mod + (["PredVol_M"] if "PredVol_M" not in var_mod else []))]

    for row in st.session_state.rows:
        stock = dict(row)
        tkr = stock.get("Ticker") or "—"
        counts = _compute_alignment_counts_core(stock, models_tbl, var_core)
        if not counts:
            continue

        like1, like0 = counts.get("FT=1",0), counts.get("FT=0",0)
        n_used = counts.get("N_Vars_Used",0)
        ft1_val = round((like1 / n_used * 100.0), 0) if n_used > 0 else 0.0
        ft0_val = round((like0 / n_used * 100.0), 0) if n_used > 0 else 0.0

        summary_rows.append({"Ticker": tkr, "FT1_val": ft1_val, "FT0_val": ft0_val})

        # ---- child details: group headers + rows ----
        drows_grouped = []
        for grp_label, grp_vars in detail_order:
            drows_grouped.append({"__group__": grp_label})
            for v in grp_vars:
                # Hide Daily_Vol_M row in the child table
                if v == "Daily_Vol_M":
                    continue

                va = pd.to_numeric(stock.get(v), errors="coerce")

                # Map PredVol_M medians to Daily_Vol_M medians (for FT columns + deltas)
                med_var = "Daily_Vol_M" if v == "PredVol_M" else v

                v1 = models_tbl.loc[med_var, "FT=1"] if (med_var in models_tbl.index) else np.nan
                v0 = models_tbl.loc[med_var, "FT=0"] if (med_var in models_tbl.index) else np.nan
                m1 = mad_tbl.loc[med_var, "FT=1"] if (not mad_tbl.empty and med_var in mad_tbl.index and "FT=1" in mad_tbl.columns) else np.nan
                m0 = mad_tbl.loc[med_var, "FT=0"] if (not mad_tbl.empty and med_var in mad_tbl.index and "FT=0" in mad_tbl.columns) else np.nan

                # Always render PredVol_M row; others keep the normal skip rule
                if v != "PredVol_M" and pd.isna(va) and pd.isna(v1) and pd.isna(v0):
                    continue

                def _sig(delta, mad):
                    if pd.isna(delta): return np.nan
                    if pd.isna(mad):   return np.nan
                    if mad == 0:
                        return np.inf if abs(delta) > 0 else 0.0
                    return abs(delta) / abs(mad)

                d1 = None if (pd.isna(va) or pd.isna(v1)) else float(va - v1)
                d0 = None if (pd.isna(va) or pd.isna(v0)) else float(va - v0)
                s1 = _sig(d1, float(m1) if pd.notna(m1) else np.nan)
                s0 = _sig(d0, float(m0) if pd.notna(m0) else np.nan)

                # Significance only for CORE vars; MODERATE -> gray background
                is_core = v in var_core
                sig1 = (not pd.isna(s1)) and (s1 >= SIG_THR) if is_core else False
                sig0 = (not pd.isna(s0)) and (s0 >= SIG_THR) if is_core else False

                drows_grouped.append({
                    "Variable": v,
                    "Value": None if pd.isna(va) else float(va),
                    "FT1":   None if pd.isna(v1) else float(v1),
                    "FT0":   None if pd.isna(v0) else float(v0),
                    "d_vs_FT1": None if d1 is None else d1,
                    "d_vs_FT0": None if d0 is None else d0,
                    "sig1": sig1,
                    "sig0": sig0,
                    "is_core": is_core
                })

        detail_map[tkr] = drows_grouped

    if summary_rows:
        import streamlit.components.v1 as components
        payload = {
            "rows": summary_rows,
            "details": detail_map
        }

        html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.responsive.min.css"/>
<style>
  body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  table.dataTable tbody tr { cursor: pointer; }

  /* Parent bars: centered look */
  .bar-wrap { display:flex; justify-content:center; align-items:center; gap:6px; }
  .bar { height: 12px; width: 120px; border-radius: 8px; background: #eee; position: relative; overflow: hidden; }
  .bar > span { position: absolute; left: 0; top: 0; bottom: 0; width: 0%; }
  .bar-label { font-size: 11px; white-space: nowrap; color:#374151; min-width: 24px; text-align:center; }
  .blue > span { background:#3b82f6; }  /* FT=1 = blue */
  .red  > span { background:#ef4444; }  /* FT=0 = red  */

  /* Align FT columns */
  #align td:nth-child(2), #align th:nth-child(2),
  #align td:nth-child(3), #align th:nth-child(3) { text-align: center; }

  /* Child table */
  .child-table { width: 100%; border-collapse: collapse; margin: 2px 0 2px 24px; table-layout: fixed; }
  .child-table th, .child-table td {
    font-size: 11px; padding: 3px 6px; border-bottom: 1px solid #e5e7eb;
    text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .child-table th:first-child, .child-table td:first-child { text-align:left; }

  /* Group header row */
  tr.group-row td {
    background: #f3f4f6 !important; color:#374151; font-weight:600; text-transform:uppercase; letter-spacing:.02em;
    border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb;
  }

  /* Moderate rows get light gray background */
  tr.moderate td { background: #f9fafb !important; }

  /* Significance for CORE rows only (directional) */
  tr.sig_up td   { background: rgba(253, 230, 138, 0.9) !important; }  /* yellow-ish */
  tr.sig_down td { background: rgba(254, 202, 202, 0.9) !important; }  /* red-ish */

  /* Column widths for child table */
  .col-var { width: 18%; }
  .col-val { width: 12%; }
  .col-ft1 { width: 18%; }
  .col-ft0 { width: 18%; }
  .col-d1  { width: 17%; }
  .col-d0  { width: 17%; }

  .pos { color:#059669; } 
  .neg { color:#dc2626; }
</style>
</head>
<body>
  <table id="align" class="display nowrap stripe" style="width:100%">
    <thead>
      <tr>
        <th>Ticker</th>
        <th>FT=1</th>
        <th>FT=0</th>
      </tr>
    </thead>
  </table>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
  <script>
    const data = %%PAYLOAD%%;

    function barCellBlue(val) {
      const v = (val==null||isNaN(val)) ? 0 : Math.max(0, Math.min(100, val));
      return `
        <div class="bar-wrap">
          <div class="bar blue"><span style="width:${v}%"></span></div>
          <div class="bar-label">${v.toFixed(0)}</div>
        </div>`;
    }
    function barCellRed(val) {
      const v = (val==null||isNaN(val)) ? 0 : Math.max(0, Math.min(100, val));
      return `
        <div class="bar-wrap">
          <div class="bar red"><span style="width:${v}%"></span></div>
          <div class="bar-label">${v.toFixed(0)}</div>
        </div>`;
    }

    function formatVal(x){ return (x==null || isNaN(x)) ? '' : Number(x).toFixed(2); }

    function childTableHTML(ticker) {
      const rows = data.details[ticker] || [];
      if (!rows.length) return '<div style="margin-left:24px;color:#6b7280;">No variable overlaps for this stock.</div>';

      const cells = rows.map(r => {
        if (r.__group__) {
          return '<tr class="group-row"><td colspan="6">' + r.__group__ + '</td></tr>';
        }
        const v  = formatVal(r.Value);
        const f1 = formatVal(r.FT1);
        const f0 = formatVal(r.FT0);
        const d1 = formatVal(r.d_vs_FT1);
        const d0 = formatVal(r.d_vs_FT0);
        const c1 = (!d1)? '' : (parseFloat(d1)>=0 ? 'pos' : 'neg');
        const c0 = (!d0)? '' : (parseFloat(d0)>=0 ? 'pos' : 'neg');

        // significance flags for CORE only
        const isCore = !!r.is_core;
        const s1 = isCore && !!r.sig1;
        const s0 = isCore && !!r.sig0;

        // numeric deltas for direction
        const d1num = (r.d_vs_FT1==null || isNaN(r.d_vs_FT1)) ? NaN : Number(r.d_vs_FT1);
        const d0num = (r.d_vs_FT0==null || isNaN(r.d_vs_FT0)) ? NaN : Number(r.d_vs_FT0);

        // class resolution: for CORE, use sig_up/sig_down; for MODERATE use gray background
        let rowClass = '';
        if (isCore && (s1 || s0)) {
          let delta = NaN;
          if (s1 && s0) {
            const abs1 = isNaN(d1num) ? -Infinity : Math.abs(d1num);
            const abs0 = isNaN(d0num) ? -Infinity : Math.abs(d0num);
            delta = (abs1 >= abs0) ? d1num : d0num;
          } else {
            delta = (!isNaN(d1num) && s1) ? d1num : d0num;
          }
          rowClass = (delta >= 0) ? 'sig_up' : 'sig_down';
        } else if (!isCore) {
          rowClass = 'moderate';
        }

        return `
          <tr class="${rowClass}">
            <td class="col-var">${r.Variable}</td>
            <td class="col-val">${v}</td>
            <td class="col-ft1">${f1}</td>
            <td class="col-ft0">${f0}</td>
            <td class="col-d1 ${c1}">${d1}</td>
            <td class="col-d0 ${c0}">${d0}</td>
          </tr>`;
      }).join('');

      return `
        <table class="child-table">
          <colgroup>
            <col class="col-var"/><col class="col-val"/><col class="col-ft1"/><col class="col-ft0"/><col class="col-d1"/><col class="col-d0"/>
          </colgroup>
          <thead>
            <tr>
              <th class="col-var">Variable</th>
              <th class="col-val">Value</th>
              <th class="col-ft1">FT=1 median</th>
              <th class="col-ft0">FT=0 median</th>
              <th class="col-d1">Δ vs FT=1</th>
              <th class="col-d0">Δ vs FT=0</th>
            </tr>
          </thead>
          <tbody>${cells}</tbody>
        </table>`;
    }

    $(function() {
      const table = $('#align').DataTable({
        data: data.rows,
        responsive: true,
        paging: false, info: false, searching: false,
        order: [[0,'asc']],
        columns: [
          { data: 'Ticker' },
          { data: 'FT1_val', render: (d)=>barCellBlue(d) },
          { data: 'FT0_val', render: (d)=>barCellRed(d) },
        ]
      });

      // Whole-row toggle child
      $('#align tbody').on('click', 'tr', function () {
        const row = table.row(this);
        if (row.child.isShown()) {
          row.child.hide(); $(this).removeClass('shown');
        } else {
          const ticker = row.data().Ticker;
          row.child(childTableHTML(ticker)).show(); $(this).addClass('shown');
        }
      });
    });
  </script>
</body>
</html>
        """
        html = html.replace("%%PAYLOAD%%", json.dumps(payload))
        import streamlit.components.v1 as components
        components.html(html, height=620, scrolling=True)
    else:
        st.info("No eligible rows yet. Add manual stocks and/or ensure FT=1/FT=0 medians are built.")
elif st.session_state.rows and (models_tbl.empty or not {"FT=1","FT=0"}.issubset(models_tbl.columns)):
    st.info("Upload DB and click **Build model stocks** to compute FT=1/FT=0 medians first.")
else:
    st.info("Add at least one stock above to compute alignment.")
