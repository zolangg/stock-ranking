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

def _percentify_col(df: pd.DataFrame, col: str):
    """If a % column looks fractional (<=2 by abs), convert to percent (×100)."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.where(s.notna() & (s.abs() <= 2), s * 100.0, s)

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []      # manual rows ONLY
if "last" not in st.session_state: st.session_state.last = {}      # last manual row
if "models" not in st.session_state: st.session_state.models = {}  # {"models_tbl":..., "mad_tbl":..., "var_core":[...], "var_moderate":[...]}
if "predvol_model" not in st.session_state: st.session_state.predvol_model = None  # {'mode':'sk'|'fallback', 'obj':..., 'coefs':...}

# ============================== Core & Moderate sets ==============================
# UPDATED per your request:
#  - Add PM_Vol_%_of_Pred and PM Vol (%) to CORE
#  - Remove PM_Vol_M from MODERATE (keep PM_$Vol_M$)
VAR_CORE = ["Gap_%", "RVOL", "FR_x", "PM$Vol/MC_%", "Catalyst", "PM_Vol_%_of_Pred", "PM Vol (%)"]
VAR_MODERATE = ["MarketCap_M$", "Float_M", "PM_$Vol_M$", "ATR_$"]
VAR_ALL = VAR_CORE + VAR_MODERATE

# ============================== Upload DB → Build Medians & Train PredVol ==============================
st.subheader("Upload Database")

uploaded = st.file_uploader("Upload .xlsx with your DB", type=["xlsx"], key="db_upl")
build_btn = st.button("Build model stocks", use_container_width=True, key="db_build_btn")

def _fit_daily_volume_model(df_map: pd.DataFrame):
    """
    Fit Model A (Predicted Daily Volume in M) on available rows.
    Tries LASSO (sklearn). If sklearn unavailable, uses fallback BIC model:
      ln(DVolM) = 3.094686 + (-0.335751)*ln_atr + (0.371318)*ln_pm
    Returns dict with prediction function and status.
    """
    # Need actual Daily Vol (M) to train
    # The merged sheet often has 'Daily Vol (M)' or similar
    # Try to find from original raw mapping if present
    # Here, df_map should contain columns: Daily_Vol_M (if present), PM_Vol_M, ATR_$, MarketCap_M$, PM_$Vol_M$, Float_M, Gap_%
    # We will build logs and fit on complete cases.
    eps = 1e-6

    # Prefer 'Daily Vol (M)' from DB if mapped in df_map as Daily_Vol_M, else bail to fallback
    y = pd.to_numeric(df_map.get("Daily_Vol_M"), errors="coerce")
    if y is None or y.dropna().empty:
        return {"mode":"fallback", "note":"No Daily_Vol_M target in DB; using fallback BIC model.", 
                "coefs":{"b0":3.0946860, "b_ln_atr":-0.3357511, "b_ln_pm":0.3713177}}

    # Features
    mcap   = pd.to_numeric(df_map.get("MarketCap_M$"), errors="coerce")
    gap    = pd.to_numeric(df_map.get("Gap_%"), errors="coerce")
    atr    = pd.to_numeric(df_map.get("ATR_$"), errors="coerce")
    pm     = pd.to_numeric(df_map.get("PM_Vol_M"), errors="coerce")
    pmdol  = pd.to_numeric(df_map.get("PM_$Vol_M$"), errors="coerce")
    flt    = pd.to_numeric(df_map.get("Float_M"), errors="coerce")
    cat    = pd.to_numeric(df_map.get("Catalyst"), errors="coerce")
    cat    = np.where(np.isnan(cat), 0.0, np.where(cat != 0, 1.0, 0.0))

    # Prepare engineered features
    ln_mcap  = np.log(np.clip(mcap, eps, None))
    ln_gapf  = np.log(np.clip((np.where(np.isnan(gap), 0.0, gap)/100.0), eps, None))
    ln_atr   = np.log(np.clip(atr, eps, None))
    ln_pm    = np.log(np.clip(pm, eps, None))
    ln_pm_d  = np.log(np.clip(pmdol, eps, None))
    ln_float = np.log(np.clip(flt, eps, None))
    FR       = np.where(np.isfinite(pm) & np.isfinite(flt) & (flt>0), pm/np.clip(flt, eps, None), np.nan)
    ln_FR    = np.log(np.clip(FR, eps, None))
    y_ln     = np.log(np.clip(pd.to_numeric(y, errors="coerce"), eps, None))

    X = pd.DataFrame({
        "ln_mcap": ln_mcap, "ln_gapf": ln_gapf, "ln_atr": ln_atr,
        "ln_pm": ln_pm, "ln_pm_dol": ln_pm_d, "ln_float": ln_float,
        "ln_FR": ln_FR, "catalyst": cat
    })
    mask = X.notna().all(axis=1) & np.isfinite(y_ln)
    X = X.loc[mask].copy()
    y_ln = y_ln.loc[mask].copy()
    if len(y_ln) < 20:
        # fallback if too few rows
        return {"mode":"fallback", "note":"Too few rows to train; fallback BIC model.",
                "coefs":{"b0":3.0946860, "b_ln_atr":-0.3357511, "b_ln_pm":0.3713177}}

    # Try sklearn LASSO selection + OLS refit on selected features
    try:
        from sklearn.linear_model import LassoCV, LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        # Standardize then LASSO CV
        lasso = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                              LassoCV(cv=5, random_state=42, n_alphas=100))
        lasso.fit(X.values, y_ln.values)
        # Coefs after scaling sit in step 1
        lasso_cv = lasso.named_steps['lassocv']
        # Retrieve selected coefficients in the original feature space using the linear model refit
        # We'll refit a plain OLS on the selected features:
        coef = lasso_cv.coef_
        names = X.columns.values
        sel = [names[i] for i, b in enumerate(coef) if abs(b) > 1e-10]
        if not sel:
            # Degenerate: fallback
            return {"mode":"fallback", "note":"LASSO selected no predictors; fallback BIC model.",
                    "coefs":{"b0":3.0946860, "b_ln_atr":-0.3357511, "b_ln_pm":0.3713177}}
        # Refit OLS on selected set
        X_sel = X[sel].values
        ols = LinearRegression()
        ols.fit(X_sel, y_ln.values)

        model = {
            "mode":"sk",
            "note":f"LASSO selected: {', '.join(sel)}; OLS refit used for prediction.",
            "sel": sel,
            "ols": ols,
            "feat_names": list(X.columns),
        }
        return model
    except Exception:
        # sklearn not available or failed → fallback BIC
        return {"mode":"fallback", "note":"sklearn unavailable; using fallback BIC model.",
                "coefs":{"b0":3.0946860, "b_ln_atr":-0.3357511, "b_ln_pm":0.3713177}}

def _predict_predvol_from_model(model_dict, row_dict):
    """Return Predicted Daily Volume (M) for a single row (dict of raw numeric vars)."""
    eps = 1e-6
    atr = float(pd.to_numeric(row_dict.get("ATR_$"), errors="coerce"))
    pm  = float(pd.to_numeric(row_dict.get("PM_Vol_M"), errors="coerce"))
    mcap = float(pd.to_numeric(row_dict.get("MarketCap_M$"), errors="coerce"))
    gap  = float(pd.to_numeric(row_dict.get("Gap_%"), errors="coerce"))
    pmdol= float(pd.to_numeric(row_dict.get("PM_$Vol_M$"), errors="coerce"))
    flt  = float(pd.to_numeric(row_dict.get("Float_M"), errors="coerce"))
    cat  = float(pd.to_numeric(row_dict.get("Catalyst"), errors="coerce"))
    cat  = 0.0 if np.isnan(cat) else (1.0 if cat != 0 else 0.0)

    ln_mcap  = np.log(max(mcap, eps))
    ln_gapf  = np.log(max((0.0 if np.isnan(gap) else gap)/100.0, eps))
    ln_atr   = np.log(max(atr, eps))
    ln_pm    = np.log(max(pm, eps))
    ln_pm_d  = np.log(max(pmdol if np.isfinite(pmdol) else eps, eps))
    ln_float = np.log(max(flt, eps))
    FR       = (pm / max(flt, eps)) if np.isfinite(pm) and np.isfinite(flt) and flt > 0 else np.nan
    ln_FR    = np.log(max((FR if np.isfinite(FR) else eps), eps))

    if model_dict is None:
        # Default to simple proxy if no model (use ln_pm only with neutral intercept)
        pred_ln = 3.0 + 0.35 * ln_pm
        return float(max(np.exp(pred_ln) - eps, 0.0))

    if model_dict.get("mode") == "fallback":
        b = model_dict["coefs"]
        pred_ln = b["b0"] + b["b_ln_atr"]*ln_atr + b["b_ln_pm"]*ln_pm
        return float(max(np.exp(pred_ln) - eps, 0.0))

    # sklearn path
    sel = model_dict.get("sel", [])
    ols = model_dict.get("ols", None)
    if not sel or ols is None:
        # safety fallback
        pred_ln = 3.0 + 0.35 * ln_pm
        return float(max(np.exp(pred_ln) - eps, 0.0))

    feats = {
        "ln_mcap": ln_mcap, "ln_gapf": ln_gapf, "ln_atr": ln_atr,
        "ln_pm": ln_pm, "ln_pm_dol": ln_pm_d, "ln_float": ln_float,
        "ln_FR": ln_FR, "catalyst": cat
    }
    X_row = np.array([[feats[n] for n in sel]], dtype=float)
    pred_ln = float(ols.predict(X_row)[0])
    return float(max(np.exp(pred_ln) - eps, 0.0))

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

            # --- auto-detect FT group column ---
            possible = [c for c in raw.columns if _norm(c) in {"ft","ft01","group","label"}]
            col_group = possible[0] if possible else None
            if col_group is None:
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

                def add_num(df, name, src_candidates):
                    src = _pick(raw, src_candidates)
                    if src:
                        df[name] = pd.to_numeric(raw[src].map(_to_float), errors="coerce")

                # Map numeric columns
                add_num(df, "MarketCap_M$", ["marketcap m","market cap (m)","mcap m","marketcap_m$","market cap m$","market cap (m$)","marketcap","market_cap m","market_cap m$"])
                add_num(df, "Float_M",      ["float m","public float (m)","float_m","float (m)","float m shares","float m shares"])
                add_num(df, "Gap_%",        ["gap %","gap%","premarket gap","gap","gap percent","gap_percent"])
                add_num(df, "ATR_$",        ["atr $","atr$","atr (usd)","atr","daily atr","daily_atr"])
                add_num(df, "RVOL",         ["rvol","relative volume","rvol @ bo","rvol at bo","rvol_bo"])
                add_num(df, "PM_Vol_M",     ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)","premarket volume (m)","pm_vol (m)","pm_vol_m"])
                add_num(df, "PM_$Vol_M$",   ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol","pm dollar volume (m)","pm $vol (m$)","pm $vol m"])
                # Also bring in "PM Vol (%)" from DB if present (to include as CORE)
                add_num(df, "PM Vol (%)",   ["pm vol (%)","pm volume (%)","pm_vol %","pm_vol (%)","pm vol percent","pm_vol_percent"])
                # Try to pull Daily Volume (M) target for training PredVol model (if present)
                add_num(df, "Daily_Vol_M",  ["daily vol (m)","daily volume (m)","daily_vol (m)","daily_vol_m","daily vol m","dvol m","dvol (m)"])

                # --- Catalyst (binary: Yes/No/1/0/True/False), if present ---
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

                # Normalize % columns (auto scale ×100 if fractional)
                for _pct_col in ["Gap_%", "PM$Vol/MC_%", "PM Vol (%)", "PM_Vol_%_of_Pred", "PM_Vol_%"]:
                    _percentify_col(df, _pct_col)

                # === Train Predicted Daily Volume model (Model A) ===
                # (Try LASSO via sklearn; fallback to BIC 2-term model if needed)
                # Use the db-mapped frame; model needs Daily_Vol_M as target if available
                st.session_state.predvol_model = _fit_daily_volume_model(df)

                # Compute PredVol_M for all rows we have enough inputs
                pred_list = []
                for i, r in df.iterrows():
                    pred = _predict_predvol_from_model(st.session_state.predvol_model, r.to_dict())
                    pred_list.append(pred)
                df["PredVol_M"] = pred_list

                # Compute PM_Vol_%_of_Pred = PM_Vol_M / PredVol_M * 100
                if {"PM_Vol_M","PredVol_M"}.issubset(df.columns):
                    num = pd.to_numeric(df["PM_Vol_M"], errors="coerce")
                    den = pd.to_numeric(df["PredVol_M"], errors="coerce")
                    df["PM_Vol_%_of_Pred"] = (num / den * 100.0).replace([np.inf, -np.inf], np.nan)
                    _percentify_col(df, "PM_Vol_%_of_Pred")

                # Ensure Gap_% is percent scale for medians (handles fractional sources)
                _percentify_col(df, "Gap_%")

                # Normalize to FT binary groups
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

                    # Limit medians/MADs to available vars (with updated CORE/MOD lists)
                    var_core = [v for v in VAR_CORE if v in df.columns]
                    var_mod  = [v for v in VAR_MODERATE if v in df.columns]
                    var_all  = var_core + var_mod

                    # medians per group
                    gmed = df.groupby("Group")[var_all].median(numeric_only=True).T

                    # robust spread: MAD per group
                    def _mad(series: pd.Series) -> float:
                        s = pd.to_numeric(series, errors="coerce").dropna()
                        if s.empty:
                            return np.nan
                        med = float(np.median(s))
                        return float(np.median(np.abs(s - med)))

                    gmads = df.groupby("Group")[var_all].apply(lambda g: g.apply(_mad)).T

                    st.session_state.models = {
                        "models_tbl": gmed,
                        "mad_tbl": gmads,
                        "var_core": var_core,
                        "var_moderate": var_mod
                    }
                    note = (st.session_state.predvol_model or {}).get("note", "")
                    st.success("Built model stocks. " + (f"({note})" if note else ""))
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

# ============================== ➕ Manual Input (NO Short Interest) ==============================
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
    catalyst = 1.0 if catalyst_yn == "Yes" else 0.0

    # Predict daily volume (M) using the trained model (if available)
    pred_model = st.session_state.get("predvol_model")
    pred_input = {
        "ATR_$": atr_usd, "PM_Vol_M": pm_vol, "MarketCap_M$": mc_m, "Gap_%": gap_pct,
        "PM_$Vol_M$": pm_dol, "Float_M": float_m, "Catalyst": catalyst
    }
    pred_vol_m = _predict_predvol_from_model(pred_model, pred_input) if pred_model else np.nan
    pm_pct_of_pred = (pm_vol / pred_vol_m * 100.0) if (pred_vol_m and pred_vol_m > 0) else np.nan

    row = {
        "Ticker": ticker,
        "MarketCap_M$": mc_m,
        "Float_M": float_m,
        "Gap_%": gap_pct,         # manual already in percent (no scaling)
        "ATR_$": atr_usd,
        "RVOL": rvol,
        "PM_Vol_M": pm_vol,
        "PM_$Vol_M$": pm_dol,
        "FR_x": fr,
        "PM$Vol/MC_%": pmmc,
        "CatalystYN": catalyst_yn,
        "Catalyst": catalyst,
        "PredVol_M": pred_vol_m,
        "PM_Vol_%_of_Pred": pm_pct_of_pred
    }
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
    detail_order = [("Core variables", var_core), ("Moderate variables", var_mod)]

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
                if v not in models_tbl.index:
                    continue
                va = pd.to_numeric(stock.get(v), errors="coerce")
                v1 = models_tbl.loc[v, "FT=1"] if (v in models_tbl.index) else np.nan
                v0 = models_tbl.loc[v, "FT=0"] if (v in models_tbl.index) else np.nan
                m1 = mad_tbl.loc[v, "FT=1"] if (not mad_tbl.empty and v in mad_tbl.index and "FT=1" in mad_tbl.columns) else np.nan
                m0 = mad_tbl.loc[v, "FT=0"] if (not mad_tbl.empty and v in mad_tbl.index and "FT=0" in mad_tbl.columns) else np.nan

                if pd.isna(va) and pd.isna(v1) and pd.isna(v0):
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
        payload = {"rows": summary_rows, "details": detail_map}

        html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css"/>
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
          return `<tr class="group-row"><td colspan="6">${r.__group__}</td></tr>`;
        }
        const v  = formatVal(r.Value);
        const f1 = formatVal(r.FT1);
        const f0 = formatVal(r.FT0);
        const d1 = formatVal(r.d_vs_FT1);
        const d0 = formatVal(r.d_vs_FT0);
        const c1 = (!d1)? '' : (parseFloat(d1)>=0 ? 'pos' : 'neg');
        const c0 = (!d0)? '' : (parseFloat(d0)>=0 ? 'pos' : 'neg');

        const isCore = !!r.is_core;
        const s1 = isCore && !!r.sig1;
        const s0 = isCore && !!r.sig0;

        const d1num = (r.d_vs_FT1==null || isNaN(r.d_vs_FT1)) ? NaN : Number(r.d_vs_FT1);
        const d0num = (r.d_vs_FT0==null || isNaN(r.d_vs_FT0)) ? NaN : Number(r.d_vs_FT0);

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
          <thead
