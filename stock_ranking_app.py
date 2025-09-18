# stock_ranking_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math

# ===========================
# R bridge (rpy2) – safe import
# ===========================
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    R_OK = True
except Exception as e:
    R_OK = False
    R_ERR = str(e)

def _pd_to_r(obj):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(obj)

def _r_to_pd(obj):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(obj)

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# =========================================================
# Load BART models (RDS) once
# =========================================================
@st.cache_resource(show_spinner=False)
def load_bart_models():
    if not R_OK:
        return dict(ok=False, err=f"rpy2/R not available: {R_ERR}")
    try:
        # Import R packages and functions properly (avoid r['readRDS'])
        base  = importr('base')
        stats = importr('stats')  # predict() generic
        readRDS = base.readRDS
        r_predict = ro.r('predict')  # generic predict; S3 dispatch in R

        # Load models + predictor name vectors
        modelA = readRDS("bart_model_A_predDVol_ln.rds")
        predsA = list(_r_to_pd(readRDS("bart_model_A_predictors.rds")))
        modelB = readRDS("bart_model_B_FT.rds")
        predsB = list(_r_to_pd(readRDS("bart_model_B_predictors.rds")))
        return dict(
            ok=True, err=None,
            base=base, stats=stats,
            readRDS=readRDS, r_predict=r_predict,
            A=modelA, A_preds=predsA,
            B=modelB, B_preds=predsB
        )
    except Exception as e:
        return dict(ok=False, err=str(e))

MODELS = load_bart_models()
if not MODELS["ok"]:
    st.error(
        "Could not load BART models (.rds). Make sure R + rpy2 are installed and the four files are in the app folder:\n\n"
        "  • bart_model_A_predDVol_ln.rds\n"
        "  • bart_model_A_predictors.rds\n"
        "  • bart_model_B_FT.rds\n"
        "  • bart_model_B_predictors.rds\n\n"
        f"Details: {MODELS['err']}"
    )
    st.stop()

# =========================================================
# Helpers
# =========================================================
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if not keep: return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy().fillna("")
    header = "| " + " | ".join(keep) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = []
        for c in keep:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---------- Qualitative criteria ----------
QUAL_CRITERIA = [
    {
        "name": "GapStruct",
        "question": "Gap & Trend Development:",
        "options": [
            "Gap fully reversed: price loses >80% of gap.",
            "Choppy reversal: price loses 50–80% of gap.",
            "Partial retracement: price loses 25–50% of gap.",
            "Sideways consolidation: gap holds, price within top 25% of gap.",
            "Uptrend with deep pullbacks (>30% retrace).",
            "Uptrend with moderate pullbacks (10–30% retrace).",
            "Clean uptrend, only minor pullbacks (<10%).",
        ],
        "weight": 0.15,
        "help": "How well the gap holds and trends.",
    },
    {
        "name": "LevelStruct",
        "question": "Key Price Levels:",
        "options": [
            "Fails at all major support/resistance; cannot hold any key level.",
            "Briefly holds/reclaims a level but loses it quickly; repeated failures.",
            "Holds one support but unable to break resistance; capped below a key level.",
            "Breaks above resistance but cannot stay; dips below reclaimed level.",
            "Breaks and holds one major level; most resistance remains above.",
            "Breaks and holds several major levels; clears most overhead resistance.",
            "Breaks and holds above all resistance; blue sky.",
        ],
        "weight": 0.15,
        "help": "Break/hold behavior at key levels.",
    },
    {
        "name": "Monthly",
        "question": "Monthly/Weekly Chart Context:",
        "options": [
            "Sharp, accelerating downtrend; new lows repeatedly.",
            "Persistent downtrend; still lower lows.",
            "Downtrend losing momentum; flattening.",
            "Clear base; sideways consolidation.",
            "Bottom confirmed; higher low after base.",
            "Uptrend begins; breaks out of base.",
            "Sustained uptrend; higher highs, blue sky.",
        ],
        "weight": 0.10,
        "help": "Higher-timeframe bias.",
    },
]

# ---------- Sidebar: weights & modifiers ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01, key="w_rvol")
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01, key="w_atr")
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01, key="w_si")
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01, key="w_fr")
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01, key="w_float")

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}"
    )

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05, key="news_weight")
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05, key="dil_weight")

st.sidebar.header("FT Threshold")
ft_thresh = st.sidebar.slider("FT label threshold", 0.10, 0.90, 0.50, 0.01)

# --- PredVol CI (display only) ---
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
                             help="For the CI on predicted volume. Does not affect BART prediction, display only.")

# Normalize blocks separately
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------- Numeric bucket scorers ----------
def pts_rvol(x: float) -> int:
    for th, p in [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)]:
        if x < th: return p
    return 7

def pts_atr(x: float) -> int:
    for th, p in [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)]:
        if x < th: return p
    return 7

def pts_si(x: float) -> int:
    for th, p in [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)]:
        if x < th: return p
    return 7

def pts_fr(pm_vol_m: float, float_m: float) -> int:
    if float_m <= 0: return 1
    rot = pm_vol_m / float_m
    for th, p in [(0.01,1),(0.03,2),(0.10,3),(0.25,4),(0.50,5),(1.00,6)]:
        if rot < th: return p
    return 7

def pts_float(float_m: float) -> int:
    if float_m <= 3: return 7
    for th, p in [(200,2),(100,3),(50,4),(35,5),(10,6)]:
        if float_m > th: return p
    return 7

def odds_label(score: float) -> str:
    if score >= 85: return "Very High Odds"
    elif score >= 70: return "High Odds"
    elif score >= 55: return "Moderate Odds"
    elif score >= 40: return "Low Odds"
    else: return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 85 else
            "A+"  if score_pct >= 80 else
            "A"   if score_pct >= 70 else
            "B"   if score_pct >= 60 else
            "C"   if score_pct >= 45 else "D")

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0:
        return 0.0, 0.0
    low  = pred_m * math.exp(-z * sigma_ln)
    high = pred_m * math.exp( z * sigma_ln)
    return low, high

# =========================================================
# Feature engineering for the BART models (must match R)
# =========================================================
def make_features_df(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst (0/1)
    """
    eps = 1e-6
    df = rows.copy()
    df["FR"]      = df["PMVolM"] / np.maximum(df["FloatM"], eps)
    df["ln_pm"]   = np.log(np.maximum(df["PMVolM"], eps))
    df["ln_pmdol"] = np.log(np.maximum(df.get("PMDolM", 0.0), eps))
    df["ln_fr"]     = np.log(np.maximum(df["FR"], eps))
    df["ln_gapf"]   = np.log(np.maximum(df["GapPct"], 0.0)/100.0 + eps)
    df["ln_atr"]    = np.log(np.maximum(df["ATR"], eps))
    df["ln_mcap"]   = np.log(np.maximum(df["MCapM"], eps))
    df["ln_pmdol_per_mcap"] = np.log(np.maximum(df.get("PMDolM", 0.0) / np.maximum(df["MCapM"], eps), eps))
    df["Catalyst"] = (df["Catalyst"] != 0).astype(int)
    return df

def predict_predVolM_from_features(feat_df: pd.DataFrame) -> np.ndarray:
    """ BART Model A: returns predicted daily volume in millions. """
    predsA = MODELS["A_preds"]
    X = feat_df.loc[:, predsA].copy()
    draws = MODELS["r_predict"](MODELS["A"], newdata=_pd_to_r(X))
    draws_pd = _r_to_pd(draws)  # draws x n
    pred_ln = np.array(draws_pd).mean(axis=0)
    return np.exp(pred_ln)

def predict_ft_prob_from_features(feat_df: pd.DataFrame) -> np.ndarray:
    """ BART Model B: returns probability in [0,1]. """
    predsA = MODELS["A_preds"]
    X_A = feat_df.loc[:, predsA].copy()
    predA_draws = MODELS["r_predict"](MODELS["A"], newdata=_pd_to_r(X_A))
    predA_pd = _r_to_pd(predA_draws)    # draws x n
    predA_ln = np.array(predA_pd).mean(axis=0)

    feat_aug = feat_df.copy()
    feat_aug["PredVol_M"] = np.exp(predA_ln)

    predsB = MODELS["B_preds"]
    X_B = feat_aug.loc[:, predsB].copy()
    p_obj = MODELS["r_predict"](MODELS["B"], newdata=_pd_to_r(X_B))

    # Try prob.test.mean, else average posterior matrix
    phat = None
    try:
        phat = np.array(_r_to_pd(p_obj.rx2("prob.test.mean")))
    except Exception:
        pass
    if phat is None or phat.size == 0:
        try:
            ppost = _r_to_pd(p_obj.rx2("ppost"))  # draws x n
            phat = np.array(ppost).mean(axis=0)
        except Exception as e:
            st.error(f"Could not extract probabilities from BART classifier: {e}")
            phat = np.zeros(len(X_B), dtype=float)
    return np.clip(phat, 0.0, 1.0)

# =========================================================
# Session state
# =========================================================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =========================================================
# Tabs
# =========================================================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL",   min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # Flow / size / PM
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_dol_m = st.number_input("PM $Vol (Millions $)",      min_value=0.0, value=0.0, step=0.01, format="%.2f")

        # Catalyst & sliders
        with c_top[2]:
            catalyst_flag = st.checkbox("Catalyst present", value=False, help="Used by BART as binary 0/1.")
            news_points     = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None)
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # === BART Features ===
        feat_input = pd.DataFrame([{
            "PMVolM": pm_vol_m,
            "PMDolM": pm_dol_m,
            "FloatM": float_m,
            "GapPct": gap_pct,
            "ATR": atr_usd,
            "MCapM": mc_m,
            "Catalyst": 1 if catalyst_flag else 0
        }])
        feat = make_features_df(feat_input)

        # === Model A: predicted daily volume (M) ===
        pred_vol_m = float(predict_predVolM_from_features(feat)[0])

        # CI bands (display only)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        # === Model B: FT probability ===
        ft_prob = float(predict_ft_prob_from_features(feat)[0])
        ft_label = "FT" if ft_prob >= ft_thresh else "Fail"

        # === Numeric points (your scoring block) ===
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # === Qualitative points ===
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            val = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}"), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(val)
        qual_pct = (qual_0_7/7.0)*100.0

        # === Combine + modifiers ===
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*news_points*10 + dilution_weight*dilution_points*10, 2)
        final_score = max(0.0, min(100.0, final_score))

        # === Diagnostics to save ===
        pm_pct_of_pred   = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x   = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc  = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            # BART predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            "FT_Prob_%": round(100.0 * ft_prob, 1),
            "FT_Label": ft_label,

            # Ratios
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),

            # raw inputs for reproducibility
            "_MCap_M": mc_m,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,
            "_Float_M": float_m,
            "_Gap_%": gap_pct,
            "_Catalyst": 1 if catalyst_flag else 0,
            "_SI_%": si_pct,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Numeric Block", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qual Block",    f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",   f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d1.caption("Premarket volume ÷ float.")
        d2.metric("PM $Vol / MC",      f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("PM dollar volume ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        ft_prob_pct = l.get("FT_Prob_%", 0.0)
        ft_lab = l.get("FT_Label","—")
        d4.metric("FT Probability", f"{ft_prob_pct:.1f}%")
        d4.caption(f"Label: {ft_lab}  (threshold {int(100*ft_thresh)}%)")

with tab_rank:
    st.subheader("Current Ranking")

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM_%_of_Pred",
            "FT_Prob_%","FT_Label",
            "PM_FloatRot_x","PM$ / MC_%"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","FT_Label") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Level"),
                "Numeric_%": st.column_config.NumberColumn("Numeric_%", format="%.2f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "FT_Prob_%": st.column_config.NumberColumn("FT Probability (%)", format="%.1f"),
                "FT_Label": st.column_config.TextColumn("FT Label"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
            }
        )

        # Row delete buttons (top 12)
        st.markdown("#### Delete rows")
        del_cols = st.columns(4)
        head12 = df.head(12).reset_index(drop=True)
        for i, r in head12.iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    do_rerun()

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        st.markdown("### 📋 Ranking (Markdown view)")
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
