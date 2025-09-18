import streamlit as st
import pandas as pd
import numpy as np
import math
import os

# ---------------- Page ----------------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ---------------- R bridge (BART models) ----------------
# Uses exactly the same engineered features & predictor sets as your R code
try:
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects as ro
    
    # Example: sending a pandas dataframe to R
    import pandas as pd
    df = pd.DataFrame({"x":[1,2,3]})
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd = ro.conversion.py2rpy(df)
    from rpy2.robjects.packages import importr
except Exception as e:
    st.error("rpy2 is required. Install R + rpy2, and R packages 'dbarts' and 'BART'.")
    st.stop()

pandas2ri.activate()
base = importr('base')
utils = importr('utils')
try:
    dbarts = importr('dbarts')
    BARTpkg = importr('BART')
    stats = importr('stats')
except Exception as e:
    st.error("Could not load R packages 'dbarts' and 'BART'. Make sure they are installed in R.")
    st.stop()

R = ro.r
R('''
compute_features <- function(df) {
  eps <- 1e-6
  # Expect columns: PMVolM, PMDolM, FloatM, GapPct, ATR, MCapM, Catalyst
  df$FR       <- df$PMVolM / pmax(df$FloatM, eps)
  df$ln_pm    <- log(pmax(df$PMVolM, eps))
  df$ln_pmdol <- log(pmax(df$PMDolM, eps))
  df$ln_fr    <- log(pmax(df$FR, eps))
  df$ln_gapf  <- log(pmax(df$GapPct, 0)/100 + eps)
  df$ln_atr   <- log(pmax(df$ATR, eps))
  df$ln_mcap  <- log(pmax(df$MCapM, eps))
  df$ln_pmdol_per_mcap <- log(pmax(df$PMDolM / pmax(df$MCapM, eps), eps))
  df$Catalyst <- as.integer(df$Catalyst != 0)
  df
}
predict_bartA_ln_draws <- function(modelA, newX) {
  # draws x n
  predict(modelA, newdata = newX)
}
predict_bartB_prob <- function(modelB, newX) {
  pr <- BART::predict(modelB, newdata = newX)
  if (!is.null(pr$prob.test.mean)) as.numeric(pr$prob.test.mean)
  else if (!is.null(pr$ppost)) rowMeans(pr$ppost)
  else stop("Unexpected structure from BART::predict (Model B).")
}
''')
compute_features_R = R['compute_features']
predict_bartA_ln_draws_R = R['predict_bartA_ln_draws']
predict_bartB_prob_R = R['predict_bartB_prob']

MODEL_A_RDS = "bart_model_A_predDVol_ln.rds"
PREDS_A_RDS = "bart_model_A_predictors.rds"
MODEL_B_RDS = "bart_model_B_FT.rds"
PREDS_B_RDS = "bart_model_B_predictors.rds"
EPS = 1e-6

@st.cache_resource
def load_r_models():
    for f in (MODEL_A_RDS, PREDS_A_RDS, MODEL_B_RDS, PREDS_B_RDS):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing model file: {f}")
    modelA = base.readRDS(MODEL_A_RDS)
    predsA = list(base.readRDS(PREDS_A_RDS))
    modelB = base.readRDS(MODEL_B_RDS)
    predsB = list(base.readRDS(PREDS_B_RDS))
    return modelA, predsA, modelB, predsB

try:
    modelA, predsA, modelB, predsB = load_r_models()
except Exception as e:
    st.error(str(e))
    st.stop()

# ---------------- Markdown table helper ----------------
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
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---------------- Session state ----------------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------------- Qualitative criteria (unchanged) ----------------
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

# ---------------- Sidebar weights & modifiers (unchanged scoring) ----------------
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

# Credible interval choice for BART pred-vol
st.sidebar.header("PredVol Credible Interval")
ci_choice = st.sidebar.select_slider("CI level (%)", options=[68, 80, 90, 95, 98], value=68)

# Normalize blocks separately (unchanged)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------------- Numeric bucket scorers (unchanged) ----------------
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

# FT prob formatting
def ft_band_label(p):
    if p >= 0.80: return "Very High"
    if p >= 0.65: return "High"
    if p >= 0.50: return "Moderate"
    if p >= 0.35: return "Low"
    return "Very Low"

def ci_quantiles(level):
    alpha = (100 - level) / 200.0
    return alpha, 1 - alpha

# ---------------- Tabs ----------------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # Float / PM volume / Market cap (and PM $Vol)
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_dol_m = st.number_input("PM $ Volume (Millions $)",   min_value=0.0, value=0.0, step=0.01, format="%.2f")

        # Modifiers & flags
        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            catalyst_flag   = st.checkbox("News/Catalyst present?", value=False)

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
        # ----- BART Model A: Predicted Daily Volume (M) with credible interval -----
        catalyst_num = 1 if catalyst_flag else 0
        one = pd.DataFrame([{
            "PMVolM": pm_vol_m,
            "PMDolM": pm_dol_m,
            "FloatM": float_m,
            "GapPct": gap_pct,
            "ATR": atr_usd,
            "MCapM": mc_m,
            "Catalyst": catalyst_num
        }])
        # Features in R (exact pipeline)
        rdf = pandas2ri.py2rpy(one)
        rdf_feat = compute_features_R(rdf)

        # Build Model A design matrix using EXACT predictors saved from R
        newX_A = R('function(df, cols) df[, cols, drop=FALSE]')(rdf_feat, ro.StrVector(predsA))
        # Posterior draws ln-scale
        ln_draws = predict_bartA_ln_draws_R(modelA, newX_A)
        ln_np = np.array(ln_draws)              # draws x 1
        predM_draws = np.exp(ln_np).reshape(-1) # millions
        pred_vol_m = float(np.mean(predM_draws))
        ql, qh = ci_quantiles(ci_choice)
        ci_l = float(np.quantile(predM_draws, ql))
        ci_u = float(np.quantile(predM_draws, qh))

        # ----- BART Model B: FT Probability (requires PredVol_M + its predictors) -----
        # attach PredVol_M back in R and slice B predictors exactly
        R.assign("tmp_feat_df", rdf_feat)
        R.assign("pred_vol_m_py", ro.FloatVector([pred_vol_m]))
        R('tmp_feat_df$PredVol_M <- pred_vol_m_py')
        newX_B = R('tmp_feat_df[, c(%s), drop=FALSE]' % (",".join([f'"{p}"' for p in predsB])))
        p_ft = float(predict_bartB_prob_R(modelB, newX_B)[0])
        ft_combo = f"{p_ft*100:.1f}% ({ft_band_label(p_ft)})"

        # ----- Scoring blocks (unchanged logic) -----
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}"), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*(catalyst_points*10) + dilution_weight*(dilution_points*10), 2)
        final_score = max(0.0, min(100.0, final_score))

        # ----- Diagnostics / display metrics -----
        pm_pct_of_pred   = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x   = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc  = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "OddsScore": final_score,
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            # BART predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI_L": round(ci_l, 2),
            "PredVol_CI_U": round(ci_u, 2),

            # Combined FT display to save space: "p% (Label)"
            "FT": ft_combo,
            "FT_Prob": round(p_ft, 4),  # keep raw prob as well (0..1)

            # Display metrics you asked to keep
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),

            # store raw inputs for later
            "_MCap_M": mc_m,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,
            "_Float_M": float_m,
            "_Gap_%": gap_pct,
            "_Catalyst": float(catalyst_points),
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
        d1.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.2f}%")
        d1.caption("PM dollar volume ÷ market cap × 100.")
        d2.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d2.caption(f"CI{ci_choice}: {l.get('PredVol_CI_L',0):.2f}–{l.get('PredVol_CI_U',0):.2f} M")
        d3.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d3.caption("PM volume ÷ predicted day volume × 100.")
        d4.metric("FT", l.get("FT","—"))
        d4.caption("BART FT probability with label")

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
            "PM$ / MC_%",
            "PredVol_M","PredVol_CI_L","PredVol_CI_U",
            "PM_%_of_Pred",
            "FT"  # fused prob + label
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","FT") else 0.0
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
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI_L": st.column_config.NumberColumn("Pred Vol CI Low (M)",  format="%.2f"),
                "PredVol_CI_U": st.column_config.NumberColumn("Pred Vol CI High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "FT": st.column_config.TextColumn("FT (Prob + Label)"),
            }
        )

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
