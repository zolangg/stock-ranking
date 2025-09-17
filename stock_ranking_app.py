import streamlit as st
import pandas as pd
import math

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ---------- Markdown table helper ----------
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
            v = [c]
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

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

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
    },
]

# ---------- Sidebar ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01
    )

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

# --- Confidence ---
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01)

# Normalize weights
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights: q_weights[k] /= qual_sum

# ---------- Scorers ----------
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

# ---------- Premarket Models ----------

def predict_day_volume_m_premarket(mcap_m, gap_pct, atr):
    """
    Premarket day-volume model (pooled log-log OLS).
    Returns **millions of shares**.
    """
    e1 = e2 = e3 = 1e-6
    gp = max(gap_pct, 0.0) / 100.0
    z_gap = (math.log(gp + e_g) - mu_g) / sd_g
    ln_y = (
        3.1435
        + 0.1608 * math.log(max(mcap_m,0)+e1)
        + 0.6704 * math.log(gp+e2)         # use fraction here
        - 0.3878 * math.log(max(atr,0)+e3)
    )
    return math.exp(ln_y)  # already in **millions** per our model spec

def predict_ft_prob_premarket(float_m_shares, gap_pct):
    """
    Premarket FT probability (logistic).
    Returns probability in [0,1].
    """
    e_f = e_g = 1e-6
    mu_f, sd_f = 2.34, 0.91
    mu_g, sd_g = -0.47, 0.56
    b0, b1, b2 = -0.982, -1.241, 1.372

    z_float = (math.log(max(float_m_shares,0)+e_f)-mu_f)/sd_f
    z_gap   = (math.log(max(gap_pct,0)+e_g)-mu_g)/sd_g
    lp = b0 + b1*z_float + b2*z_gap
    return 1/(1+math.exp(-lp))

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    return pred_m*math.exp(-z*sigma_ln), pred_m*math.exp(z*sigma_ln)

def ft_label_and_color(prob_pct: float):
    if prob_pct < 40: return "Low Odds", "🔴"
    elif prob_pct < 70: return "Moderate Odds", "🟠"
    else: return "High Odds", "🟢"

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2,1.2,1.0])

        with c_top[0]:
            ticker   = st.text_input("Ticker","").strip().upper()
            rvol     = st.number_input("RVOL",0.0,step=0.01,format="%.2f")
            atr_usd  = st.number_input("ATR ($)",0.0,step=0.01,format="%.2f")
            float_m  = st.number_input("Public Float (Millions)",0.0,step=0.01,format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)",0.0,step=0.1,format="%.1f")

        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)",0.0,step=0.01,format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)",0.0,step=0.01,format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)",0.0,step=0.01,format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)",0.0,step=0.0001,format="%.4f")

        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)",-1.0,1.0,0.0,0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)",-1.0,1.0,0.0,0.05)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i%3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"],1)),
                    format_func=lambda x:x[1],
                    key=f"qual_{crit['name']}"
                )
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Predictions
        pred_vol_m = predict_day_volume_m_premarket(mc_m,gap_pct,atr_usd)
        ci68_l,ci68_u = ci_from_logsigma(pred_vol_m,sigma_ln,1.0)
        ci95_l,ci95_u = ci_from_logsigma(pred_vol_m,sigma_ln,1.96)
        ft_prob = predict_ft_prob_premarket(float_m,gap_pct)
        ft_pct = round(100*ft_prob,1)
        ft_label, ft_color = ft_label_and_color(ft_pct)

        # Scores
        p_rvol, p_atr, p_si = pts_rvol(rvol), pts_atr(atr_usd), pts_si(si_pct)
        p_fr, p_float = pts_fr(pm_vol_m,float_m), pts_float(float_m)
        num_0_7 = w_rvol*p_rvol+w_atr*p_atr+w_si*p_si+w_fr*p_fr+w_float*p_float
        num_pct = (num_0_7/7.0)*100.0
        qual_0_7= sum(q_weights[c["name"]]*float(st.session_state.get(f"qual_{c['name']}",(1,))[0]) for c in QUAL_CRITERIA)
        qual_pct= (qual_0_7/7.0)*100.0
        combo_pct=0.5*num_pct+0.5*qual_pct
        final_score=round(combo_pct+news_weight*catalyst_points*10+dilution_weight*dilution_points*10,2)
        final_score=max(0.0,min(100.0,final_score))

        row={
            "Ticker":ticker,
            "Odds":odds_label(final_score),
            "Level":grade(final_score),
            "OddsScore":final_score,
            "Numeric_%":round(num_pct,2),
            "Qual_%":round(qual_pct,2),
            "FinalScore":final_score,
            "PredVol_M":round(pred_vol_m,2),
            "PredVol_CI68_L":round(ci68_l,2),
            "PredVol_CI68_U":round(ci68_u,2),
            "PredVol_CI95_L":round(ci95_l,2),
            "PredVol_CI95_U":round(ci95_u,2),
            "FT_Prob_%":ft_pct,
            "FT_Label":f"{ft_color} {ft_label}",
            "_MCap_M":mc_m,"_SI_%":si_pct,
            "_ATR_$":atr_usd,"_PM_M":pm_vol_m,
            "_Float_M":float_m,
            "_Gap_%":gap_pct,
            "_Catalyst":float(catalyst_points),
            "_PM_VWAP": pm_vwap
        }
        st.session_state.rows.append(row)
        st.session_state.last=row
        st.session_state.flash=f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # Preview card
    l=st.session_state.last
    if l:
        st.markdown("---")
        cA,cB,cC,cD=st.columns(4)
        cA.metric("Last Ticker",l.get("Ticker","—"))
        cB.metric("Numeric Block",f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qual Block",f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Predicted Day Vol (M)",f"{l.get('PredVol_M',0):.2f}")
        d1.caption(f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} · CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f}")
        d2.metric("FT Probability",f"{l.get('FT_Prob_%',0):.1f}%")
        d2.caption(l.get("FT_Label","—"))
        d3.metric("PM Float Rotation",f"{(l.get('_PM_M',0)/max(l.get('_Float_M',1e-6),1e-6)):.3f}×")
        pm_dollar_vs_mc = 100.0 * (l.get("_PM_M", 0.0) * l.get("_PM_VWAP", 0.0)) / max(l.get("_MCap_M", 0.0), 1e-6)
        d4.metric("PM $Vol / MC", f"{pm_dollar_vs_mc:.1f}%")

# Ranking tab
with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df=pd.DataFrame(st.session_state.rows)
        df=df.loc[:,~df.columns.duplicated(keep="first")]
        df=df.sort_values("FinalScore",ascending=False).reset_index(drop=True)
        cols=["Ticker","Odds","Level","Numeric_%","Qual_%","FinalScore","PredVol_M","FT_Prob_%","FT_Label"]
        for c in cols:
            if c not in df.columns: df[c]=""
        st.dataframe(df[cols],use_container_width=True,hide_index=True)
        st.download_button("Download CSV",df.to_csv(index=False).encode("utf-8"),"ranking.csv","text/csv",use_container_width=True)
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
