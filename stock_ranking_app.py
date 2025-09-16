import streamlit as st
import pandas as pd
import math

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Premarket Stock Ranking (2-Step Model)", layout="wide")
st.title("Premarket Stock Ranking — with Predicted Day Volume & FT Probability")

# =========================================================
# Small helpers
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

# =========================================================
# Session State
# =========================================================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =========================================================
# Qualitative criteria (unchanged)
# =========================================================
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

# =========================================================
# Sidebar — weights, modifiers, model params
# =========================================================
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

# Confidence (log-space) for volume prediction bands
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ for DayVol (residual std dev)", 0.10, 1.50, 0.60, 0.01)

# Normalize blocks
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights: q_weights[k] = q_weights[k] / qual_sum

# =========================================================
# Bucket scorers (unchanged)
# =========================================================
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

# =========================================================
# STEP 1 — Trimmed Day Volume Model (coefficients)
# =========================================================
# We fit in ln-space and exponentiate back to linear millions.
# ln(Y) = b0 + b_ATR*ln(1+ATR) + b_PM*ln(1+PMVol_M) + b_Float*ln(1+Float_M)
#         + b_Int * [ ln(1+Gap%) * ln(1+Float_M) ] + b_Cat * Catalyst_score

STEP1_COEF = {
    "b0":     3.90,   # intercept (tune from your "Production Formula" table)
    "b_ATR": -2.33,
    "b_PM":   3.16,
    "b_Flt": -0.44,
    "b_Int":  0.34,
    "b_Cat":  1.14,
}
def predict_day_volume_m_trimmed(atr_usd: float, pm_vol_m: float, float_m: float,
                                 gap_pct: float, catalyst_score: float) -> float:
    eps = 1e-12
    ln1p_ATR   = math.log1p(max(atr_usd, 0.0))
    ln1p_PM    = math.log1p(max(pm_vol_m, 0.0))
    ln1p_Float = math.log1p(max(float_m, 0.0))
    ln1p_Gap   = math.log1p(max(gap_pct, 0.0))
    inter_term = ln1p_Gap * ln1p_Float

    lnY = (STEP1_COEF["b0"]
           + STEP1_COEF["b_ATR"] * ln1p_ATR
           + STEP1_COEF["b_PM"]  * ln1p_PM
           + STEP1_COEF["b_Flt"] * ln1p_Float
           + STEP1_COEF["b_Int"] * inter_term
           + STEP1_COEF["b_Cat"] * float(catalyst_score))
    return float(math.exp(lnY))  # millions of shares

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    low  = pred_m * math.exp(-z * sigma_ln)
    high = pred_m * math.exp( z * sigma_ln)
    return low, high

def sanity_flags(mc_m, si_pct, atr_usd, pm_vol_m, float_m):
    flags = []
    if mc_m > 50000: flags.append("⚠️ Market Cap looks > $50B — is it in *millions*?")
    if float_m > 10000: flags.append("⚠️ Float > 10,000M — is it in *millions*?")
    if pm_vol_m > 1000: flags.append("⚠️ PM volume > 1,000M — is it in *millions*?")
    if si_pct > 100: flags.append("⚠️ Short interest > 100% — enter SI as percent (e.g., 25.0).")
    if atr_usd > 20: flags.append("⚠️ ATR > $20 — double-check units.")
    fr = (pm_vol_m / max(float_m, 1e-12)) if float_m > 0 else 0.0
    if float_m <= 1.0:
        if fr > 60: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is extreme even for micro-float.")
    elif float_m <= 5.0:
        if fr > 20: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is unusually high.")
    elif float_m <= 20.0:
        if fr > 10: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is high.")
    else:
        if fr > 3.0: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× may indicate unit mismatch.")
    return flags

# =========================================================
# STEP 2 — FT Probability (logit) — coefficients + function
# =========================================================
# logit(P(FT)) = a0 + a1*ln(PredVol_M) + a2*ln1p(FR_PM) + a3*ln1p(Gap%) + a4*Catalyst_score + (optional) a5*ln1p(MpB_PM)

# Defaults based on our fits; expose in sidebar to fine-tune if desired
st.sidebar.header("FT Probability — Coefficients")
a0  = st.sidebar.number_input("a0 (intercept)", value=-2.00, step=0.05, format="%.2f")
a1  = st.sidebar.number_input("a1 · ln(PredVol_M)", value=0.80, step=0.05, format="%.2f")
a2  = st.sidebar.number_input("a2 · ln1p(FR_PM)", value=1.20, step=0.05, format="%.2f")
a3  = st.sidebar.number_input("a3 · ln1p(Gap%)",   value=0.50, step=0.05, format="%.2f")
a4  = st.sidebar.number_input("a4 · Catalyst",      value=0.90, step=0.05, format="%.2f")
a5  = st.sidebar.number_input("a5 · ln1p(MpB_PM)",  value=0.00, step=0.05, format="%.2f")

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def predict_ft_prob(pred_vol_m: float, pm_vol_m: float, float_m: float,
                    gap_pct: float, catalyst_score: float, mpb_pm_frac: float | None) -> float:
    eps = 1e-12
    ln_pred   = math.log(max(pred_vol_m, eps))
    fr_pm     = pm_vol_m / max(float_m, eps) if float_m > 0 else 0.0
    ln1p_fr   = math.log1p(max(fr_pm, 0.0))
    ln1p_gap  = math.log1p(max(gap_pct, 0.0))
    z = a0 + a1*ln_pred + a2*ln1p_fr + a3*ln1p_gap + a4*float(catalyst_score)
    if mpb_pm_frac is not None and mpb_pm_frac >= 0:
        z += a5 * math.log1p(mpb_pm_frac)
    return sigmoid(z)

# =========================================================
# Tabs
# =========================================================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.1, 1.1])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        # Cap / PM inputs
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

        with c_top[2]:
            gap_pct  = st.number_input("Gap % (prior close → PMH)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            mpb_pm   = st.number_input("MpB% PM (→ 9:30)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            pm_rvol  = st.number_input("PM RVOL (optional)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        with c_top[3]:
            catalyst_points = st.slider("Catalyst score (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None)
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # === STEP 1: Predict Day Volume (M) ===
        pred_vol_m = predict_day_volume_m_trimmed(
            atr_usd=atr_usd,
            pm_vol_m=pm_vol_m,
            float_m=float_m,
            gap_pct=gap_pct,
            catalyst_score=catalyst_points
        )
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        # === STEP 2: FT Probability ===
        mpb_frac = mpb_pm/100.0 if mpb_pm and mpb_pm > 0 else None
        ft_prob = predict_ft_prob(
            pred_vol_m=pred_vol_m,
            pm_vol_m=pm_vol_m,
            float_m=float_m,
            gap_pct=gap_pct,
            catalyst_score=catalyst_points,
            mpb_pm_frac=mpb_frac
        )
        ft_odds_pct = round(100.0 * ft_prob, 1)

        # === Numeric points (unchanged logic for your composite score) ===
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
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)
        final_score = max(0.0, min(100.0, final_score))

        # Diagnostics you like
        pm_float_rot_x = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc = (pm_vol_m * pm_vwap) / mc_m * 100.0 if mc_m > 0 else 0.0

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,
            # 2-step predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),
            "FT_Prob_%": ft_odds_pct,
            # extra diagnostics
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            # raw inputs
            "_MCap_M": mc_m,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_Float_M": float_m,
            "_Gap_%": gap_pct,
            "_Catalyst": float(catalyst_points),
            "_PM_VWAP": pm_vwap
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – FT Prob {ft_odds_pct:.1f}% · PredVol {row['PredVol_M']:.2f}M"
        do_rerun()

    # === Preview card ===
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        cC.metric("FT Probability", f"{l.get('FT_Prob_%',0):.1f}%")
        cD.metric("Composite Score",   f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")

        d1, d2, d3, d4 = st.columns(4)
        d1.caption("DayVol CI (68% / 95%)")
        d1.write(f"{l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
                 f"{l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M")
        d2.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d3.metric("PM $Vol / MC",      f"{l.get('PM$ / MC_%',0):.1f}%")
        d4.metric("Numeric / Qual (%)", f"{l.get('Numeric_%',0):.1f} / {l.get('Qual_%',0):.1f}")

    with st.expander("🔎 Sanity sniff test (units)"):
        mc_m_dbg  = l.get("_MCap_M", 0.0)
        si_pct_dbg= l.get("_SI_%", 0.0)
        atr_dbg   = l.get("_ATR_$", 0.0)
        pm_m_dbg  = l.get("_PM_M", 0.0)
        flt_m_dbg = l.get("_Float_M", 0.0)
        flags = sanity_flags(mc_m_dbg, si_pct_dbg, atr_dbg, pm_m_dbg, flt_m_dbg)
        if flags:
            for f in flags: st.warning(f)
        else:
            st.success("Inputs look plausible at a glance.")

# =========================================================
# Ranking tab
# =========================================================
with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values(["FT_Prob_%","FinalScore"], ascending=[False, False]).reset_index(drop=True)

        cols_to_show = [
            "Ticker","FT_Prob_%","PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
            "Numeric_%","Qual_%","FinalScore","PM_FloatRot_x","PM$ / MC_%"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker",) else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "FT_Prob_%": st.column_config.NumberColumn("FT Probability %", format="%.1f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("CI68 High (M)", format="%.2f"),
                "Numeric_%": st.column_config.NumberColumn("Numeric_%", format="%.1f"),
                "Qual_%": st.column_config.NumberColumn("Qual_%", format="%.1f"),
                "FinalScore": st.column_config.NumberColumn("Composite Score", format="%.2f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
            }
        )

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        st.markdown("### 📋 Ranking (Markdown view)")
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

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
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
