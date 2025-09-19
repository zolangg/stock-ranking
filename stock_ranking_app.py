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

# ---------- Session state (safe defaults) ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}   # dict, not None
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- Qualitative criteria (YOUR original) ----------
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

# ---------- Sidebar: weights & modifiers (YOUR original) ----------
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

# --- Confidence (log-space) ---
st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
                             help="Estimated std dev of residuals in ln(volume). 0.60 ≈ typical for your sheet.")

# Normalize blocks separately
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------- Numeric bucket scorers (YOUR original logic) ----------
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
    # rotation × directly (not percent)
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

# ---------- New premarket day-volume model (millions out) ----------
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435
            + 0.1608*ln(MCap_M)
            + 0.6704*ln(Gap_frac)        # Gap_frac = Gap_% / 100
            - 0.3878*ln(ATR_$)
    Returns Y in **millions of shares**
    """
    e = 1e-6
    mc = max(float(mcap_m or 0.0), 0.0)
    gp = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)

    ln_y = (
        3.1435
        + 0.1608 * math.log(mc + e)
        + 0.6704 * math.log(gp + e)
        - 0.3878 * math.log(atr + e)
    )
    return math.exp(ln_y)  # already in millions

def ln_terms_for_display(mcap_m, gap_pct, atr_usd):
    e = 1e-6
    t0 = 3.1435
    t1 = 0.1608 * math.log(max(float(mcap_m or 0.0), 0.0) + e)
    t2 = 0.6704 * math.log(max(float(gap_pct or 0.0), 0.0) / 100.0 + e)
    t3 = -0.3878 * math.log(max(float(atr_usd or 0.0), 0.0) + e)
    lnY = t0 + t1 + t2 + t3
    Y   = math.exp(lnY)
    return {
        "ln_components": {
            "base": t0,
            "+0.1608 ln(MCap M)": t1,
            "+0.6704 ln(Gap frac)": t2,
            "−0.3878 ln(ATR $)": t3,
            "lnY total": lnY
        },
        "Predicted Y (millions shares)": Y
    }

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    """
    Given a point prediction in *millions* and log-space std dev (sigma_ln),
    return (low, high) in millions for a two-sided CI using multiplier exp(±z·σ).
    """
    if pred_m <= 0:
        return 0.0, 0.0
    low  = pred_m * math.exp(-z * sigma_ln)
    high = pred_m * math.exp( z * sigma_ln)
    return low, high

# ---------- FT model params (placeholders — keep as-is unless you paste trained values) ----------
_FT_INTERCEPT = -0.20
_FT_COEF = {
    'ln_gapf':    1.20,
    'ln_pmvol_f': 0.80,
    'ln_fr':      0.30,
    'ln_pmvol_m': 0.10,
    'ln_mcap':   -0.40,
    'ln_atr':    -0.30,
    'ln_float':  -0.20,
    'catalyst':   0.40,
}
_FT_MEAN  = {k: 0.0 for k in _FT_COEF.keys()}
_FT_SCALE = {k: 1.0 for k in _FT_COEF.keys()}

def _std(x, m, s):
    s = float(s) if s not in (None, 0.0) else 1.0
    return (x - float(m)) / s

def predict_ft_prob_premarket(float_m: float, mcap_m: float, atr_usd: float,
                              gap_pct: float, pm_vol_m: float,
                              pred_vol_m: float,
                              catalyst_flag: int = 0) -> float:
    """
    FT probability model (premarket).
    PM % of day = PM Vol (M) / Predicted Day Volume (M)
    """
    e = 1e-6
    ln_float   = math.log(max(float_m, e))
    ln_mcap    = math.log(max(mcap_m, e))
    ln_atr     = math.log(max(atr_usd, e))
    ln_gapf    = math.log(max(gap_pct, 0.0)/100.0 + e)
    ln_pmvol_m = math.log(max(pm_vol_m, 0.0) + 1.0)
    fr         = (pm_vol_m / max(float_m, e)) if float_m > 0 else 0.0
    ln_fr      = math.log(fr + 1.0)
    denom  = max(float(pred_vol_m or 0.0), 0.0)
    pm_frac = (pm_vol_m / denom) if denom > 0 else 0.0
    pm_frac = max(0.0, min(pm_frac, 5.0))
    ln_pmvol_f = math.log(pm_frac + 1.0)

    lp = _FT_INTERCEPT
    features = {
        'ln_float': ln_float,
        'ln_mcap': ln_mcap,
        'ln_atr': ln_atr,
        'ln_gapf': ln_gapf,
        'ln_pmvol_f': ln_pmvol_f,
        'ln_pmvol_m': ln_pmvol_m,
        'ln_fr': ln_fr,
        'catalyst': float(catalyst_flag),
    }
    for name, val in features.items():
        if name in _FT_COEF:
            z = _std(val, _FT_MEAN.get(name, 0.0), _FT_SCALE.get(name, 1.0))
            lp += _FT_COEF[name] * z

    if lp >= 0:
        p = 1.0 / (1.0 + math.exp(-lp))
    else:
        elp = math.exp(lp)
        p = elp / (1.0 + elp)
    return max(0.0, min(1.0, p))

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # Form that clears on submit
    with st.form("add_form", clear_on_submit=True):
        # === Your requested order ===
        col1, col2 = st.columns([1.2, 1.2])

        # First column: Ticker, Market Cap, Float, SI %, Gap %
        with col1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)",   min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (%)",        min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # Second column: ATR, RVOL, Premarket Volume, $Volume
        with col2:
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            rvol     = st.number_input("RVOL",    min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions shares)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_dol_m = st.number_input("Premarket $Volume (Millions $)",     min_value=0.0, value=0.0, step=0.01, format="%.2f")

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

        st.markdown("---")
        c_mod1, c_mod2 = st.columns(2)
        with c_mod1:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
        with c_mod2:
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    # After submit
    if submitted and ticker:
        # === Day volume prediction (M) ===
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)

        # Confidence bands (millions)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)    # ~68%
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)   # ~95%

        # === Numeric points ===
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # === Qualitative points (weighted 1..7) ===
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(st.session_state.get(f"qual_{crit['name']}"), tuple) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # === Combine + modifiers (YOUR original 50/50 + sliders) ===
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)
        final_score = max(0.0, min(100.0, final_score))

        # === Diagnostics to save ===
        pm_pct_of_pred   = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x   = pm_vol_m / float_m if float_m > 0 else 0.0
        # NOTE: now uses input $Volume directly (in millions $)
        pm_dollar_vs_mc  = 100.0 * pm_dol_m / mc_m if mc_m > 0 else 0.0

        # === FT Probability (uses PredVol_M as denominator) ===
        ft_prob = predict_ft_prob_premarket(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_vol_m=pm_vol_m,
            pred_vol_m=pred_vol_m,
            catalyst_flag=1 if catalyst_points != 0 else 0
        )
        ft_pct = round(100.0 * ft_prob, 1)
        ft_label = ("High FT" if ft_pct >= 70 else
                    "Moderate FT" if ft_pct >= 55 else
                    "Low FT" if ft_pct >= 40 else
                    "Very Low FT")
        ft_display = f"{ft_pct:.1f}% ({ft_label})"

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "OddsScore": final_score,
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,

            # Prediction fields
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),

            # $Vol/MC now from provided $Volume (M$)
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),

            # FT fields
            "FT": ft_display,
            "FT_Prob_%": ft_pct,
            "FT_Label": ft_label,

            # raw inputs for debug / sanity
            "_MCap_M": mc_m,
            "_Gap_%": gap_pct,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,   # <— NEW: store $Volume input
            "_Float_M": float_m,
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
        cA, cB, cC, cD, cE = st.columns(5)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Numeric Block", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qual Block",    f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score",   f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cE.metric("Odds", l.get("Odds","—"))

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d1.caption("Premarket volume ÷ float.")
        d2.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("Premarket dollar volume ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        d4.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d4.caption("PM volume ÷ predicted day volume × 100.")
        d5.metric("FT Probability", f"{l.get('FT_Prob_%',0):.1f}%")
        d5.caption(f"FT Label: {l.get('FT_Label','—')}")

# ---------- Ranking tab ----------
with tab_rank:
    st.subheader("Current Ranking")

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # Always sort by FinalScore highest first
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PM$ / MC_%",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM_%_of_Pred",
            "FT"
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
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "FT": st.column_config.TextColumn("FT (p/label)"),
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
