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

# ---------- Prediction model (fixed; SI as fraction; FR = PM/Float) ----------
def predict_day_volume_m(mc_m: float, si_pct: float, atr_usd: float,
                         pm_vol_m: float, float_m: float, catalyst_points: float) -> float:
    """
    ln(Y) = 5.307
            - 0.015481*ln(MCap_M)
            + 1.007036*ln(1 + SI_frac)      # SI_frac = SI_% / 100
            - 1.267843*ln(1 + ATR_$)
            + 0.114066*ln(1 + PM_M/Float_M)
            + 0.074*Catalyst
    Returns Y in millions of shares.
    """
    eps = 1e-12
    mc   = max(mc_m, eps)                 # $ millions
    si_f = max(si_pct, 0.0) / 100.0       # convert % -> fraction
    atr  = max(atr_usd, 0.0)              # $
    pm   = max(pm_vol_m, 0.0)             # shares (millions)
    flt  = max(float_m, eps)              # shares (millions)
    fr   = pm / flt                       # rotation ×

    lnY = (
        5.307
        - 0.015481 * math.log(mc)
        + 1.007036 * math.log1p(si_f) 
        - 1.267843 * math.log1p(atr)
        + 0.114066 * math.log1p(fr)
        + 0.074    * float(catalyst_points)
    )
    return float(math.exp(lnY))
    
def sanity_flags(mc_m, si_pct, atr_usd, pm_vol_m, float_m):
    flags = []
    # Unit sanity
    if mc_m > 50000: flags.append("⚠️ Market Cap looks > $50B — is it in *millions*?")
    if float_m > 10000: flags.append("⚠️ Float > 10,000M — is it in *millions*?")
    if pm_vol_m > 1000: flags.append("⚠️ PM volume > 1,000M — is it in *millions*?")
    if si_pct > 100: flags.append("⚠️ Short interest > 100% — enter SI as percent (e.g., 25.0).")
    if atr_usd > 20: flags.append("⚠️ ATR > $20 — double-check units.")

    # Adaptive FR threshold by float size
    fr = (pm_vol_m / max(float_m, 1e-12)) if float_m > 0 else 0.0
    if float_m <= 1.0:
        # micro-float: allow much higher rotations
        if fr > 60: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is extreme even for micro-float.")
    elif float_m <= 5.0:
        if fr > 20: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is unusually high.")
    elif float_m <= 20.0:
        if fr > 10: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× is high.")
    else:
        if fr > 3.0: flags.append(f"⚠️ FR=PM/Float = {fr:.2f}× may indicate unit mismatch.")
    return flags

def ln_terms_for_display(mc_m, si_pct, atr_usd, pm_vol_m, float_m, catalyst):
    eps = 1e-12
    mc = max(mc_m, eps)
    si_fr = max(si_pct, 0.0) / 100.0
    atr = max(atr_usd, 0.0)
    pm  = max(pm_vol_m, 0.0)
    flt = max(float_m, eps)
    fr  = pm / flt

    t0 = 5.597780
    t1 = -0.015481 * math.log(mc)
    t2 =  1.007036 * math.log1p(si_fr)
    t3 = -1.267843 * math.log1p(atr)
    t4 =  0.114066 * math.log1p(fr)
    t5 =  0.074    * float(catalyst)
    lnY = t0 + t1 + t2 + t3 + t4 + t5
    Y   = math.exp(lnY)

    return {
        "inputs_expected_units": {
            "MCap (millions $)": mc_m,
            "SI (%)": si_pct,
            "ATR ($)": atr_usd,
            "PM Volume (millions shares)": pm_vol_m,
            "Float (millions shares)": float_m,
            "Catalyst (-1..+1)": catalyst,
        },
        "derived": {
            "FR = PM/Float (×)": fr
        },
        "ln_components": {
            "base": t0,
            "−0.015481 ln(MCap)": t1,
            "+1.007036 ln(1+SI_frac)": t2,
            "−1.267843 ln(1+ATR)": t3,
            "+0.114066 ln(1+FR)": t4,
            "+0.074 Catalyst": t5,
            "lnY total": lnY
        },
        "Predicted Y (millions shares)": Y
    }

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # Form that clears on submit (YOUR original layout)
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.1)
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=1.0)

        # Float / SI / PM volume
        with c_top[1]:
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.5)
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.1)
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        # Cap & Modifiers
        with c_top[2]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=5.0)
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.subheader("Qualitative Context")

        q_cols = st.columns(3)
        # store chosen level 1..7 for each criterion in session_state via radio
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

    # After submit
    if submitted and ticker:
        # === Prediction ===
        pred_vol_m = predict_day_volume_m(mc_m, si_pct, atr_usd, pm_vol_m, float_m, catalyst_points)

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
            # stored as (idx, text) in radio; we saved only key, so read index:
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
        pm_dollar_vs_mc  = 100.0 * (pm_vol_m * pm_vwap) / mc_m if mc_m > 0 else 0.0  # keep in display

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
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            # Keep $Vol/MC (you said it's good)
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
      
            # ... your existing fields ...
            "PredVol_M": round(pred_vol_m, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
        
            # store raw inputs for debug / sanity
            "_MCap_M": mc_m,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_Float_M": float_m,
            "_Catalyst": float(catalyst_points),
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card (with ALL numbers you like) ----------
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
        d1.caption("Premarket volume ÷ float (unitless).")
        d2.metric("PM $Vol / MC",      f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("PM dollar volume ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption("Exponential model (PM/Float, SI, ATR, MCap, Catalyst).")
        d4.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d4.caption("PM volume ÷ predicted day volume × 100.")

    # --- Sanity sniff test ---
    with st.expander("🔎 Sanity sniff test (units & term contributions)"):
        mc_m_dbg  = l.get("_MCap_M", 0.0)
        si_pct_dbg= l.get("_SI_%", 0.0)
        atr_dbg   = l.get("_ATR_$", 0.0)
        pm_m_dbg  = l.get("_PM_M", 0.0)
        flt_m_dbg = l.get("_Float_M", 0.0)
        cat_dbg   = l.get("_Catalyst", 0.0)

        # Flags
        flags = sanity_flags(mc_m_dbg, si_pct_dbg, atr_dbg, pm_m_dbg, flt_m_dbg)
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("Inputs look plausible at a glance.")

        # Term-by-term breakdown
        details = ln_terms_for_display(mc_m_dbg, si_pct_dbg, atr_dbg, pm_m_dbg, flt_m_dbg, cat_dbg)
        st.write(details)

# ---------- Ranking tab ----------
with tab_rank:
    st.subheader("Current Ranking")

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "OddsScore" in df.columns:
            df = df.sort_values("OddsScore", ascending=False)
        df = df.reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PM_FloatRot_x","PM$ / MC_%",
            "PredVol_M","PM_%_of_Pred"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level") else 0.0
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
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
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
