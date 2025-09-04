import streamlit as st
import pandas as pd
import math

# ---------- Markdown table helper (no tabulate required) ----------
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    keep_cols = [c for c in cols if c in df.columns]
    if not keep_cols:
        return "| (no data) |\n| --- |"

    sub = df.loc[:, keep_cols].copy().fillna("")

    header = "| " + " | ".join(keep_cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep_cols)) + " |"
    lines = [header, sep]

    for _, row in sub.iterrows():
        cells = []
        for c in keep_cols:
            v = row[c]
            if isinstance(v, float):
                # show integers without .00, otherwise 2 decimals
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ---------- Compat rerun ----------
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

# ---------- Session state ----------
if "rows" not in st.session_state:
    st.session_state.rows = []
if "last" not in st.session_state:
    st.session_state.last = None
if "flash" not in st.session_state:
    st.session_state.flash = None

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
news_weight = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05, key="news_weight")
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05, key="dil_weight")

# Normalize blocks separately
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------- Mappers & labels ----------
def pts_rvol(x: float) -> int:
    cuts = [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_atr(x: float) -> int:
    cuts = [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_si(x: float) -> int:
    cuts = [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)]
    for th, p in cuts:
        if x < th: return p
    return 7

def pts_fr(pm_vol_m: float, float_m: float) -> int:
    # Rotation (×): pm_vol / float
    if float_m <= 0:
        return 1
    rot = pm_vol_m / float_m  # ×
    # thresholds equivalent to 1%, 3%, 10%, 25%, 50%, 100% as rotation
    cuts = [(0.01,1),(0.03,2),(0.10,3),(0.25,4),(0.50,5),(1.00,6)]
    for th, p in cuts:
        if rot < th: return p
    return 7

def pts_float(float_m: float) -> int:
    cuts = [(200,2),(100,3),(50,4),(35,5),(10,6)]
    if float_m <= 3: return 7
    for th, p in cuts:
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

# Strict model: expects MCap, Float, PM all in *millions*; SI in percent; ATR in dollars; Catalyst in [-1,1]
def predict_day_volume_m(mc_m: float, si_pct: float, atr_usd: float,
                         pm_vol_m: float, float_m: float, catalyst_points: float) -> float:
    """
    ln(Y) =
       5.597780
     - 0.015481*ln(MCap_M)
     + 1.007036*ln(1 + SI_%/100)
     - 1.267843*ln(1 + ATR_$)
     + 0.114066*ln(1 + PM_M / Float_M)
     + 0.074*Catalyst
    All *_M are in millions of shares or $ millions.
    """
    eps = 1e-12
    mc_m  = max(mc_m,  eps)                 # $ millions
    si_fr = max(si_pct, 0.0) / 100.0        # fraction
    atr   = max(atr_usd, 0.0)               # $
    pm_m  = max(pm_vol_m, 0.0)              # shares (millions)
    flt_m = max(float_m,  eps)              # shares (millions)
    fr    = pm_m / flt_m                    # rotation ×

    lnY = (
        5.597780
        - 0.015481 * math.log(mc_m)
        + 1.007036 * math.log1p(si_fr)      # ln(1+si)
        - 1.267843 * math.log1p(atr)        # ln(1+atr)
        + 0.114066 * math.log1p(fr)         # ln(1+fr)
        + 0.074 * float(catalyst_points)
    )
    return float(math.exp(lnY))              # shares (millions)

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # Form that clears on submit
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.1)
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=1.0)

        # Float / SI / PM volume (Target removed)
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
        qual_points = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None)
                )
                qual_points[crit["name"]] = choice[0]  # 1..7

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    # Scoring after submit
    if submitted and ticker:
        # Prediction replaces manual target volume
        pred_vol_m = predict_day_volume_m(mc_m, si_pct, atr_usd, pm_vol_m, float_m, catalyst_points)

        with st.expander("🔎 Prediction debug (check units)"):
    eps = 1e-12
    mc_m_dbg  = max(mc_m, eps)
    si_fr_dbg = max(si_pct, 0.0)/100.0
    atr_dbg   = max(atr_usd, 0.0)
    pm_m_dbg  = max(pm_vol_m, 0.0)
    flt_m_dbg = max(float_m, eps)
    fr_dbg    = pm_m_dbg / flt_m_dbg

    t0 = 5.597780
    t1 = -0.015481 * math.log(mc_m_dbg)
    t2 =  1.007036 * math.log1p(si_fr_dbg)
    t3 = -1.267843 * math.log1p(atr_dbg)
    t4 =  0.114066 * math.log1p(fr_dbg)
    t5 =  0.074    * float(catalyst_points)
    lnY = t0 + t1 + t2 + t3 + t4 + t5
    Y   = math.exp(lnY)

    st.write({
        "INPUTS (expected units)": {
            "MCap (millions $)": mc_m,
            "SI (%)": si_pct,
            "ATR ($)": atr_usd,
            "PM Volume (millions shares)": pm_vol_m,
            "Float (millions shares)": float_m,
            "Catalyst (-1..+1)": catalyst_points,
        },
        "Derived": {
            "FR = PM/Float (×)": fr_dbg
        },
        "ln-components": {
            "base": t0,
            "-0.015481 ln(MCap)": t1,
            "+1.007036 ln(1+SI_frac)": t2,
            "-1.267843 ln(1+ATR)": t3,
            "+0.114066 ln(1+FR)": t4,
            "+0.074 Catalyst": t5,
            "lnY total": lnY
        },
        "Predicted Y (millions shares)": Y
    })

        # Numeric points
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)

        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # Qualitative points
        qual_0_7 = sum(q_weights[c["name"]] * qual_points[c["name"]] for c in QUAL_CRITERIA)
        qual_pct = (qual_0_7/7.0)*100.0

        # Combine + modifiers
        combo_pct = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)

        # Diagnostics (updated: rotation ×, no $ metrics)
        pm_pct_of_pred = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x = pm_vol_m / float_m if float_m > 0 else 0.0

        # Save row
        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "OddsScore": final_score,
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,
            "PredVol_M": round(pred_vol_m, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row

        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # Preview card (legacy-safe)
    if st.session_state.last:
        st.markdown("---")
        l = st.session_state.last

        # Safe pulls for new/old rows
        pred_vol_m      = l.get("PredVol_M", None)
        pm_pct_of_pred  = l.get("PM_%_of_Pred", None)
        pm_rot_x        = l.get("PM_FloatRot_x", None)

        # Legacy fallback for rotation: convert old percent to × if present
        if pm_rot_x is None and isinstance(l.get("PM_Float_%"), (int, float)):
            try:
                pm_rot_x = round(float(l["PM_Float_%"]) / 100.0, 3)
            except Exception:
                pm_rot_x = None

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker", "—"))
        cB.metric("Numeric Block", f'{l["Numeric_%"]}%' if isinstance(l.get("Numeric_%"), (int, float)) else "—")
        cC.metric("Qual Block", f'{l["Qual_%"]}%' if isinstance(l.get("Qual_%"), (int, float)) else "—")
        cD.metric("Final Score",
                  f'{l["FinalScore"]} ({l.get("Level","—")})' if isinstance(l.get("FinalScore"), (int, float)) else "—")

        d1, d2, d3 = st.columns(3)
        d1.metric("Predicted Day Volume (M)", f'{pred_vol_m}' if isinstance(pred_vol_m, (int, float)) else "—")
        d2.metric("PM % of Prediction", f'{pm_pct_of_pred}%' if isinstance(pm_pct_of_pred, (int, float)) else "—")
        d3.metric("PM Float Rotation", f'{pm_rot_x}×' if isinstance(pm_rot_x, (int, float)) else "—")

        d1.caption("Model-predicted total day shares (millions).")
        d2.caption("PM volume ÷ predicted day volume × 100.")
        d3.caption("Premarket volume ÷ float (×).")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.sort_values("OddsScore", ascending=False).reset_index(drop=True)

        # Columns for new model
        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PredVol_M","PM_%_of_Pred","PM_FloatRot_x"
        ]

        # Normalize to avoid KeyError for legacy rows
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
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
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

        c1, c2 = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = None
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
