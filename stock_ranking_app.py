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
w_fr    = st.sidebar.slider("PM Float Rotation (%)", 0.0, 1.0, 0.45, 0.01, key="w_fr")
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
    if float_m <= 0:
        return 1
    pct = 100.0 * pm_vol_m / float_m
    cuts = [(1,1),(3,2),(10,3),(25,4),(50,5),(100,6)]
    for th, p in cuts:
        if pct < th: return p
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

# ---------- Prediction model (added) ----------
def predict_day_volume_m(float_m: float, mc_m: float, si_pct: float,
                         atr_usd: float, pm_vol_m: float,
                         catalyst: float) -> float:
    """
    ln(Y) =
       5.597780
     - 0.015481*ln(MCap)
     + 1.007036*ln(SI+1)
     - 1.267843*ln(ATR+1)
     + 0.114066*ln(1 + PM/Float)
     + 0.074*Catalyst
    """
    eps = 1e-9
    Float = max(float_m, eps)
    MCap  = max(mc_m,   eps)
    SI    = max(si_pct, 0.0)
    ATR   = max(atr_usd,0.0)
    PM    = max(pm_vol_m, eps)
    FR    = PM / Float

    ln_y = (
        5.597780
        - 0.015481 * math.log(MCap)
        + 1.007036 * math.log(SI + 1.0)
        - 1.267843 * math.log(ATR + 1.0)
        + 0.114066 * math.log(1.0 + FR)
        + 0.074    * float(catalyst)
    )
    return float(math.exp(ln_y))  # millions of shares

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # OPTION A: form that clears on submit
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Basics
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.1)
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=1.0)

        # Float / SI / PM volume + Target
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=5.0)
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.5)
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.1)
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.05, format="%.2f")

        # Price, Cap & Modifiers
        with c_top[2]:
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

        # Combine + modifiers (kept as in your app)
        combo_pct = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)

       # Diagnostics (rotation is unitless; no percent)
        pm_float_rot = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc_pct = 100.0 * (pm_vol_m * pm_vwap) / mc_m if mc_m > 0 else 0.0
        
        # NEW: model prediction + PM % of predicted
        pred_day_vol_m = predict_day_volume_m(
            float_m=float_m, mc_m=mc_m, si_pct=si_pct,
            atr_usd=atr_usd, pm_vol_m=pm_vol_m, catalyst=catalyst_points
        )
        pm_pred_pct = 100.0 * pm_vol_m / pred_day_vol_m if pred_day_vol_m > 0 else 0.0

        # Save row (unchanged fields + new fields)
        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "OddsScore": final_score,
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,
        
            # Keep
            "PM$ / MC_%": round(pm_dollar_vs_mc_pct, 1),
        
            # Change: rotation (unitless) instead of percent
            "PM_FloatRot": round(pm_float_rot, 3),
        
            # Prediction fields
            "Pred_DayVol_M": round(pred_day_vol_m, 2),
            "PM_Pred_%": round(pm_pred_pct, 1),
        }
        st.session_state.rows.append(row)
        st.session_state.last = row

        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

   # Preview card (robust to missing keys)
if st.session_state.last:
    st.markdown("---")
    l = st.session_state.last

    def fmt_num(x, dec=2):
        try:
            if x is None: return "—"
            x = float(x)
            if abs(x - round(x)) < 1e-9:
                return f"{int(round(x))}"
            return f"{x:.{dec}f}"
        except Exception:
            return str(x) if x is not None else "—"

    def fmt_pct(x, dec=1):
        try:
            if x is None: return "—"
            return f"{float(x):.{dec}f}%"
        except Exception:
            return "—"

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Last Ticker", l.get("Ticker", "—"))
    cB.metric("Numeric Block", fmt_pct(l.get("Numeric_%")))
    cC.metric("Qual Block", fmt_pct(l.get("Qual_%")))
    cD.metric("Final Score", f'{fmt_num(l.get("FinalScore"))} ({l.get("Level","—")})')

    d1, d2, d3, d4 = st.columns(4)
    rot_val = l.get("PM_FloatRot", None)
    rot_txt = f"{rot_val:.3f}×" if isinstance(rot_val, (int, float)) else "—"
    d1.metric("PM Float Rotation", rot_txt)
    d1.caption("Premarket volume ÷ float (unitless).")
    
    d2.metric("PM $Vol / MC", f'{l["PM$ / MC_%"]}%')
    d2.caption("PM dollar volume ÷ market cap × 100.")
    
    d3.metric("Predicted Day Vol (M)", f'{l["Pred_DayVol_M"]}')
    d3.caption("Exponential model prediction.")
    
    d4.metric("PM % of Predicted", f'{l["PM_Pred_%"]}%')
    d4.caption("Premarket volume ÷ predicted day volume × 100.")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.sort_values("OddsScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PM_FloatRot","PM$ / MC_%",
            "Pred_DayVol_M","PM_Pred_%"
        ]

        # --- Normalize to avoid KeyError for legacy rows ---
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level") else 0.0

        df = df[cols_to_show]

        # Editable table (unchanged)
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
                "PM_Target_%": st.column_config.NumberColumn("PM % of Target", format="%.1f"),
                "PM_FloatRot": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "Pred_DayVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PM_Pred_%": st.column_config.NumberColumn("PM % of Predicted", format="%.1f"),
            }
        )

        # ---- Row delete buttons (ADDED) ----
        st.markdown("#### Delete rows")
        del_cols = st.columns(4)
        for i, r in df.head(12).reset_index(drop=True).iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    do_rerun()

        # Download + markdown view (unchanged)
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
