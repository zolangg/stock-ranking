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
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- Simple scoring (unchanged from your app) ----------
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
    if score >= 70: return "High Odds"
    if score >= 55: return "Moderate Odds"
    if score >= 40: return "Low Odds"
    return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 85 else
            "A+"  if score_pct >= 80 else
            "A"   if score_pct >= 70 else
            "B"   if score_pct >= 60 else
            "C"   if score_pct >= 45 else "D")

# ---------- Our daily volume model (premarket) ----------
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y_M) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_frac) - 0.3878*ln(ATR_$)
    Inputs: mcap_m ($M), gap_pct (%), atr_usd ($)
    Output: Y in **millions of shares**
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gpF = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    lnY = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gpF + e) - 0.3878*math.log(atr + e)
    return float(math.exp(lnY))  # millions

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    return pred_m*math.exp(-z*sigma_ln), pred_m*math.exp(z*sigma_ln)

# ---------- FT probability model (premarket, LASSO+refit) ----------
def predict_ft_prob_premarket(float_m_shares, gap_pct, pm_vol_m, pm_vol_pct):
    """
    Uses ln(Gap frac), ln(PM Vol % of day), ln(Float M), ln(PM Vol M)
    Units:
      float_m_shares : millions
      gap_pct        : percent
      pm_vol_m       : millions (premarket)
      pm_vol_pct     : percent of expected day volume
    """
    e = 1e-6
    gap_f     = max(float(gap_pct or 0.0), 0.0) / 100.0
    ln_gapf   = math.log(gap_f + e)
    ln_pmvolF = math.log(max(float(pm_vol_pct or 0.0), 0.0)/100.0 + e)
    ln_float  = math.log(max(float(float_m_shares or 0.0), 0.0) + e)
    ln_pmvolM = math.log(max(float(pm_vol_m or 0.0), 0.0) + e)

    # coefficients (final refit)
    B0       = 0.0
    B_gapf   =  3.075110500928716
    B_pmvolF = -3.4243519203665262
    B_float  = -0.0718180225576189
    B_pmvolM =  0.5139398462880007

    lp = (B0 + B_gapf*ln_gapf + B_pmvolF*ln_pmvolF + B_float*ln_float + B_pmvolM*ln_pmvolM)
    if lp >= 0:
        p = 1.0/(1.0 + math.exp(-lp))
    else:
        elp = math.exp(lp); p = elp/(1.0 + elp)
    return max(0.0, min(1.0, p))

def ft_label(p: float) -> str:
    if p >= 0.85: return "Very High FT odds"
    if p >= 0.70: return "High FT odds"
    if p >= 0.55: return "Moderate FT odds"
    if p >= 0.40: return "Low FT odds"
    return "Very Low FT odds"

# ---------- Sidebar (weights, modifiers, sigma) ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ", 0.10, 1.50, 0.60, 0.01)

# Normalize weights
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

# ---------- Tabs ----------
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

        # PM + cap
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

        # Modifiers
        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # --- Daily volume prediction (millions) ---
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        # --- FT probability ---
        # Proxy PM Vol % if not entered elsewhere: 100 * PM / Pred
        pm_vol_pct = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_vol_pct = max(0.0, min(100.0, pm_vol_pct))

        ft_prob = predict_ft_prob_premarket(float_m, gap_pct, pm_vol_m, pm_vol_pct)
        ft_pct  = round(100.0 * ft_prob, 1)
        ft_text = ft_label(ft_prob)

        # --- Numeric block (unchanged) ---
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # --- Score combo (keep as-is) ---
        final_score = round(num_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)
        final_score = max(0.0, min(100.0, final_score))

        # --- Diagnostics ---
        pm_pct_of_pred   = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_float_rot_x   = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc  = 100.0 * (pm_vol_m * pm_vwap) / mc_m if mc_m > 0 else 0.0

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "FinalScore": final_score,

            # Predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),

            # FT
            "PM_Vol_%": round(pm_vol_pct, 1),
            "FT_Prob_%": ft_pct,
            "FT_Label": ft_text,

            # Raw inputs for debug
            "_MCap_M": mc_m, "_SI_%": si_pct, "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m, "_Float_M": float_m, "_Gap_%": gap_pct,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – FT {ft_pct:.1f}% · PredVol {row['PredVol_M']:.2f}M"
        do_rerun()

    # Preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Pred Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        cC.metric("FT Probability", f"{l.get('FT_Prob_%',0):.1f}%")
        cD.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")

        st.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M · "
            f"PM% of Pred: {l.get('PM_%_of_Pred',0):.1f}% · PM $/MC: {l.get('PM$ / MC_%',0):.1f}%"
        )
        st.info(l.get("FT_Label",""))

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df = df.sort_values(["FT_Prob_%","FinalScore"], ascending=[False, False]).reset_index(drop=True)

        cols_to_show = [
            "Ticker","PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
            "PM_Vol_%","PM_FloatRot_x","PM$ / MC_%",
            "FT_Prob_%","FT_Label",
            "FinalScore","Numeric_%","PM_%_of_Pred"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","FT_Label") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True, hide_index=True,
            column_config={
                "PredVol_M": st.column_config.NumberColumn("Pred Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("CI68 L", format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("CI68 R", format="%.2f"),
                "PM_Vol_%": st.column_config.NumberColumn("PM Vol % of Day", format="%.1f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $/MC %", format="%.1f"),
                "FT_Prob_%": st.column_config.NumberColumn("FT Prob %", format="%.1f"),
                "Numeric_%": st.column_config.NumberColumn("Numeric %", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Pred", format="%.1f"),
            }
        )

        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "ranking.csv","text/csv", use_container_width=True)
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
