import json, math, subprocess, shutil
import streamlit as st
import pandas as pd

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking (BART-powered)")

R_SCRIPT = "./predict_bart_cli.R"  # must be alongside this file

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
    try:
        st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- Numeric scorers (your originals) ----------
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

# ---------- Sidebar: weights & modifiers ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider("Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01)

# ---------- BART via R CLI ----------
def call_bart_cli(rows: list[dict]) -> pd.DataFrame:
    if not shutil.which("Rscript"):
        raise RuntimeError("Rscript not found on PATH.")
    payload = []
    for r in rows:
        payload.append({
            "PMVolM": float(r.get("PMVolM", 0) or 0),
            "PMDolM": float(r.get("PMDolM", 0) or 0),
            "FloatM": float(r.get("FloatM", 0) or 0),
            "GapPct": float(r.get("GapPct", 0) or 0),
            "ATR":    float(r.get("ATR", 0) or 0),
            "MCapM":  float(r.get("MCapM", 0) or 0),
            "Catalyst": float(r.get("Catalyst", 0) or 0),
        })
    p = subprocess.run(
        ["Rscript", R_SCRIPT, "--both"],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError("R CLI failed.\nSTDERR:\n" + p.stderr.decode("utf-8", errors="ignore"))
    out = json.loads(p.stdout.decode("utf-8"))
    df = pd.DataFrame(out)
    if "PredVol_M" not in df.columns: df["PredVol_M"] = float("nan")
    if "FT_Prob"   not in df.columns: df["FT_Prob"]   = float("nan")
    return df

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_dol_m = st.number_input("PM $ Volume (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        try:
            cli_df = call_bart_cli([{
                "PMVolM": pm_vol_m,
                "PMDolM": pm_dol_m,
                "FloatM": float_m,
                "GapPct": gap_pct,
                "ATR":    atr_usd,
                "MCapM":  mc_m,
                "Catalyst": 1 if catalyst_points > 0 else 0,  # binarize
            }])
        except Exception as e:
            st.error(
                "Could not load BART models (.rds). Ensure R and required R packages "
                "are installed, and place these files in the app folder:\n"
                "• bart_model_A_predDVol_ln.rds\n• bart_model_A_predictors.rds\n"
                "• bart_model_B_FT.rds\n• bart_model_B_predictors.rds\n"
                "Also include predict_bart_cli.R.\n\nDetails:\n" + str(e)
            )
            st.stop()

        pred_vol_m = float(cli_df.loc[0, "PredVol_M"])
        ft_prob    = float(cli_df.loc[0, "FT_Prob"])

        # Confidence bands (display-only; sigma from sidebar)
        ci68_l = pred_vol_m * math.exp(-1.0 * sigma_ln)
        ci68_u = pred_vol_m * math.exp(+1.0 * sigma_ln)
        ci95_l = pred_vol_m * math.exp(-1.96 * sigma_ln)
        ci95_u = pred_vol_m * math.exp(+1.96 * sigma_ln)

        # Scoring block (numeric only, like before)
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # Combine + modifiers
        combo_pct   = num_pct
        final_score = round(combo_pct + 10*(1 if catalyst_points>0 else 0)*news_weight + 10*(dilution_points)*dilution_weight, 2)
        final_score = max(0.0, min(100.0, final_score))

        pm_float_rot_x  = (pm_vol_m / float_m) if float_m > 0 else 0.0
        pm_pct_of_pred  = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        ft_label = ("FT likely" if ft_prob >= 0.6 else
                    "Toss-up"  if ft_prob >= 0.4 else
                    "FT unlikely")

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "FinalScore": final_score,

            # BART predictions
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),

            "FT_Prob": round(ft_prob, 3),
            "FT_Label": ft_label,

            # Diagnostics
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),

            # Raw inputs for reference
            "_MCap_M": mc_m,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,
            "_Float_M": float_m,
            "_Gap_%": gap_pct,
            "_Catalyst": 1 if catalyst_points>0 else 0,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – {ft_label} ({ft_prob*100:.1f}%) · Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # Preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown("---")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Pred Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        cC.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f} M"
        )
        cD.metric("FT Probability", f"{l.get('FT_Prob',0)*100:.1f}%")
        cD.caption(l.get("FT_Label","—"))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d2.metric("PM $Vol / MC",      f"{l.get('PM$ / MC_%',0):.1f}%")
        d3.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d4.metric("RVOL Weighted",     f"{w_rvol:.2f}")

with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","FT_Label","FT_Prob",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
            "PM_%_of_Pred","PM$ / MC_%","Numeric_%","FinalScore","Level"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","FT_Label","Level") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "FT_Label": st.column_config.TextColumn("FT Label"),
                "FT_Prob": st.column_config.NumberColumn("FT Prob", format="%.3f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "Numeric_%": st.column_config.NumberColumn("Numeric_%", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("FinalScore", format="%.2f"),
                "Level": st.column_config.TextColumn("Level"),
            }
        )

        st.markdown("### 📋 Ranking (Markdown view)")
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
