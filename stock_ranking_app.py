import json, math, subprocess, shutil
import streamlit as st
import pandas as pd
import os

# ---------- Page + light CSS ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .smallcap { color:#6b7280; font-size:12px; margin-top:-8px; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      div[role="radiogroup"] label p { font-size: 0.88rem; line-height:1.25rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      section[data-testid="stSidebar"] .stSlider { margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Premarket Stock Ranking")

# Resolve absolute paths so CWD changes don't break us
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
R_SCRIPT  = os.path.join(APP_DIR, "predict_bart_cli.R")
MODELS_DIR = os.path.join(APP_DIR, "models")

def _find_rscript():
    # Try PATH first
    path = shutil.which("Rscript")
    if path:
        return path
    # Try common locations
    for candidate in ("/usr/bin/Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript", "/opt/R/bin/Rscript"):
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "Rscript not found on PATH. Install system R (r-base). "
        "On Streamlit Cloud add '.streamlit/packages.txt' with 'r-base' and redeploy."
    )

# ---------- helpers ----------
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
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

# ---------- session ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- qualitative ----------
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

# ---------- sidebar ----------
st.sidebar.header("Numeric Weights")
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, 0.20, 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, 0.15, 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, 0.15, 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.45, 0.01)
w_float = st.sidebar.slider("Public Float (penalty/bonus)", 0.0, 1.0, 0.05, 0.01)
num_sum = max(1e-9, w_rvol + w_atr + w_si + w_fr + w_float)
w_rvol, w_atr, w_si, w_fr, w_float = [w/num_sum for w in (w_rvol, w_atr, w_si, w_fr, w_float)]

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}"
    )
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05)
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05)

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider(
    "Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
    help="Used for CI bands around predicted day volume."
)

# ---------- numeric scorers ----------
def call_bart_cli(rows: list[dict]) -> pd.DataFrame:
    rscript = _find_rscript()

    # Sanity checks to give clear errors
    if not os.path.exists(R_SCRIPT):
        raise RuntimeError(f"R script not found: {R_SCRIPT}")
    needed = [
        "bart_model_A_predDVol_ln.rds",
        "bart_model_A_predictors.rds",
        "bart_model_B_FT.rds",
        "bart_model_B_predictors.rds",
    ]
    for f in needed:
        fp = os.path.join(MODELS_DIR, f)
        if not os.path.exists(fp):
            raise RuntimeError(f"Missing model file: {fp}")

    payload = []
    for r in rows:
        payload.append({
            "PMVolM":  float(r.get("PMVolM", 0) or 0),
            "PMDolM":  float(r.get("PMDolM", 0) or 0),
            "FloatM":  float(r.get("FloatM", 0) or 0),
            "GapPct":  float(r.get("GapPct", 0) or 0),
            "ATR":     float(r.get("ATR", 0) or 0),
            "MCapM":   float(r.get("MCapM", 0) or 0),
            "Catalyst": float(r.get("Catalyst", 0) or 0),
        })

    # Pass explicit models directory via arg (and env for good measure)
    env = os.environ.copy()
    env["MODELS_DIR"] = MODELS_DIR

    p = subprocess.run(
        [rscript, R_SCRIPT, "--both", f"--models-dir={MODELS_DIR}"],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=APP_DIR,          # run from app root
        env=env,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "R CLI failed.\n"
            f"CWD: {APP_DIR}\n"
            f"Rscript: {rscript}\n"
            f"R script: {R_SCRIPT}\n"
            f"Models dir arg: {MODELS_DIR}\n"
            "STDERR:\n" + p.stderr.decode("utf-8", errors="ignore")
        )
    out = json.loads(p.stdout.decode("utf-8"))
    df = pd.DataFrame(out)
    if "PredVol_M" not in df.columns: df["PredVol_M"] = float("nan")
    if "FT_Prob"   not in df.columns: df["FT_Prob"]   = float("nan")
    return df

# ---------- UI ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Numeric Context</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.25, 1.25, 1.0])

        # LEFT
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # MIDDLE (reordered per your request)
        with c_top[1]:
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vwap  = st.number_input("PM VWAP ($)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

        # RIGHT
        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Qualitative Context</div>', unsafe_allow_html=True)

        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: x[1],
                    key=f"qual_{crit['name']}",
                    help=crit.get("help", None),
                )

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # compute PM $Vol (Millions) from inputs (shares in M × $)
        pm_dol_m = pm_vol_m * pm_vwap

        # ---- BART predictions via R (Model A + B) ----
        try:
            cli_df = call_bart_cli([{
                "PMVolM": pm_vol_m,
                "PMDolM": pm_dol_m,
                "FloatM": float_m,
                "GapPct": gap_pct,
                "ATR":    atr_usd,
                "MCapM":  mc_m,
                "Catalyst": 1 if catalyst_points > 0 else 0,
            }])
        except Exception as e:
            st.error(
                "Could not load BART models (.rds). Ensure R and required R packages are installed, "
                "and place these files in the app folder:\n\n"
                "• bart_model_A_predDVol_ln.rds\n• bart_model_A_predictors.rds\n"
                "• bart_model_B_FT.rds\n• bart_model_B_predictors.rds\n\n"
                "Also include predict_bart_cli.R.\n\nDetails:\n" + str(e)
            )
            st.stop()

        pred_vol_m = float(cli_df.loc[0, "PredVol_M"])
        ft_prob    = float(cli_df.loc[0, "FT_Prob"])

        # CI bands for predicted volume
        ci68_l = pred_vol_m * math.exp(-1.0 * sigma_ln)
        ci68_u = pred_vol_m * math.exp(+1.0 * sigma_ln)
        ci95_l = pred_vol_m * math.exp(-1.96 * sigma_ln)
        ci95_u = pred_vol_m * math.exp(+1.96 * sigma_ln)

        # ---- Numeric points ----
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)
        num_0_7 = (w_rvol*p_rvol) + (w_atr*p_atr) + (w_si*p_si) + (w_fr*p_fr) + (w_float*p_float)
        num_pct = (num_0_7/7.0)*100.0

        # ---- Qualitative (weighted 1..7) ----
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            raw = st.session_state.get(f"qual_{crit['name']}", (1,))
            sel = raw[0] if isinstance(raw, tuple) else raw
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # ---- Combine + modifiers ----
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(
            combo_pct + news_weight*(1 if catalyst_points>0 else 0)*10 + dilution_weight*(dilution_points)*10, 2
        )
        final_score = max(0.0, min(100.0, final_score))

        # ---- Diagnostics for display ----
        pm_float_rot_x  = (pm_vol_m / float_m) if float_m > 0 else 0.0
        pm_pct_of_pred  = 100.0 * pm_vol_m / pred_vol_m if pred_vol_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * (pm_dol_m) / mc_m if mc_m > 0 else 0.0

        ft_label = ("FT likely" if ft_prob >= 0.60 else
                    "Toss-up"   if ft_prob >= 0.40 else
                    "FT unlikely")

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

            "FT_Prob": round(ft_prob, 3),
            "FT_Label": ft_label,

            # Compact diagnostics
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),

            # raw
            "_MCap_M": mc_m, "_ATR_$": atr_usd, "_PM_M": pm_vol_m, "_PM$_M": pm_dol_m,
            "_Float_M": float_m, "_Gap_%": gap_pct, "_Catalyst": 1 if catalyst_points>0 else 0,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = (
            f"Saved {ticker} — {ft_label} ({ft_prob*100:.1f}%) · "
            f"Odds {row['Odds']} (Score {row['FinalScore']})"
        )
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
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
        d1.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d1.caption("Premarket volume ÷ predicted day volume × 100.")
        d2.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("PM dollar volume ÷ market cap × 100.")
        d3.metric("Numeric Block", f"{l.get('Numeric_%',0):.1f}%")
        d3.caption("Weighted buckets: RVOL / ATR / SI / FR / Float.")
        d4.metric("Qual Block", f"{l.get('Qual_%',0):.1f}%")
        d4.caption("Weighted radios: structure, levels, higher TF.")

with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","FT_Label","FT_Prob",
            "Numeric_%","Qual_%","FinalScore","Level",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U",
            "PM_%_of_Pred","PM$ / MC_%"
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
                "Numeric_%": st.column_config.NumberColumn("Numeric %", format="%.1f"),
                "Qual_%": st.column_config.NumberColumn("Qual %", format="%.1f"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "Level": st.column_config.TextColumn("Level"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
            }
        )

        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, c2, _ = st.columns([0.25, 0.25, 0.5])
        with c1:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "ranking.csv",
                "text/csv",
                use_container_width=True
            )
        with c2:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
