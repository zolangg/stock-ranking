# Premarket Stock Ranking — Data-driven evaluator (no arbitrary weights)
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import math
from typing import Optional

# ---------- Page ----------
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking (Data-Driven)")

# ---------- Global CSS ----------
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
      .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
      .stMetric label { font-size: 0.85rem; font-weight: 600; color:#374151;}
      .stMetric [data-testid="stMetricValue"] { font-size: 1.15rem; }
      .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
      .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
      .checklist pre { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Comma-friendly numeric input ----------
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"
    default_str = fmt.format(float(value))
    s = st.text_input(label, default_str, key=key, help=help)
    v = _parse_local_float(s)
    if v is None:
        st.caption('<span class="hint">Enter a number, e.g. 5,05</span>', unsafe_allow_html=True)
        return float(value)
    if v < min_value:
        st.caption(f'<span class="hint">Clamped to minimum: {fmt.format(min_value)}</span>', unsafe_allow_html=True)
        v = min_value
    if max_value is not None and v > max_value:
        st.caption(f'<span class="hint">Clamped to maximum: {fmt.format(max_value)}</span>', unsafe_allow_html=True)
        v = max_value
    if ("," in s) or (" " in s) or ("'" in s):
        st.caption(f'<span class="hint">= {fmt.format(v)}</span>', unsafe_allow_html=True)
    return float(v)

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

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =====================================================================
# DATA-DRIVEN PHB RULES  (from your FT vs Fail analysis)
# =====================================================================

PHB_RULES = {
    # Core “Look For”
    "FLOAT_MAX_M": 20.0,             # sweet: 5–12M
    "MCAP_MAX_M": 150.0,
    "ATR_MIN": 0.15,                 # 0.2–0.4 very typical
    "GAP_MIN": 70.0,                 # sweet ~100
    "GAP_SWEET_MAX": 180.0,
    "GAP_VIABLE_MAX": 280.0,         # thins above here
    "GAP_OUTLIER": 300.0,            # >300% = unproven tail in your sample

    "PM_DOLLAR_MIN": 7.0,            # $M
    "PM_DOLLAR_MAX": 30.0,           # $M (30–40M starts overcooked)
    "PM_SHARE_MIN_PCT": 10.0,        # PM shares as % of predicted day
    "PM_SHARE_MAX_PCT": 20.0,

    "RVOL_MIN": 100.0,               # × baseline
    "RVOL_MAX": 1500.0,              # × baseline (good)
    "RVOL_WARN_MAX": 3000.0,         # beyond = blowout risk
}

def _line(ok: bool, ok_msg: str, bad_msg: str, severe: bool = False):
    if ok:
        return f"✅ {ok_msg}", 2, 0   # +2 points for sweet/ok
    else:
        if severe:
            return f"🚫 {bad_msg}", -2, 1  # hard red: -2 points, counts red
        else:
            return f"⚠️ {bad_msg}", -1, 0  # soft warn: -1 point

def evaluate_premarket(float_m: float, mcap_m: float, atr_usd: float,
                       gap_pct: float, pm_vol_m: float, pm_dol_m: float,
                       rvol: float, pm_pct_of_pred: float,
                       catalyst_flag: bool, dilution_flag: bool):
    R = PHB_RULES
    lines = []
    score = 0
    reds = 0

    # Catalyst
    if catalyst_flag:
        lines.append("✅ Catalyst present (news/PR).")
        score += 1        # small bonus
    else:
        lines.append("⚠️ No clear catalyst (FT odds lower).")
        score -= 1

    # Float
    if float_m > 0 and float_m < 20:
        if 5 <= float_m <= 12:
            lines.append(f"✅ Float sweet spot 5–12M (yours {float_m:.2f}M).")
            score += 3    # strong positive (sweet spot)
        else:
            lines.append(f"✅ Float <20M (yours {float_m:.2f}M).")
            score += 2
    elif float_m >= 50:
        lines.append(f"🚫 Float very large (>{50}M; yours {float_m:.2f}M).")
        score -= 3; reds += 1
    else:
        lines.append(f"⚠️ Float high (≥20M; yours {float_m:.2f}M).")
        score -= 2

    # Market cap
    if mcap_m > 0 and mcap_m < 150:
        lines.append(f"✅ MarketCap <150M (yours {mcap_m:.1f}M).")
        score += 2
    elif mcap_m >= 500:
        lines.append(f"🚫 MarketCap huge (≥500M; yours {mcap_m:.1f}M).")
        score -= 3; reds += 1
    else:
        lines.append(f"⚠️ MarketCap elevated (≥150M; yours {mcap_m:.1f}M).")
        score -= 2

    # ATR
    if atr_usd >= 0.15:
        if 0.2 <= atr_usd <= 0.4:
            lines.append(f"✅ ATR in typical runner band 0.2–0.4 (yours {atr_usd:.2f}).")
            score += 2
        else:
            lines.append(f"✅ ATR ≥0.15 (yours {atr_usd:.2f}).")
            score += 1
    else:
        lines.append(f"⚠️ ATR thin (<0.15; yours {atr_usd:.2f}).")
        score -= 1

    # Gap %
    if gap_pct < R["GAP_MIN"]:
        lines.append(f"🚫 Gap small (<{R['GAP_MIN']:.0f}%; yours {gap_pct:.1f}%).")
        score -= 3; reds += 1
    elif R["GAP_MIN"] <= gap_pct <= R["GAP_SWEET_MAX"]:
        lines.append(f"✅ Gap sweet {R['GAP_MIN']:.0f}–{R['GAP_SWEET_MAX']:.0f}% (yours {gap_pct:.1f}%).")
        score += 3
    elif gap_pct <= R["GAP_VIABLE_MAX"]:
        lines.append(f"✅ Gap viable up to {R['GAP_VIABLE_MAX']:.0f}% (yours {gap_pct:.1f}%).")
        score += 1
    else:
        lines.append(f"⚠️ Gap outlier (> {R['GAP_VIABLE_MAX']:.0f}%; yours {gap_pct:.1f}%).")
        score -= 1
        if gap_pct > R["GAP_OUTLIER"]:
            lines.append(f"⚠️ >{R['GAP_OUTLIER']:.0f}% gaps are unproven in sample (exhaustion risk).")

    # Premarket $Vol
    if pm_dol_m < 3:
        lines.append(f"🚫 PM $Vol very thin (<$3M; yours ${pm_dol_m:.1f}M).")
        score -= 3; reds += 1
    elif 7 <= pm_dol_m <= 15:
        lines.append(f"✅ PM $Vol sweet $7–15M (yours ${pm_dol_m:.1f}M).")
        score += 3
    elif 5 <= pm_dol_m <= 22:
        lines.append(f"✅ PM $Vol viable (~$5–22M; yours ${pm_dol_m:.1f}M).")
        score += 1
    elif 30 < pm_dol_m <= 40:
        lines.append(f"⚠️ PM $Vol elevated (${pm_dol_m:.1f}M) — risk of front-loading.")
        score -= 1
    elif pm_dol_m > 40:
        lines.append(f"🚫 PM $Vol bloated (>${40}M; yours ${pm_dol_m:.1f}M).")
        score -= 3; reds += 1
    else:
        lines.append(f"⚠️ PM $Vol marginal (yours ${pm_dol_m:.1f}M).")
        score -= 1

    # PM shares as % of predicted day
    if pm_pct_of_pred <= 0:
        lines.append("⚠️ PM % of Predicted: cannot compute (missing inputs).")
        score -= 1
    elif 10 <= pm_pct_of_pred <= 20:
        lines.append(f"✅ PM shares sweet {10}–{20}% of predicted day (yours {pm_pct_of_pred:.1f}%).")
        score += 3
    elif 7 <= pm_pct_of_pred <= 25:
        lines.append(f"✅ PM shares viable {7}–{25}% (yours {pm_pct_of_pred:.1f}%).")
        score += 1
    elif pm_pct_of_pred < 5:
        lines.append(f"🚫 PM shares too thin (<5%; yours {pm_pct_of_pred:.1f}%).")
        score -= 3; reds += 1
    elif pm_pct_of_pred > 35:
        lines.append(f"🚫 PM shares front-loaded (>35%; yours {pm_pct_of_pred:.1f}%).")
        score -= 3; reds += 1
    else:
        lines.append(f"⚠️ PM shares outside sweet band (yours {pm_pct_of_pred:.1f}%).")
        score -= 1

    # RVOL
    if rvol <= 0:
        lines.append("⚠️ RVOL missing/zero.")
        score -= 1
    elif 100 <= rvol <= 1500:
        lines.append(f"✅ RVOL sweet 100–1500× (yours {rvol:.0f}×).")
        score += 2
    elif 70 <= rvol <= 2000:
        lines.append(f"✅ RVOL viable ~70–2000× (yours {rvol:.0f}×).")
        score += 1
    elif rvol < 50:
        lines.append(f"🚫 RVOL very low (<50×; yours {rvol:.0f}×).")
        score -= 2; reds += 1
    else:
        lines.append(f"⚠️ RVOL outside sweet band (yours {rvol:.0f}×).")
        score -= 1
    if rvol > PHB_RULES["RVOL_WARN_MAX"]:
        lines.append(f"⚠️ RVOL >{int(PHB_RULES['RVOL_WARN_MAX'])}× — blowout/exhaustion risk in sample.")

    # Dilution
    if dilution_flag:
        lines.append("⚠️ Dilution/overhang flagged.")
        score -= 2

    # Verdict / Normalization
    # theoretical max: around +17 (3+2+2+2+3+3+2+1 etc.). we normalize to 0–100.
    RAW_MAX = 17
    RAW_MIN = -12
    norm = (score - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100.0
    norm = max(0.0, min(100.0, norm))

    if reds >= 2:
        verdict = "Weak / Avoid"
    elif norm >= 75:
        verdict = "Very High Odds"
    elif norm >= 60:
        verdict = "High Odds"
    elif norm >= 45:
        verdict = "Moderate Odds"
    else:
        verdict = "Low Odds"

    return {
        "lines": lines,
        "raw_score": score,
        "score_pct": round(norm, 2),
        "reds": reds,
        "verdict": verdict
    }

# =====================================================================
# Models you already use (kept)
# =====================================================================

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

# Day-volume model (millions out) — unchanged
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """
    ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$)
    Returns Y in **millions of shares**
    """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z * sigma_ln), pred_m * math.exp(z * sigma_ln)

# =====================================================================
# Sidebar — only uncertainty input now
# =====================================================================

st.sidebar.header("Prediction Uncertainty")
sigma_ln = st.sidebar.slider(
    "Log-space σ (residual std dev)", 0.10, 1.50, 0.60, 0.01,
    help="Estimated std dev of residuals in ln(volume). 0.60 ≈ typical for your sheet."
)

# =====================================================================
# Tabs
# =====================================================================

tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.markdown('<div class="section-title">Premarket Inputs</div>', unsafe_allow_html=True)

    with st.form("add_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1.2, 1.2, 0.9])

        with col1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=3)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)

        with col2:
            rvol     = input_float("RVOL (× baseline)",    0.0, min_value=0.0, decimals=1)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=3)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%) [optional]", 0.0, min_value=0.0, decimals=1)

        with col3:
            catalyst = st.checkbox("Catalyst (news/PR)", value=False)
            dilution = st.checkbox("Dilution/ATM risk", value=False,
                                   help="Tick if there’s clear evidence of dilution/overhang.")

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Prediction
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)

        pm_pct_of_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else 0.0
        pm_float_rot_x = (pm_vol_m / float_m) if float_m > 0 else 0.0   # FYI metric

        # Data-driven evaluation
        eval_out = evaluate_premarket(
            float_m=float_m, mcap_m=mc_m, atr_usd=atr_usd,
            gap_pct=gap_pct, pm_vol_m=pm_vol_m, pm_dol_m=pm_dol_m,
            rvol=rvol, pm_pct_of_pred=pm_pct_of_pred,
            catalyst_flag=catalyst, dilution_flag=dilution
        )

        final_score = eval_out["score_pct"]
        level = grade(final_score)
        odds  = odds_label(final_score)

        row = {
            "Ticker": ticker,
            "Odds": odds,
            "Level": level,
            "FinalScore": round(final_score, 2),

            # Prediction fields
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "PredVol_CI95_L": round(ci95_l, 2),
            "PredVol_CI95_U": round(ci95_u, 2),
            "PM_%_of_Pred": round(pm_pct_of_pred, 1),

            # Ratios / diagnostics
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            "PM$ / MC_%": round((100.0*pm_dol_m/mc_m) if mc_m>0 else 0.0, 1),

            # Inputs to keep
            "_MCap_M": mc_m,
            "_Gap_%": gap_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM$_M": pm_dol_m,
            "_Float_M": float_m,
            "_RVOL_x": rvol,
            "_SI_%": si_pct,
            "_Catalyst": int(catalyst),
            "_Dilution": int(dilution),

            # Checklist
            "PremarketVerdict": eval_out["verdict"],
            "PremarketChecklist": "\n".join(eval_out["lines"]),
            "Reds": eval_out["reds"],
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        cA, cB, cC = st.columns(3)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        cC.metric("Odds", l.get("Odds","—"))

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        d1.caption("Premarket volume ÷ float.")
        d2.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")
        d2.caption("Premarket $Vol ÷ market cap × 100.")
        d3.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")
        d3.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} · "
            f"CI95: {l.get('PredVol_CI95_L',0):.2f}–{l.get('PredVol_CI95_U',0):.2f}"
        )
        d4.metric("PM % of Predicted", f"{l.get('PM_%_of_Pred',0):.1f}%")
        d4.caption("PM shares ÷ predicted day shares × 100.")
        d5.metric("Reds (hard flags)", f"{l.get('Reds',0)}")

        with st.expander("Premarket Checklist (data-driven)", expanded=True):
            st.markdown(f"**Verdict:** {l.get('PremarketVerdict','—')}")
            st.markdown(l.get("PremarketChecklist","(no checks)"))

# ---------- Ranking tab ----------
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level","FinalScore",
            "PremarketVerdict","Reds",
            "PM$ / MC_%","PredVol_M","PredVol_CI68_L","PredVol_CI68_U","PM_%_of_Pred",
            "PM_FloatRot_x"
        ]
        for c in cols_to_show:
            if c not in df.columns:
                df[c] = "" if c in ("Ticker","Odds","Level","PremarketVerdict") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Level"),
                "FinalScore": st.column_config.NumberColumn("Score", format="%.2f"),
                "PremarketVerdict": st.column_config.TextColumn("Verdict"),
                "Reds": st.column_config.NumberColumn("Hard Flags"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
                "PM_%_of_Pred": st.column_config.NumberColumn("PM % of Prediction", format="%.1f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.3f"),
            }
        )

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "ranking.csv",
            "text/csv",
            use_container_width=True
        )

        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
