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

# ---------- Sidebar: weights & modifiers ----------
st.sidebar.header("Numeric Weights")
w_mc      = st.sidebar.slider("Market Cap (Millions $)", 0.0, 1.0, 0.05, 0.01, key="w_mc")
w_float   = st.sidebar.slider("Public Float (Millions)",  0.0, 1.0, 0.06, 0.01, key="w_float")
w_si      = st.sidebar.slider("Short Interest (%)",       0.0, 1.0, 0.12, 0.01, key="w_si")
w_gap     = st.sidebar.slider("Gap %",                    0.0, 1.0, 0.10, 0.01, key="w_gap")
w_atr     = st.sidebar.slider("ATR ($)",                  0.0, 1.0, 0.10, 0.01, key="w_atr")
w_rvol    = st.sidebar.slider("RVOL",                     0.0, 1.0, 0.22, 0.01, key="w_rvol")
w_pmvol   = st.sidebar.slider("Premarket Volume (M)",     0.0, 1.0, 0.05, 0.01, key="w_pmvol")
w_pmmc    = st.sidebar.slider("PM $Vol / MC (%)",         0.0, 1.0, 0.10, 0.01, key="w_pmmc")
w_fr      = st.sidebar.slider("PM Float Rotation (×)",    0.0, 1.0, 0.20, 0.01, key="w_fr")

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01, key=f"wq_{crit['name']}"
    )

st.sidebar.header("Modifiers")
news_weight     = st.sidebar.slider("Catalyst (× on value)", 0.0, 2.0, 1.0, 0.05, key="news_weight")
dilution_weight = st.sidebar.slider("Dilution (× on value)", 0.0, 2.0, 1.0, 0.05, key="dil_weight")

# Normalize numeric & qualitative blocks separately
num_sum = max(1e-9, w_mc + w_float + w_si + w_gap + w_atr + w_rvol + w_pmvol + w_pmmc + w_fr)
w_mc, w_float, w_si, w_gap, w_atr, w_rvol, w_pmvol, w_pmmc, w_fr = [
    w/num_sum for w in (w_mc, w_float, w_si, w_gap, w_atr, w_rvol, w_pmvol, w_pmmc, w_fr)
]
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights:
    q_weights[k] = q_weights[k] / qual_sum

# ---------- Numeric bucket scorers ----------
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

def pts_gap(x: float) -> int:
    for th, p in [(2,1),(5,2),(8,3),(12,4),(18,5),(25,6)]:
        if x < th: return p
    return 7

def pts_mcap(mcap_m: float) -> int:
    if mcap_m <= 100: return 7
    for th, p in [(20000,1),(10000,2),(5000,3),(2000,4),(1000,5),(500,6)]:
        if mcap_m > th: return p
    return 7

def pts_pmvol(pm_vol_m: float) -> int:
    for th, p in [(0.10,1),(0.30,2),(0.80,3),(2.00,4),(5.00,5),(10.00,6)]:
        if pm_vol_m < th: return p
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

def pts_pmmc(pm_dol_m: float, mcap_m: float) -> int:
    if mcap_m <= 0: return 1
    pct = 100.0 * pm_dol_m / mcap_m
    for th, p in [(0.10,1),(0.30,2),(0.80,3),(2.00,4),(5.00,5),(10.00,6)]:
        if pct < th: return p
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

# ---------- Sanity checks ----------
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

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")

    # Form that clears on submit (keeps your order & $ volume)
    with st.form("add_form", clear_on_submit=True):
        c_top = st.columns([1.2, 1.2, 1.0])

        # Column 1
        with c_top[0]:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = st.number_input("Market Cap (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            float_m  = st.number_input("Public Float (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            si_pct   = st.number_input("Short Interest (% of float)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            gap_pct  = st.number_input("Gap % (Open vs prior close)", min_value=0.0, value=0.0, step=0.1, format="%.1f")

        # Column 2
        with c_top[1]:
            atr_usd  = st.number_input("ATR ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            rvol     = st.number_input("RVOL", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_vol_m = st.number_input("Premarket Volume (Millions)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            pm_dol_m = st.number_input("Premarket Dollar Volume (Millions $)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

        # Column 3
        with c_top[2]:
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)
            dilution_points = st.slider("Dilution (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05)

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

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    # After submit
    if submitted and ticker:
        # === Numeric points ===
        p_mc     = pts_mcap(mc_m)
        p_float  = pts_float(float_m)
        p_si     = pts_si(si_pct)
        p_gap    = pts_gap(gap_pct)
        p_atr    = pts_atr(atr_usd)
        p_rv     = pts_rvol(rvol)
        p_pmvol  = pts_pmvol(pm_vol_m)
        p_fr     = pts_fr(pm_vol_m, float_m)
        p_pmmc   = pts_pmmc(pm_dol_m, mc_m)

        # Weighted (normalized) numeric block
        num_0_7 = (
            w_mc*p_mc + w_float*p_float + w_si*p_si + w_gap*p_gap +
            w_atr*p_atr + w_rvol*p_rv + w_pmvol*p_pmvol + w_pmmc*p_pmmc + w_fr*p_fr
        )
        num_pct = (num_0_7/7.0)*100.0

        # === Qualitative points (weighted 1..7) ===
        qual_0_7 = 0.0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] if isinstance(
                st.session_state.get(f"qual_{crit['name']}"), tuple
            ) else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]] * float(sel)
        qual_pct = (qual_0_7/7.0)*100.0

        # === Combine + modifiers (YOUR original 50/50 + sliders) ===
        combo_pct   = 0.5*num_pct + 0.5*qual_pct
        final_score = round(combo_pct + news_weight*catalyst_points*10 + dilution_weight*dilution_points*10, 2)
        final_score = max(0.0, min(100.0, final_score))

        # === Display/diagnostics we keep ===
        pm_float_rot_x  = pm_vol_m / float_m if float_m > 0 else 0.0
        pm_dollar_vs_mc = 100.0 * pm_dol_m / mc_m if mc_m > 0 else 0.0

        row = {
            "Ticker": ticker,
            "Odds": odds_label(final_score),
            "Level": grade(final_score),
            "Numeric_%": round(num_pct, 2),
            "Qual_%": round(qual_pct, 2),
            "FinalScore": final_score,
            "PM$ / MC_%": round(pm_dollar_vs_mc, 1),
            "PM_FloatRot_x": round(pm_float_rot_x, 3),
            # raw inputs (for export)
            "_MCap_M": mc_m,
            "_Gap_%": gap_pct,
            "_SI_%": si_pct,
            "_ATR_$": atr_usd,
            "_PM_M": pm_vol_m,
            "_PM_$M": pm_dol_m,
            "_Float_M": float_m,
            "_Catalyst": float(catalyst_points),
            "_Dilution": float(dilution_points),
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} – Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # ---------- Preview card (only the three you asked for, plus basics) ----------
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown("---")
        a, b, c, d, e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Level','—')})")
        c.metric("Odds", l.get("Odds","—"))
        d.metric("PM Float Rotation", f"{l.get('PM_FloatRot_x',0):.3f}×")
        e.metric("PM $Vol / MC", f"{l.get('PM$ / MC_%',0):.1f}%")

# ---------- Ranking tab ----------
with tab_rank:
    st.subheader("Current Ranking")

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        # sort by FinalScore highest first
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level",
            "Numeric_%","Qual_%","FinalScore",
            "PM$ / MC_%","PM_FloatRot_x"
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
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rotation ×", format="%.3f"),
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

        c1, _ = st.columns([0.25, 0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
