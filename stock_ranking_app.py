import streamlit as st
import pandas as pd

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

# ---------- Session state ----------
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------- Qualitative criteria ----------
QUAL_CRITERIA = [
    {"name": "GapStruct", "question": "Gap & Trend Development:",
     "options": ["Gap fully reversed: price loses >80% of gap.",
                 "Choppy reversal: price loses 50–80% of gap.",
                 "Partial retracement: price loses 25–50% of gap.",
                 "Sideways consolidation: gap holds, price within top 25% of gap.",
                 "Uptrend with deep pullbacks (>30% retrace).",
                 "Uptrend with moderate pullbacks (10–30% retrace).",
                 "Clean uptrend, only minor pullbacks (<10%)."],
     "weight": 0.15},
    {"name": "LevelStruct", "question": "Key Price Levels:",
     "options": ["Fails at all major support/resistance; cannot hold any key level.",
                 "Briefly holds/reclaims a level but loses it quickly; repeated failures.",
                 "Holds one support but unable to break resistance; capped below a key level.",
                 "Breaks above resistance but cannot stay; dips below reclaimed level.",
                 "Breaks and holds one major level; most resistance remains above.",
                 "Breaks and holds several major levels; clears most overhead resistance.",
                 "Breaks and holds above all resistance; blue sky."],
     "weight": 0.15},
    {"name": "Monthly", "question": "Monthly/Weekly Chart Context:",
     "options": ["Sharp, accelerating downtrend; new lows repeatedly.",
                 "Persistent downtrend; still lower lows.",
                 "Downtrend losing momentum; flattening.",
                 "Clear base; sideways consolidation.",
                 "Bottom confirmed; higher low after base.",
                 "Uptrend begins; breaks out of base.",
                 "Sustained uptrend; higher highs, blue sky."],
     "weight": 0.10},
]

# ---------- Sidebar weights ----------
st.sidebar.header("Numeric Weights")
w_mc      = st.sidebar.slider("Market Cap (M$)",       0.0, 1.0, 0.10, 0.01)
w_float   = st.sidebar.slider("Public Float (M)",      0.0, 1.0, 0.05, 0.01)
w_si      = st.sidebar.slider("Short Interest (%)",    0.0, 1.0, 0.15, 0.01)
w_gap     = st.sidebar.slider("Gap %",                 0.0, 1.0, 0.15, 0.01)
w_atr     = st.sidebar.slider("ATR ($)",               0.0, 1.0, 0.10, 0.01)
w_rvol    = st.sidebar.slider("RVOL",                  0.0, 1.0, 0.20, 0.01)
w_pmvol   = st.sidebar.slider("Premarket Volume (M)",  0.0, 1.0, 0.15, 0.01)
w_pmdol   = st.sidebar.slider("Premarket $Volume (M$)",0.0, 1.0, 0.10, 0.01)
w_fr      = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, 0.20, 0.01)
w_pmmc    = st.sidebar.slider("PM $Vol / MC (%)",      0.0, 1.0, 0.15, 0.01)
w_cat     = st.sidebar.slider("Catalyst",              0.0, 1.0, 0.10, 0.01)
w_dil     = st.sidebar.slider("Dilution",              0.0, 1.0, 0.10, 0.01)

st.sidebar.header("Qualitative Weights")
q_weights = {}
for crit in QUAL_CRITERIA:
    q_weights[crit["name"]] = st.sidebar.slider(
        crit["name"], 0.0, 1.0, crit["weight"], 0.01
    )

# Normalize weights
num_sum = max(1e-9, sum([w_mc, w_float, w_si, w_gap, w_atr, w_rvol, w_pmvol, w_pmdol, w_fr, w_pmmc, w_cat, w_dil]))
weights = {k: v/num_sum for k, v in {
    "mc": w_mc, "float": w_float, "si": w_si, "gap": w_gap, "atr": w_atr, "rvol": w_rvol,
    "pmvol": w_pmvol, "pmdol": w_pmdol, "fr": w_fr, "pmmc": w_pmmc, "cat": w_cat, "dil": w_dil
}.items()}
qual_sum = max(1e-9, sum(q_weights.values()))
for k in q_weights: q_weights[k] = q_weights[k]/qual_sum

# ---------- Bucketing functions ----------
def bucket(x, thresholds):
    for th, p in thresholds:
        if x < th: return p
    return 7

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 85 else
            "A+"  if score_pct >= 80 else
            "A"   if score_pct >= 70 else
            "B"   if score_pct >= 60 else
            "C"   if score_pct >= 45 else "D")

# ---------- Tabs ----------
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

with tab_add:
    st.subheader("Numeric Context")
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            ticker  = st.text_input("Ticker", "").strip().upper()
            mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01)
            float_m = st.number_input("Public Float (M)", 0.0, step=0.01)
            si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01)
            gap_pct = st.number_input("Gap %", 0.0, step=0.1)
        with c2:
            atr_usd = st.number_input("ATR ($)", 0.0, step=0.01)
            rvol    = st.number_input("RVOL", 0.0, step=0.01)
            pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01)
            pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01)
        with c3:
            catalyst = st.slider("Catalyst (−1…+1)", -1.0, 1.0, 0.0, 0.05)
            dilution = st.slider("Dilution (−1…+1)", -1.0, 1.0, 0.0, 0.05)
        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                st.radio(crit["question"],
                         options=list(enumerate(crit["options"], 1)),
                         format_func=lambda x: x[1],
                         key=f"qual_{crit['name']}")
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived metrics
        fr = pm_vol/float_m if float_m > 0 else 0
        pmmc = (pm_dol/mc_m*100) if mc_m > 0 else 0

        # Points
        p_mc     = bucket(mc_m, [(500,6),(1000,5),(2000,4),(5000,3),(10000,2),(20000,1)])
        p_float  = bucket(float_m, [(10,6),(35,5),(50,4),(100,3),(200,2)])
        p_si     = bucket(si_pct, [(2,1),(5,2),(10,3),(15,4),(20,5),(30,6)])
        p_gap    = bucket(gap_pct, [(2,1),(5,2),(8,3),(12,4),(18,5),(25,6)])
        p_atr    = bucket(atr_usd, [(0.05,1),(0.10,2),(0.20,3),(0.35,4),(0.60,5),(1.00,6)])
        p_rvol   = bucket(rvol, [(3,1),(4,2),(5,3),(7,4),(10,5),(15,6)])
        p_pmvol  = bucket(pm_vol, [(0.1,1),(0.3,2),(0.8,3),(2,4),(5,5),(10,6)])
        p_pmdol  = bucket(pm_dol, [(1,1),(3,2),(8,3),(20,4),(50,5),(100,6)])
        p_fr     = bucket(fr, [(0.01,1),(0.03,2),(0.10,3),(0.25,4),(0.50,5),(1.0,6)])
        p_pmmc   = bucket(pmmc, [(0.1,1),(0.3,2),(0.8,3),(2,4),(5,5),(10,6)])
        p_cat    = 4 + catalyst*3  # center around 4
        p_dil    = 4 - dilution*3  # inverse for dilution

        num_0_7 = (weights["mc"]*p_mc + weights["float"]*p_float + weights["si"]*p_si +
                   weights["gap"]*p_gap + weights["atr"]*p_atr + weights["rvol"]*p_rvol +
                   weights["pmvol"]*p_pmvol + weights["pmdol"]*p_pmdol +
                   weights["fr"]*p_fr + weights["pmmc"]*p_pmmc +
                   weights["cat"]*p_cat + weights["dil"]*p_dil)
        num_pct = (num_0_7/7.0)*100

        qual_0_7 = 0
        for crit in QUAL_CRITERIA:
            sel = st.session_state.get(f"qual_{crit['name']}", (1,))[0] \
                  if isinstance(st.session_state.get(f"qual_{crit['name']}"), tuple) \
                  else st.session_state.get(f"qual_{crit['name']}", 1)
            qual_0_7 += q_weights[crit["name"]]*float(sel)
        qual_pct = (qual_0_7/7.0)*100

        final_score = round(0.5*num_pct + 0.5*qual_pct,2)

        row = {"Ticker": ticker, "Numeric_%": num_pct, "Qual_%": qual_pct,
               "FinalScore": final_score, "Grade": grade(final_score)}
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} — {row['Grade']} ({row['FinalScore']})"
        do_rerun()

    # Preview
    l = st.session_state.last
    if l:
        st.markdown("---")
        cA,cB,cC,cD = st.columns(4)
        cA.metric("Last Ticker", l.get("Ticker","—"))
        cB.metric("Numeric Score", f"{l.get('Numeric_%',0):.2f}%")
        cC.metric("Qualitative Score", f"{l.get('Qual_%',0):.2f}%")
        cD.metric("Final Score", f"{l.get('FinalScore',0):.2f} ({l.get('Grade','—')})")

# ---------- Ranking tab ----------
with tab_rank:
    st.subheader("Current Ranking")
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           "ranking.csv","text/csv",use_container_width=True)
        if st.button("Clear Ranking", use_container_width=True):
            st.session_state.rows,st.session_state.last=[],{}
            do_rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add Stock** tab.")
