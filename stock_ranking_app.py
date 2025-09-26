# app.py — Premarket Stock Ranking (Weight-only; DB-derived weights; $Vol inputs)
# ------------------------------------------------------------------------------
# • Upload workbook -> evaluates AUC-based weights for numeric metrics (no models).
# • Inputs organized in two numeric columns + a third column for Catalyst & Dilution.
# • Scoring = weighted numeric buckets (1..7 → %), + catalyst bonus, − dilution penalty.
# • Uses $ Premarket Dollar Volume (M) instead of VWAP anywhere.

import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket Stock Ranking", layout="wide")
st.title("Premarket Stock Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:.78rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 11.5px; color:#374151; }
  ul { margin: 4px 0 0 0; padding-left: 16px; }
  li { margin-bottom: 2px; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Session State ==============================
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {}  # evaluated from DB

# ============================== Helpers ==============================
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip().replace(" ", "").replace("’","").replace("'","")
    if s == "": return None
    if "," in s and "." not in s: s = s.replace(",", ".")
    else: s = s.replace(",", "")
    try: return float(s)
    except: return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"
    s = st.text_input(label, fmt.format(value), key=key, help=help)
    v = _parse_local_float(s)
    if v is None: return float(value)
    v = max(min_value, v)
    if max_value is not None: v = min(max_value, v)
    return float(v)

def _fmt_value(v: float) -> str:
    if v is None or not np.isfinite(v): return "—"
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 1:    return f"{v:.2f}"
    return f"{v:.3f}"

def df_to_markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
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
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("$","").replace("%","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns); nm = {c:_norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n: return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]: return c
    return None

# ============================== Numeric bucket scorers (1..7) ==============================
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
    return ("A++" if score_pct >= 92 else
            "A+"  if score_pct >= 85 else
            "A"   if score_pct >= 75 else
            "B"   if score_pct >= 65 else
            "C"   if score_pct >= 50 else "D")

# ============================== Sidebar: Show/Adjust Weights ==============================
st.sidebar.header("Weights")
st.sidebar.caption("Upload your workbook below to evaluate data-driven weights. You can still tweak them here.")

# Default fallback weights (from prior analysis on your sheet): RVOL 0.56, FR 0.25, SI 0.19
DEFAULT_W = {"RVOL":0.56, "PM Float Rotation (×)":0.25, "Short Interest (%)":0.19, "ATR ($)":0.0, "Float (M)":0.0}

# Start with evaluated weights if present, else defaults
currW = st.session_state.WEIGHTS.copy() if st.session_state.WEIGHTS else DEFAULT_W.copy()

# Manual fine-tune sliders
w_rvol  = st.sidebar.slider("RVOL", 0.0, 1.0, float(currW.get("RVOL", DEFAULT_W["RVOL"])), 0.01)
w_fr    = st.sidebar.slider("PM Float Rotation (×)", 0.0, 1.0, float(currW.get("PM Float Rotation (×)", DEFAULT_W["PM Float Rotation (×)"])), 0.01)
w_si    = st.sidebar.slider("Short Interest (%)", 0.0, 1.0, float(currW.get("Short Interest (%)", DEFAULT_W["Short Interest (%)"])), 0.01)
w_atr   = st.sidebar.slider("ATR ($)", 0.0, 1.0, float(currW.get("ATR ($)", DEFAULT_W["ATR ($)"])), 0.01)
w_float = st.sidebar.slider("Float (M) (penalty/bonus)", 0.0, 1.0, float(currW.get("Float (M)", DEFAULT_W["Float (M)"])), 0.01)

# Normalize to sum 1
w_sum = max(1e-9, w_rvol + w_fr + w_si + w_atr + w_float)
W = {
    "RVOL": w_rvol / w_sum,
    "PM Float Rotation (×)": w_fr / w_sum,
    "Short Interest (%)": w_si / w_sum,
    "ATR ($)": w_atr / w_sum,
    "Float (M)": w_float / w_sum,
}

# Modifiers for catalyst/dilution impact on final score (kept simple & transparent)
st.sidebar.header("Modifiers")
cat_mult = st.sidebar.slider("Catalyst weight (score pts per 1.0)", 0.0, 20.0, 10.0, 0.5)
dil_mult = st.sidebar.slider("Dilution penalty (score pts at 1.0)", 0.0, 30.0, 12.0, 0.5)

# ============================== Upload workbook → Evaluate Weights ==============================
st.markdown('<div class="section-title">Upload workbook (to evaluate numeric weights only)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Evaluate weights from workbook", use_container_width=True)

if learn_btn:
    if not uploaded:
        st.error("Upload a workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if merged_sheet not in xls.sheet_names:
                st.error(f"Sheet '{merged_sheet}' not found. Available: {xls.sheet_names}")
            else:
                raw = pd.read_excel(xls, merged_sheet)

                # pick columns
                col_ft    = _pick(raw, ["ft","FT"])
                col_atr   = _pick(raw, ["atr","atr $","atr$","atr (usd)","daily atr"])
                col_rvol  = _pick(raw, ["rvol @ bo","rvol","relative volume"])
                col_pmvol = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)"])
                col_float = _pick(raw, ["float m shares","public float (m)","float (m)","float"])
                col_si    = _pick(raw, ["si","short interest %","short float %","short interest (float) %"])

                if col_ft is None:
                    st.error("No 'FT' column found in merged sheet.")
                else:
                    df = pd.DataFrame()
                    df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce")

                    if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                    if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                    if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                    if col_float: df["float_m"]  = pd.to_numeric(raw[col_float], errors="coerce")
                    if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")

                    # derived FR
                    if {"pm_vol_m","float_m"}.issubset(df.columns):
                        df["fr_x"] = df["pm_vol_m"] / df["float_m"]

                    # AUC-based weights
                    from scipy.stats import rankdata
                    metric_cols = {
                        "RVOL": "rvol",
                        "ATR ($)": "atr_usd",
                        "Short Interest (%)": "si_pct",
                        "PM Float Rotation (×)": "fr_x",
                        "Float (M)": "float_m",
                    }

                    y = df["FT"].astype(float)
                    rows = []
                    for label, col in metric_cols.items():
                        if col not in df.columns:
                            rows.append({"Metric": label, "AUC": np.nan, "n": 0})
                            continue
                        x = pd.to_numeric(df[col], errors="coerce")
                        m = x.notna() & y.notna()
                        xv = x[m].to_numpy(); yv = y[m].astype(int).to_numpy()
                        if xv.size == 0 or yv.sum() == 0 or yv.sum() == yv.size:
                            rows.append({"Metric": label, "AUC": np.nan, "n": int(xv.size)})
                            continue
                        r = rankdata(xv)  # average ranks
                        n1 = yv.sum(); n0 = (1 - yv).sum()
                        s1 = r[yv == 1].sum()
                        auc = (s1 - n1*(n1+1)/2) / (n0*n1)
                        rows.append({"Metric": label, "AUC": float(auc), "n": int(xv.size)})

                    eff = pd.DataFrame(rows)
                    eff["gain"] = eff["AUC"] - 0.5
                    eff.loc[eff["gain"] < 0, "gain"] = 0.0
                    total_gain = eff["gain"].sum()
                    if total_gain <= 0:
                        # fallback equal weights across available metrics
                        k = max(1, (eff["gain"].notna()).sum())
                        eff["weight"] = 1.0 / k
                    else:
                        eff["weight"] = eff["gain"] / total_gain

                    # stash into session & reflect to sidebar
                    st.session_state.WEIGHTS = {
                        row["Metric"]: float(row["weight"])
                        for _, row in eff.iterrows() if np.isfinite(row["weight"])
                    }

                    st.success("Evaluated numeric weights from the workbook.")
                    st.dataframe(
                        eff[["Metric","AUC","n","weight"]].sort_values("weight", ascending=False),
                        use_container_width=True, hide_index=True
                    )
        except Exception as e:
            st.error(f"Weight evaluation failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

# ============================== Add Stock (two numeric cols + third for catalyst/dilution) ==============================
with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 0.9])

        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %",                    0.0, min_value=0.0, decimals=1)

        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL", 0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)

        with c3:
            # Catalyst weighted numeric (kept as requested)
            catalyst_points = st.slider("Catalyst (−1.0 … +1.0)", -1.0, 1.0, 0.0, 0.05,
                                        help="Positive news adds, negative news subtracts.")
            # Dilution slider stays here
            dilution_flag = st.slider("Dilution present? (0 = none, 1 = strong)", 0.0, 1.0, 0.0, 0.1,
                                      help="Continuous penalty applied to score.")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived diagnostics (shown in ranking table)
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Numeric points (1..7)
        p_rvol  = pts_rvol(rvol)
        p_atr   = pts_atr(atr_usd)
        p_si    = pts_si(si_pct)
        p_fr    = pts_fr(pm_vol_m, float_m)
        p_float = pts_float(float_m)

        # Weighted numeric % using current (normalized) weights
        # Map metric names -> points
        pts_map = {
            "RVOL": p_rvol,
            "PM Float Rotation (×)": p_fr,
            "Short Interest (%)": p_si,
            "ATR ($)": p_atr,
            "Float (M)": p_float,
        }
        # Weighted average of points
        num_0_7 = sum(W[k] * pts_map[k] for k in pts_map.keys())
        numeric_pct = (num_0_7 / 7.0) * 100.0

        # Modifiers
        score = numeric_pct + cat_mult * float(catalyst_points) - dil_mult * float(dilution_flag)
        final_score = float(np.clip(score, 0.0, 100.0))

        odds = odds_label(final_score)
        level = grade(final_score)
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if level in ("A++","A+","A") else
            '<span class="pill pill-warn">Constructive</span>' if level in ("B","C") else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )

        row = {
            "Ticker": ticker,
            "Odds": odds,
            "Level": level,
            "Numeric_%": round(numeric_pct, 2),
            "FinalScore": round(final_score, 2),
            "PM_FloatRot_x": round(fr_x, 3) if np.isfinite(fr_x) else "",
            "PM$ / MC_%": round(pmmc_pct, 1) if np.isfinite(pmmc_pct) else "",
            "Catalyst": round(float(catalyst_points), 2),
            "Dilution": round(float(dilution_flag), 2),
            # raw inputs (optional export/debug)
            "_MCap_M": mc_m, "_Float_M": float_m, "_SI_%": si_pct, "_ATR_$": atr_usd,
            "_PM_Vol_M": pm_vol_m, "_PM_$Vol_M": pm_dol_m, "_Gap_%": gap_pct,
            "VerdictPill": verdict_pill,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})")

    # Preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        c.metric("Grade", l.get('Level','—'))
        d.metric("Odds", l.get('Odds','—'))

        with st.expander("Premarket Snapshot", expanded=True):
            st.markdown(f"**Verdict:** {l.get('VerdictPill','—')}", unsafe_allow_html=True)
            st.markdown(
                f"- PM Float Rotation: **{_fmt_value(l.get('PM_FloatRot_x'))}×**  \n"
                f"- PM $Vol / MC: **{_fmt_value(l.get('PM$ / MC_%'))}%**  \n"
                f"- Catalyst: **{_fmt_value(l.get('Catalyst'))}** · Dilution: **{_fmt_value(l.get('Dilution'))}**",
                unsafe_allow_html=True
            )

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","Numeric_%","FinalScore","PM_FloatRot_x","PM$ / MC_%","Catalyst","Dilution"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level") else 0.0

        st.dataframe(
            df[cols_to_show], use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "Numeric_%": st.column_config.NumberColumn("Numeric %", format="%.2f"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PM_FloatRot_x": st.column_config.NumberColumn("PM Float Rot (×)", format="%.3f"),
                "PM$ / MC_%": st.column_config.NumberColumn("PM $Vol / MC %", format="%.1f"),
                "Catalyst": st.column_config.NumberColumn("Catalyst", format="%.2f"),
                "Dilution": st.column_config.NumberColumn("Dilution", format="%.2f"),
            }
        )

        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        st.markdown("### 📋 Ranking (Markdown view)")
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")
    else:
        st.info("Add at least one stock.")
