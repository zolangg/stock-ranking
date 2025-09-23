import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# =========================================================
# Page & Styles
# =========================================================
st.set_page_config(page_title="Premarket Stock Ranking — Rank Curves (Merged)", layout="wide")
st.title("Premarket Stock Ranking — Rank Curves on PMH BO Merged")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color:#374151; }
  ul { margin: 6px 0 0 0; padding-left: 18px; }
  li { margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Sidebar (settings)
# =========================================================
st.sidebar.header("Settings")

sb_bw       = st.sidebar.slider("Smoothing bandwidth (rank space)", 0.02, 0.12, 0.06, 0.005)
sb_grid     = st.sidebar.slider("Curve grid points", 101, 401, 201, 50)
sb_minrows  = st.sidebar.slider("Min rows per variable", 20, 200, 40, 5)
sb_support  = st.sidebar.slider("Min local support (sweet/risk)", 2, 20, 4, 1)
sb_lift     = st.sidebar.slider("Lift threshold over baseline (sweet/risk)", 0.01, 0.20, 0.06, 0.01)
sb_gapmerge = st.sidebar.slider("Sweet/risk merge gap (rank span)", 0.00, 0.10, 0.02, 0.005)
sb_minspan  = st.sidebar.slider("Min span (rank) to keep interval", 0.005, 0.10, 0.03, 0.005)

st.sidebar.markdown("---")
tail_strength = st.sidebar.slider("Tail penalty strength (exhaustion vars)", 0.0, 1.0, 0.70, 0.05)
st.sidebar.caption("Higher → stronger cut of lift at extreme ranks (Gap, RVOL, PM$Vol, PM$Vol/MC, FR).")
dilution_coeff = st.sidebar.slider("Dilution penalty (Δ log-odds if Dilution=1)", -2.0, 0.0, -0.90, 0.05)

st.sidebar.markdown("---")
show_baseline = st.sidebar.checkbox("Show baseline on curves", True)
plot_all_curves = st.sidebar.checkbox("Curves tab: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or pick a single variable",
    ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","fr_x","pmmc_pct","catalyst"]
)

# =========================================================
# Session State
# =========================================================
if "CURVES" not in st.session_state:   st.session_state.CURVES = {}   # {"curves":{var:model}, "base_p":float}
if "WEIGHTS" not in st.session_state:  st.session_state.WEIGHTS = {}  # var -> normalized weight
if "BASEP" not in st.session_state:    st.session_state.BASEP = 0.5
if "rows" not in st.session_state:     st.session_state.rows = []
if "last" not in st.session_state:     st.session_state.last = {}

# =========================================================
# Helpers
# =========================================================
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

def odds_label(score: float) -> str:
    if score >= 85: return "Very High Odds"
    elif score >= 70: return "High Odds"
    elif score >= 55: return "Moderate Odds"
    elif score >= 40: return "Low Odds"
    else: return "Very Low Odds"

def grade(score_pct: float) -> str:
    return ("A++" if score_pct >= 90 else
            "A+"  if score_pct >= 85 else
            "A"   if score_pct >= 75 else
            "B"   if score_pct >= 65 else
            "C"   if score_pct >= 50 else "D")

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

def _fmt_value(v: float) -> str:
    if v is None or not np.isfinite(v): return "—"
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 1:    return f"{v:.2f}"
    return f"{v:.3f}"

# =========================================================
# Predicted day volume (for PM % of Pred display)
# =========================================================
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """ ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$) """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

def ci_from_logsigma(pred_m: float, sigma_ln: float, z: float):
    if pred_m <= 0: return 0.0, 0.0
    return pred_m * math.exp(-z), pred_m * math.exp(z)

# =========================================================
# Upload (main pane) & Learn (MERGED ONLY)
# =========================================================
st.markdown('<div class="section-title">Upload workbook (sheet: PMH BO Merged)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")

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

def _tail_penalize(var_key: str, grid: np.ndarray, p: np.ndarray, p0: float, strength: float) -> np.ndarray:
    EXH = {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}
    if var_key not in EXH or strength <= 0: return p
    tail = np.clip((grid - 0.90) / 0.08, 0, 1)   # 0 until 0.90, ramps to 1 by ~0.98
    penalty = 1.0 - strength * tail
    return p0 + (p - p0) * penalty

def rank_smooth(x, y, bandwidth=0.06, grid_points=201, min_rows=40):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(float), y[mask].to_numpy(float)
    if x.size < min_rows: return None

    ranks = pd.Series(x).rank(pct=True).to_numpy()
    grid = np.linspace(0, 1, grid_points)
    h = float(bandwidth)
    p_grid = np.empty_like(grid)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((ranks - g)/h)**2)
        sw = w.sum()
        p_grid[i] = (w*y).sum()/sw if sw > 0 else np.nan
    p0 = float(y.mean())
    p_grid = pd.Series(p_grid).interpolate(limit_direction="both").fillna(p0).to_numpy()

    # local support (for sweet/risk)
    bins = np.linspace(0,1,51)
    counts, _ = np.histogram(ranks, bins=bins)
    bin_idx = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(counts)-1)
    support = counts[bin_idx]

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)
    return {"grid":grid, "p_grid":p_grid, "p0":p0, "support":support,
            "ranks":ranks, "n":x.size, "quantiles":{"pr":pr,"vals":vals}}

def auc_weight(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if y.nunique()!=2 or len(y) < 10: return 0.0
    r = x.rank(method="average")
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return 0.0
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    w = max(0.05, abs(float(auc)-0.5)*2.0)   # 0..1 with floor
    return float(w)

if st.button("Learn rank-curves", use_container_width=True):
    if not uploaded:
        st.error("Upload a workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if merged_sheet not in xls.sheet_names:
                st.error(f"Sheet '{merged_sheet}' not found. Available: {xls.sheet_names}")
            else:
                raw = pd.read_excel(xls, merged_sheet)

                # Map columns in merged sheet
                col_ft    = _pick(raw, ["ft"])
                col_gap   = _pick(raw, ["gap %","gap%","premarket gap"])
                col_atr   = _pick(raw, ["atr","atr $"])
                col_rvol  = _pick(raw, ["rvol @ bo","rvol"])
                col_pmvol = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)"])
                col_pmdol = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)"])
                col_float = _pick(raw, ["float m shares","public float (m)","float (m)"])
                col_mcap  = _pick(raw, ["marketcap m","market cap (m)","mcap m","mcap"])
                col_si    = _pick(raw, ["si","short interest %","short float %","short interest (float) %"])
                col_cat   = _pick(raw, ["catalyst","news","pr"])

                df = pd.DataFrame()
                if col_ft:    df["FT"]       = pd.to_numeric(raw[col_ft],    errors="coerce")
                if col_gap:   df["gap_pct"]  = pd.to_numeric(raw[col_gap],   errors="coerce")
                if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                if col_pmdol: df["pm_dol_m"] = pd.to_numeric(raw[col_pmdol], errors="coerce")
                if col_float: df["float_m"]  = pd.to_numeric(raw[col_float], errors="coerce")
                if col_mcap:  df["mcap_m"]   = pd.to_numeric(raw[col_mcap],  errors="coerce")
                if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")
                if col_cat:   df["catalyst"] = pd.to_numeric(raw[col_cat],   errors="coerce")

                if {"pm_vol_m","float_m"}.issubset(df.columns):
                    df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                    df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

                df = df[df["FT"].notna()]
                if df.empty:
                    st.error("No valid rows after cleaning FT. Check your sheet.")
                else:
                    y = df["FT"].astype(float)
                    var_list = [v for v in [
                        "gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m",
                        "float_m","mcap_m","fr_x","pmmc_pct","catalyst"
                    ] if v in df.columns]

                    curves, weights = {}, {}
                    for v in var_list:
                        # lower min_rows for binary catalyst
                        min_rows = sb_minrows if v != "catalyst" else max(10, int(sb_minrows/2))
                        model = rank_smooth(df[v], y, bandwidth=sb_bw, grid_points=sb_grid, min_rows=min_rows)
                        if model is not None:
                            curves[v] = model
                            weights[v] = auc_weight(df[v], y)

                    sw = sum(weights.values()) or 1.0
                    weights = {k: v/sw for k,v in weights.items()}

                    st.session_state.CURVES  = {"curves":curves, "base_p":float(y.mean())}
                    st.session_state.WEIGHTS = weights
                    st.session_state.BASEP   = float(y.mean())
                    st.success(f"Learned {len(curves)} curves · baseline P(FT) ≈ {y.mean():.2f}")
        except Exception as e:
            st.error(f"Learning failed: {e}")

# =========================================================
# Core funcs after learning
# =========================================================
def _value_to_rank(model: Dict[str,Any], x_val: float) -> float:
    q = model.get("quantiles", None)
    if q is None or not np.isfinite(x_val): return 0.5
    pr = np.asarray(q["pr"], dtype=float)
    vals = np.asarray(q["vals"], dtype=float)
    if x_val <= vals.min(): return 0.0
    if x_val >= vals.max(): return 1.0
    idx = np.searchsorted(vals, x_val)
    i0 = max(1, min(idx, len(vals)-1))
    x0, x1 = vals[i0-1], vals[i0]
    p0, p1 = pr[i0-1], pr[i0]
    if x1 == x0: return float(p0)
    t = (x_val - x0) / (x1 - x0)
    return float(p0 + t*(p1 - p0))

def _prob_from_model(model: Dict[str,Any], var_key: str, x_val: float, tail_strength: float) -> float:
    if model is None or not np.isfinite(x_val): return 0.5
    grid = model["grid"].copy()
    p_grid = model["p_grid"].copy()
    p0 = model["p0"]
    # tail penalty if needed
    p_grid = _tail_penalize(var_key, grid, p_grid, p0, tail_strength)
    r = _value_to_rank(model, x_val)
    j = int(np.clip(round(r * (len(grid)-1)), 0, len(grid)-1))
    p = float(p_grid[j])
    return float(np.clip(p, 1e-3, 1-1e-3))

def _intervals_with_gap(mask: np.ndarray, x: np.ndarray, max_gap: float, min_span: float) -> List[Tuple[float,float]]:
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    ivals, start, last = [], idx[0], idx[0]
    for k in idx[1:]:
        if (x[k] - x[last]) <= max_gap:
            last = k
        else:
            lo, hi = x[start], x[last]
            if (hi - lo) >= min_span: ivals.append((float(lo), float(hi)))
            start = last = k
    lo, hi = x[start], x[last]
    if (hi - lo) >= min_span: ivals.append((float(lo), float(hi)))
    return ivals

# =========================================================
# Tabs
# =========================================================
tab_add, tab_rank, tab_curves, tab_spots = st.tabs(["➕ Add Stock", "📊 Ranking", "📈 Curves", "🎯 Sweet Spots"])

# =========================================================
# Add Stock
# =========================================================
with tab_add:
    st.markdown('<div class="section-title">Add stock</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2,1.2,1.0])
        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL @ BO", 0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"])
            # Dilution slider sits right here (below PM $Vol input per your request)
            dilution_flag = st.select_slider("Dilution present?", options=[0,1], value=0,
                                             help="0 = none/negligible, 1 = present/overhang (penalizes log-odds)")
        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        sigma_ln = 0.60
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        curves  = (st.session_state.CURVES or {}).get("curves", {})
        weights = st.session_state.WEIGHTS or {}
        base_p  = float(st.session_state.BASEP or 0.5)

        var_vals = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "pm_vol_m": pm_vol_m, "pm_dol_m": pm_dol_m, "float_m": float_m, "mcap_m": mc_m,
            "fr_x": fr_x, "pmmc_pct": pmmc_pct, "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }

        parts = {}
        log_odds_sum = 0.0
        for k, x in var_vals.items():
            m = curves.get(k)
            p = _prob_from_model(m, k, x, tail_strength) if m else 0.5
            parts[k] = {"p": p, "x": x, "w": weights.get(k, 0.0)}
            z = math.log(p/(1-p))
            log_odds_sum += weights.get(k, 0.0) * z

        log_odds_sum += dilution_coeff * float(dilution_flag)
        log_odds_sum = float(np.clip(log_odds_sum, -12, 12))
        numeric_prob = 1.0 / (1.0 + math.exp(-log_odds_sum))

        final_score = float(np.clip(numeric_prob*100.0, 0.0, 100.0))
        odds_name = odds_label(final_score)
        level = grade(final_score)
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if final_score >= 70 else
            '<span class="pill pill-warn">Constructive</span>' if final_score >= 55 else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )

        def stance(var, p):
            # exhaustion vars rarely "Good" at extremes; keep neutral unless lift strong
            if p >= base_p + sb_lift and var not in {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}:
                return "Good"
            if p <= base_p - sb_lift:
                return "Risk"
            return "Caution"

        name_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL @ BO","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","catalyst":"Catalyst"
        }
        good, warn, risk = [], [], []
        for k, d in parts.items():
            nm = name_map.get(k, k)
            cat = stance(k, d["p"])
            line = f"{nm}: {_fmt_value(d['x'])} — {cat.lower()} (p≈{round(d['p']*100):d}%)"
            if cat=="Good": good.append(line)
            elif cat=="Risk": risk.append(line)
            else: warn.append(line)

        pm_pred_line = f"PM Vol % of Pred: {_fmt_value(pm_pct_pred)}%" if np.isfinite(pm_pct_pred) else "PM Vol % of Pred: —"

        row = {
            "Ticker": ticker,
            "Odds": odds_name,
            "Level": level,
            "FinalScore": round(final_score, 2),
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),
            "VerdictPill": verdict_pill,
            "GoodList": good, "WarnList": warn, "RiskList": risk,
            "PM_Pct_Pred_Line": pm_pred_line
        }
        st.session_state.rows.append(row)
        st.session_state.last = row
        st.success(f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})")

    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c,d,e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        c.metric("Grade", l.get("Level","—"))
        d.metric("Odds", l.get("Odds","—"))
        e.metric("PredVol (M)", f"{l.get('PredVol_M',0):.2f}")
        st.caption(f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M")

        with st.expander("Premarket Checklist", expanded=True):
            st.markdown(f"**Verdict:** {l.get('VerdictPill','—')}", unsafe_allow_html=True)
            st.markdown(f"<div class='mono'>{l.get('PM_Pct_Pred_Line','')}</div>", unsafe_allow_html=True)
            g,w,r = st.columns(3)
            def ul(items): return "<ul>"+"".join([f"<li>{x}</li>" for x in items])+"</ul>" if items else "<ul><li>—</li></ul>"
            with g: st.markdown("**Good**");    st.markdown(ul(l.get("GoodList",[])), unsafe_allow_html=True)
            with w: st.markdown("**Caution**"); st.markdown(ul(l.get("WarnList",[])), unsafe_allow_html=True)
            with r: st.markdown("**Risk**");    st.markdown(ul(l.get("RiskList",[])), unsafe_allow_html=True)

# =========================================================
# Ranking
# =========================================================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PredVol_M","PredVol_CI68_L","PredVol_CI68_U"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level") else 0.0
        st.dataframe(
            df[cols_to_show], use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "PredVol_CI68_L": st.column_config.NumberColumn("Pred Vol CI68 Low (M)",  format="%.2f"),
                "PredVol_CI68_U": st.column_config.NumberColumn("Pred Vol CI68 High (M)", format="%.2f"),
            }
        )
        st.download_button("Download CSV", df[cols_to_show].to_csv(index=False).encode("utf-8"),
                           "ranking.csv", "text/csv", use_container_width=True)
    else:
        st.info("Add at least one stock.")

# =========================================================
# Curves (plots)
# =========================================================
with tab_curves:
    st.markdown('<div class="section-title">Learned Rank Curves</div>', unsafe_allow_html=True)
    curves = (st.session_state.CURVES or {}).get("curves", {})
    base_p = float(st.session_state.BASEP or 0.5)
    if not curves:
        st.info("Upload + Learn first.")
    else:
        if plot_all_curves:
            # plot all learned curves in a grid
            learned_vars = list(curves.keys())
            n = len(learned_vars)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.8, nrows*3.2))
            axes = np.atleast_2d(axes)
            for i, var in enumerate(learned_vars):
                ax = axes[i//ncols, i % ncols]
                m = curves[var]
                grid = m["grid"].copy()
                p    = _tail_penalize(var, grid, m["p_grid"].copy(), m["p0"], tail_strength)
                ax.plot(grid, p, lw=2)
                if show_baseline:
                    ax.axhline(base_p, ls="--", lw=1, color="gray")
                ax.set_title(var)
                ax.set_xlabel("Rank")
                ax.set_ylabel("P(FT)")
            # hide unused axes
            for j in range(i+1, nrows*ncols):
                fig.delaxes(axes[j//ncols, j % ncols])
            st.pyplot(fig, clear_figure=True)
        else:
            var = sel_curve_var
            m = curves.get(var)
            if m is None:
                st.warning(f"No curve learned for '{var}'.")
            else:
                grid = m["grid"].copy()
                p    = _tail_penalize(var, grid, m["p_grid"].copy(), m["p0"], tail_strength)
                fig, ax = plt.subplots(figsize=(6.4, 3.2))
                ax.plot(grid, p, lw=2)
                if show_baseline:
                    ax.axhline(base_p, ls="--", lw=1, color="gray")
                ax.set_xlabel("Rank (percentile of variable)")
                ax.set_ylabel("P(FT | rank)")
                ax.set_title(f"{var} — rank-smoothed curve")
                st.pyplot(fig, clear_figure=True)

# =========================================================
# Sweet Spots (table)
# =========================================================
with tab_spots:
    st.markdown('<div class="section-title">Sweet Spots (lift over baseline, tail-aware)</div>', unsafe_allow_html=True)
    curves = (st.session_state.CURVES or {}).get("curves", {})
    base_p = float(st.session_state.BASEP or 0.5)
    if not curves or not uploaded:
        st.info("Upload + Learn first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            raw = pd.read_excel(xls, merged_sheet)

            col_ft    = _pick(raw, ["ft"])
            df = pd.DataFrame()
            if col_ft is None:
                st.error("No 'FT' column found in merged sheet.")
            else:
                df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce")
                # reuse the same mapping keys used to build curves
                keys = ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","catalyst"]
                mapping = {
                    "gap_pct": ["gap %","gap%","premarket gap"],
                    "atr_usd": ["atr","atr $"],
                    "rvol": ["rvol @ bo","rvol"],
                    "si_pct": ["si","short interest %","short float %","short interest (float) %"],
                    "pm_vol_m": ["pm vol (m)","premarket vol (m)","pm volume (m)"],
                    "pm_dol_m": ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)"],
                    "float_m": ["float m shares","public float (m)","float (m)"],
                    "mcap_m": ["marketcap m","market cap (m)","mcap m","mcap"],
                    "catalyst": ["catalyst","news","pr"]
                }
                for k in keys:
                    col = _pick(raw, mapping[k])
                    if col is not None:
                        df[k] = pd.to_numeric(raw[col], errors="coerce")
                if {"pm_vol_m","float_m"}.issubset(df.columns):
                    df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                    df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]

                df = df[df["FT"].notna()]

                rows = []
                for v, m in curves.items():
                    # skip if we don't have raw values for quantile back-mapping
                    if v not in df.columns: 
                        rows.append({"Variable": v, "Base p(FT)": round(base_p,3),
                                     "Sweet rank": "", "Sweet values": "", "Risk rank": "", "Risk values": ""})
                        continue
                    x = pd.to_numeric(df[v], errors="coerce")
                    y = pd.to_numeric(df["FT"], errors="coerce")
                    mask = x.notna() & y.notna()
                    x = x[mask].to_numpy(float); y = y[mask].to_numpy(float)
                    if x.size < sb_minrows:
                        rows.append({"Variable": v, "Base p(FT)": round(base_p,3),
                                     "Sweet rank": "", "Sweet values": "", "Risk rank": "", "Risk values": ""})
                        continue

                    # Use learned curve (already smoothed) with support from this dataset
                    grid  = m["grid"].copy()
                    p     = _tail_penalize(v, grid, m["p_grid"].copy(), m["p0"], tail_strength)
                    ranks = pd.Series(x).rank(pct=True).to_numpy()

                    # Local support along grid for THIS var using ranks
                    bins = np.linspace(0,1,51)
                    counts,_ = np.histogram(ranks, bins=bins)
                    bin_idx = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(counts)-1)
                    sup = counts[bin_idx]

                    peak = float(np.nanmax(p))
                    thr  = max(base_p + sb_lift, peak - 0.08)   # adaptive threshold
                    sweet_mask = (p >= thr) & (sup >= sb_support)
                    risk_mask  = (p <= base_p - sb_lift) & (sup >= sb_support)

                    s_rank = _intervals_with_gap(sweet_mask, grid, max_gap=sb_gapmerge, min_span=sb_minspan)
                    r_rank = _intervals_with_gap(risk_mask,  grid, max_gap=sb_gapmerge, min_span=sb_minspan)

                    # Map rank intervals to value intervals using variable quantiles from learned model
                    pr = m["quantiles"]["pr"]; vals = m["quantiles"]["vals"]
                    def r2v(lo, hi):
                        lo_v = float(np.interp(lo, pr, vals))
                        hi_v = float(np.interp(hi, pr, vals))
                        return lo_v, hi_v

                    s_vals = [r2v(lo,hi) for lo,hi in s_rank]
                    r_vals = [r2v(lo,hi) for lo,hi in r_rank]

                    rows.append({
                        "Variable": v,
                        "Base p(FT)": round(base_p,3),
                        "Sweet rank": "; ".join([f"[{lo:.2f},{hi:.2f}]" for lo,hi in s_rank]) or "",
                        "Sweet values": "; ".join([f"[{_fmt_value(lo)},{_fmt_value(hi)}]" for lo,hi in s_vals]) or "",
                        "Risk rank": "; ".join([f"[{lo:.2f},{hi:.2f}]" for lo,hi in r_rank]) or "",
                        "Risk values": "; ".join([f"[{_fmt_value(lo)},{_fmt_value(hi)}]" for lo,hi in r_vals]) or "",
                    })

                tbl = pd.DataFrame(rows)
                if tbl.empty:
                    st.info("No robust sweet/risk intervals. Loosen lift/support or increase sample.")
                else:
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                    st.download_button("Download sweet/risk table (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                                       "sweet_risk_table.csv","text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Sweet-spot analysis failed: {e}")
