import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List

# =================== Page & Styles ===================
st.set_page_config(page_title="Premarket Stock Ranking (Rank Curves, Merged)", layout="wide")
st.title("Premarket Stock Ranking — Rank-curves on PMH BO Merged")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; }
  .section-title { font-weight: 700; font-size: 1.05rem; letter-spacing:.2px; margin: 4px 0 8px 8px; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 16px 0; }
  .hint { color:#6b7280; font-size:12px; margin-top:-6px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:.75rem; }
  .pill-good { background:#e7f5e9; color:#166534; border:1px solid #bbf7d0; }
  .pill-warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
  .pill-bad  { background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color:#374151; }
  ul { margin: 4px 0; padding-left: 18px; }
  li { margin-bottom: 2px; }
  .debug-table th, .debug-table td { font-size:12px; }
</style>
""", unsafe_allow_html=True)

# =================== Session State ===================
if "CURVES" not in st.session_state: st.session_state.CURVES = {}
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {}
if "BASEP" not in st.session_state:   st.session_state.BASEP = 0.5
if "rows" not in st.session_state:    st.session_state.rows = []
if "last" not in st.session_state:    st.session_state.last = {}
if "flash" not in st.session_state:   st.session_state.flash = None

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# =================== Helpers ===================
def _parse_local_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "": return None
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try: return float(s)
    except Exception: return None

def input_float(label: str, value: float = 0.0, min_value: float = 0.0,
                max_value: Optional[float] = None, decimals: int = 2,
                key: Optional[str] = None, help: Optional[str] = None) -> float:
    fmt = f"{{:.{decimals}f}}"
    default_str = fmt.format(float(value))
    s = st.text_input(label, default_str, key=key, help=help)
    v = _parse_local_float(s)
    if v is None:
        st.caption('<span class="hint">Enter a number, e.g. 5,05</span>', unsafe_allow_html=True); return float(value)
    if v < min_value:
        st.caption(f'<span class="hint">Clamped to minimum: {fmt.format(min_value)}</span>', unsafe_allow_html=True)
        v = min_value
    if max_value is not None and v > max_value:
        st.caption(f'<span class="hint">Clamped to maximum: {fmt.format(max_value)}</span>', unsafe_allow_html=True)
        v = max_value
    if ("," in s) or (" " in s) or ("'" in s):
        st.caption(f'<span class="hint">= {fmt.format(v)}</span>', unsafe_allow_html=True)
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
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v - round(v)) > 1e-9 else f"{int(round(v))}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# =================== Prediction model (Day Vol) ===================
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

# =================== Upload & Learn (MERGED ONLY) ===================
st.markdown('<div class="section-title">1) Upload your Excel · Use sheet: PMH BO Merged</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx) containing 'PMH BO Merged'", type=["xlsx"], label_visibility="collapsed")
col_sheet = st.columns([1,1,2,2,2])
with col_sheet[0]:
    merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
with col_sheet[1]:
    sigma_ln = st.number_input("PredVol residual σ (log-space)", min_value=0.1, max_value=1.5, value=0.60, step=0.01)

learn = st.button("Learn rank-curves from uploaded sheet", use_container_width=True)

# =================== Column mapping helpers ===================
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = s.replace("$","").replace("%","").replace("’","").replace("'","")
    return s

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm = {c: _norm(c) for c in cols}
    # exact then contains
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if norm[c] == n: return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in norm[c]: return c
    return None

def build_numeric_from_merged(df_raw: pd.DataFrame) -> Dict[str, Any]:
    df = df_raw.copy()
    # map essentials
    col_ft    = _pick(df, ["ft"])
    col_gap   = _pick(df, ["gap %","gap%","premarket gap"])
    col_atr   = _pick(df, ["atr","atr $"])
    col_rvol  = _pick(df, ["rvol @ bo","rvol"])
    col_pmvol = _pick(df, ["pm vol (m)","premarket vol (m)","pm volume (m)"])
    col_pmdol = _pick(df, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)"])
    col_float = _pick(df, ["float m shares","public float (m)","float (m)"])
    col_mcap  = _pick(df, ["marketcap m","market cap (m)","mcap m","mcap"])
    col_si    = _pick(df, ["si","short interest %","short float %"])
    col_cat   = _pick(df, ["catalyst","news","pr"])  # optional/binary-ish

    out = pd.DataFrame()
    if col_ft:    out["FT"]        = pd.to_numeric(df[col_ft],    errors="coerce")
    if col_gap:   out["gap_pct"]   = pd.to_numeric(df[col_gap],   errors="coerce")
    if col_atr:   out["atr_usd"]   = pd.to_numeric(df[col_atr],   errors="coerce")
    if col_rvol:  out["rvol"]      = pd.to_numeric(df[col_rvol],  errors="coerce")
    if col_pmvol: out["pm_vol_m"]  = pd.to_numeric(df[col_pmvol], errors="coerce")
    if col_pmdol: out["pm_dol_m"]  = pd.to_numeric(df[col_pmdol], errors="coerce")
    if col_float: out["float_m"]   = pd.to_numeric(df[col_float], errors="coerce")
    if col_mcap:  out["mcap_m"]    = pd.to_numeric(df[col_mcap],  errors="coerce")
    if col_si:    out["si_pct"]    = pd.to_numeric(df[col_si],    errors="coerce")
    if col_cat:   out["catalyst"]  = pd.to_numeric(df[col_cat],   errors="coerce")

    # Derived
    if {"pm_vol_m","float_m"}.issubset(out.columns):
        out["fr_x"] = out["pm_vol_m"] / out["float_m"]
    if {"pm_dol_m","mcap_m"}.issubset(out.columns):
        out["pmmc_pct"] = 100.0 * out["pm_dol_m"] / out["mcap_m"]

    # Filter to rows with FT notna
    out = out[out["FT"].notna()]
    return {"data": out, "mapped_cols": {
        "FT": col_ft, "gap_pct": col_gap, "atr_usd": col_atr, "rvol": col_rvol,
        "pm_vol_m": col_pmvol, "pm_dol_m": col_pmdol, "float_m": col_float,
        "mcap_m": col_mcap, "si_pct": col_si, "catalyst": col_cat
    }}

# =================== Rank smoothing & stacking ===================
_EXHAUSTION_VARS = {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}

def _rank_smooth(x, y, bandwidth=0.06, grid_points=201, min_rows=30):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask].to_numpy(float), y[mask].to_numpy(float)
    if x.size < min_rows: return None
    ranks = pd.Series(x).rank(pct=True).to_numpy()
    grid = np.linspace(0, 1, grid_points)
    h = float(bandwidth)
    # kernel smooth
    p_grid = np.empty_like(grid)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((ranks - g)/h)**2)
        sw = w.sum()
        p_grid[i] = (w*y).sum()/sw if sw > 0 else np.nan
    # fill edges with baseline
    p0 = float(y.mean())
    p_grid = pd.Series(p_grid).interpolate(limit_direction="both").fillna(p0).to_numpy()
    # support via histogram
    bins = np.linspace(0,1,51)
    counts, _ = np.histogram(ranks, bins=bins)
    bin_idx = np.clip(np.searchsorted(bins, grid, side="right")-1, 0, len(counts)-1)
    support = counts[bin_idx]
    return {"grid":grid, "p_grid":p_grid, "p0":p0, "support":support, "ranks":ranks, "n":x.size}

def _tail_penalize(var_key, grid, p, p0):
    if var_key not in _EXHAUSTION_VARS: return p
    tail = np.clip((grid - 0.90) / 0.08, 0, 1)  # ramp from 0.90→0.98
    penalty = 1.0 - 0.7 * tail                  # cut lift by up to 70%
    return p0 + (p - p0) * penalty

def _to_prob(model: Dict[str,Any], x_val: float, var_key: str) -> float:
    if model is None or not np.isfinite(x_val): return 0.5
    # inverse-rank via ECDF
    ranks = model["ranks"]
    xv = np.nanpercentile(ranks, (pd.Series(ranks) <= np.inf).mean()*100)  # dummy to quiet warnings
    # robust rank: percentile of x among training x (ties → average)
    # we reconstruct by comparing to the original x distribution from ranks
    # Approximation: map x_val to its rank by empirical CDF from original training x stored in ranks proxy.
    # Simpler: project x_val onto grid via quantiles of the original x; we don't have original x, so fallback to nearest grid by median rank (neutral).
    # To avoid complexity, use nearest grid by assuming uniform rank if we cannot rebuild:
    # BUT we can estimate rank using model["grid"] and assume monotonic mapping; instead, use mid grid (0.5) if unknown.
    # Better: we store ECDF per var in CURVES later; for now, fallback:
    # This fallback will be replaced in learning phase where we store quantiles for inverse mapping.
    r = None
    # If we stored quantiles, use them:
    q = model.get("quantiles", None)
    if q is not None:
        # piecewise linear between quantiles: q["pr"] percentiles, q["vals"] values
        pr = np.asarray(q["pr"], dtype=float)  # in [0,1]
        vals = np.asarray(q["vals"], dtype=float)
        # monotone: find where x_val falls between vals and interpolate rank
        if np.isfinite(x_val):
            if x_val <= vals.min(): r = 0.0
            elif x_val >= vals.max(): r = 1.0
            else:
                idx = np.searchsorted(vals, x_val)  # right index
                i0 = max(1, min(idx, len(vals)-1))
                x0, x1 = vals[i0-1], vals[i0]
                p0r, p1r = pr[i0-1], pr[i0]
                if x1 == x0: r = float(p0r)
                else:
                    t = (x_val - x0) / (x1 - x0)
                    r = float(p0r + t*(p1r - p0r))
    if r is None:
        r = 0.5  # neutral fallback
    # evaluate kernel at r via nearest index
    grid = model["grid"]; p_grid = model["p_grid"]; p0 = model["p0"]
    # tail penalty on the fly
    p_grid_pen = _tail_penalize(var_key, grid, p_grid, p0)
    j = int(np.clip(round(r * (len(grid)-1)), 0, len(grid)-1))
    p = float(p_grid_pen[j])
    return float(np.clip(p, 1e-3, 1-1e-3))

def _intervals_with_gap(mask, x, max_gap=0.02, min_span=0.03):
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

def _fmt_value(v: float) -> str:
    if v is None or not np.isfinite(v): return "—"
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 1:    return f"{v:.2f}"
    return f"{v:.3f}"

# AUC-based weights (Mann–Whitney)
def _auc_weight(x, y) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if y.nunique() != 2: return 0.0
    # rank x
    r = x.rank(method="average")
    n1 = (y==1).sum()
    n0 = (y==0).sum()
    if n1 == 0 or n0 == 0: return 0.0
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    # convert to non-negative relative weight, min 0.05
    w = max(0.05, abs(float(auc) - 0.5) * 2.0)  # 0..1 → 0..1
    return float(w)

# Learn curves from merged
def learn_curves_merged(xls, sheet_name: str):
    raw = pd.read_excel(xls, sheet_name)
    built = build_numeric_from_merged(raw)
    data = built["data"]
    if "FT" not in data.columns or data["FT"].isna().all():
        raise ValueError("FT column missing or empty in merged sheet.")
    y = data["FT"].astype(float)

    var_list = [v for v in [
        "gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m",
        "float_m","mcap_m","fr_x","pmmc_pct","catalyst"
    ] if v in data.columns]

    curves = {}
    weights = {}
    for v in var_list:
        res = _rank_smooth(data[v], y, bandwidth=0.06, grid_points=201, min_rows=30 if v!="catalyst" else 10)
        if res is None: 
            continue
        # store quantiles for inverse map rank<->value
        xvals = pd.to_numeric(data[v], errors="coerce").dropna().to_numpy(float)
        pr = np.linspace(0,1,41)
        vals = np.quantile(xvals, pr)
        res["quantiles"] = {"pr": pr, "vals": vals}
        curves[v] = res
        weights[v] = _auc_weight(data[v], y)

    # normalize weights to sum=1
    sw = sum(weights.values()) or 1.0
    weights = {k: v/sw for k,v in weights.items()}

    st.session_state.CURVES = {"curves": curves, "base_p": float(y.mean())}
    st.session_state.WEIGHTS = weights
    st.session_state.BASEP = float(y.mean())
    return {"curves": curves, "weights": weights, "base_p": float(y.mean()), "mapped_cols": built["mapped_cols"]}

# =================== Learn click ===================
if learn:
    if not uploaded:
        st.error("Upload an Excel first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if merged_sheet not in xls.sheet_names:
                st.error(f"Sheet '{merged_sheet}' not found. Available: {xls.sheet_names}")
            else:
                res = learn_curves_merged(xls, merged_sheet)
                st.success(f"Learned curves for {len(res['curves'])} variables · base P(FT) ≈ {res['base_p']:.2f}")
                with st.expander("Detected Column Mapping (optional)", expanded=False):
                    st.json(res["mapped_cols"], expanded=False)
        except Exception as e:
            st.error(f"Learning failed: {e}")

# =================== Tabs ===================
tab_add, tab_rank, tab_sweet = st.tabs(["➕ Add Stock", "📊 Ranking", "📊 Variable Sweet Spots"])

# =================== Add Stock ===================
with tab_add:
    st.markdown('<div class="section-title">2) Add stock (numeric inputs)</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            ticker   = st.text_input("Ticker", "").strip().upper()
            mc_m     = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
            float_m  = input_float("Public Float (Millions)",  0.0, min_value=0.0, decimals=2)
            si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
            gap_pct  = input_float("Gap %", 0.0, min_value=0.0, decimals=1)
        with c2:
            atr_usd  = input_float("ATR ($)", 0.0, min_value=0.0, decimals=2)
            rvol     = input_float("RVOL @ BO",    0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"])
            # Dilution slider under PM $Vol (per your spec): place it here in the same form column
            dilution_flag = st.select_slider("Dilution present?", options=[0,1], value=0, help="0 = none/negligible, 1 = present/overhang")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived real-time
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted Day Vol & PM% of Predicted
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        ci95_l, ci95_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.96)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        # Evaluate curves
        curves = (st.session_state.CURVES or {}).get("curves", {})
        weights= st.session_state.WEIGHTS or {}
        base_p = float(st.session_state.BASEP or 0.5)

        def p_of(var_key, x):
            m = curves.get(var_key)
            if m is None or x is None or not np.isfinite(x): return 0.5, "neutral: no curve or bad input"
            try:
                p = _to_prob(m, float(x), var_key)
                return p, f"ok: {p:.3f}"
            except Exception as e:
                return 0.5, f"neutral: error {e}"

        var_vals = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "pm_vol_m": pm_vol_m, "pm_dol_m": pm_dol_m, "float_m": float_m, "mcap_m": mc_m,
            "fr_x": fr_x, "pmmc_pct": pmmc_pct, "pm_pct_pred": pm_pct_pred,  # pm_pct_pred will be neutral if no curve
            "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }

        # per-variable probabilities + weighted odds stacking
        log_odds_sum = 0.0
        parts = {}
        for k, x in var_vals.items():
            if k == "pm_pct_pred":
                # usually not learned from merged; we still show in checklist (neutral if no curve)
                continue
            p, note = p_of(k, x)
            parts[k] = {"p": p, "note": note, "x": x, "w": weights.get(k, 0.0)}
            z = math.log(p/(1-p))
            log_odds_sum += weights.get(k, 0.0) * z

        # Dilution penalty (0 or 1): negative effect on log-odds
        DILUTION_LOGODDS = -0.75  # ~ -15 to -20 ppt around mid depending on stack
        log_odds_sum += DILUTION_LOGODDS * float(dilution_flag)

        # Convert to probability
        numeric_prob = 1.0 / (1.0 + math.exp(-log_odds_sum))
        numeric_pct  = 100.0 * numeric_prob

        # Final score purely from numeric method
        final_score = float(max(0.0, min(100.0, numeric_pct)))
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if final_score >= 70 else
            '<span class="pill pill-warn">Constructive</span>' if final_score >= 55 else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )
        odds_name = odds_label(final_score)
        level = grade(final_score)

        # Checklist stances
        def stance(var, p):
            p0 = base_p
            # exhaustion extra rule: if var value is in extreme upper tail, downgrade
            is_exhaust = var in _EXHAUSTION_VARS
            if p >= p0 + 0.06 and (not is_exhaust):
                return "Good"
            if p <= p0 - 0.06:
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

        # Also show PM Vol % of Predicted (info only in checklist)
        pm_pred_line = f"PM Vol % of Pred: {_fmt_value(pm_pct_pred)}%" if np.isfinite(pm_pct_pred) else "PM Vol % of Pred: —"

        row = {
            "Ticker": ticker,
            "Odds": odds_name,
            "Level": level,
            "FinalScore": round(final_score, 2),

            # Prediction fields in table
            "PredVol_M": round(pred_vol_m, 2),
            "PredVol_CI68_L": round(ci68_l, 2),
            "PredVol_CI68_U": round(ci68_u, 2),

            # Checklist bundles
            "VerdictPill": verdict_pill,
            "GoodList": good,
            "WarnList": warn,
            "RiskList": risk,
            "PM_Pct_Pred_Line": pm_pred_line,
        }

        st.session_state.rows.append(row)
        st.session_state.last = row
        st.session_state.flash = f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']})"
        do_rerun()

    # Preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c,d,e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        c.metric("Grade", l.get("Level","—"))
        d.metric("Odds", l.get("Odds","—"))
        e.metric("Predicted Day Vol (M)", f"{l.get('PredVol_M',0):.2f}")

        c1, c2 = st.columns(2)
        c1.caption(
            f"CI68: {l.get('PredVol_CI68_L',0):.2f}–{l.get('PredVol_CI68_U',0):.2f} M · "
            f"CI95 approx: ±1.96σ"
        )
        with st.expander("Premarket Checklist", expanded=True):
            st.markdown(f"**Verdict:** {l.get('VerdictPill','—')}", unsafe_allow_html=True)
            st.markdown(f"<div class='mono'>{l.get('PM_Pct_Pred_Line','')}</div>", unsafe_allow_html=True)
            g,w,r = st.columns(3)
            def ul(items):
                if not items: return "<ul><li><span class='hint'>None</span></li></ul>"
                return "<ul>" + "".join([f"<li>{x}</li>" for x in items]) + "</ul>"
            with g: st.markdown("**Good**");    st.markdown(ul(l.get("GoodList",[])),  unsafe_allow_html=True)
            with w: st.markdown("**Caution**"); st.markdown(ul(l.get("WarnList",[])),  unsafe_allow_html=True)
            with r: st.markdown("**Risk**");    st.markdown(ul(l.get("RiskList",[])),  unsafe_allow_html=True)

# =================== Ranking tab ===================
with tab_rank:
    st.markdown('<div class="section-title">3) Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "FinalScore" in df.columns:
            df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

        cols_to_show = [
            "Ticker","Odds","Level","FinalScore",
            "PredVol_M","PredVol_CI68_L","PredVol_CI68_U"
        ]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level") else 0.0
        df = df[cols_to_show]

        st.dataframe(
            df, use_container_width=True, hide_index=True,
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

        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        col_del, col_dl = st.columns([1,1])
        with col_del:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []; st.session_state.last = {}; do_rerun()
        with col_dl:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "ranking.csv",
                "text/csv",
                use_container_width=True
            )

        st.markdown('<div class="section-title">📋 Ranking (Markdown view)</div>', unsafe_allow_html=True)
        st.code(df_to_markdown_table(df, cols_to_show), language="markdown")
    else:
        st.info("No rows yet. Upload your DB and click **Learn rank-curves**, then add a stock in the **Add Stock** tab.")

# =================== Variable Sweet Spots (data table) ===================
with tab_sweet:
    st.markdown('<div class="section-title">4) Variable Sweet Spots (from rank-curves)</div>', unsafe_allow_html=True)
    if not uploaded or "curves" not in (st.session_state.CURVES or {}):
        st.info("Upload your DB and learn curves first.")
    else:
        try:
            # Re-read merged for mapping to value space
            xls = pd.ExcelFile(uploaded)
            raw = pd.read_excel(xls, merged_sheet)
            built = build_numeric_from_merged(raw)
            data = built["data"]; y = data["FT"].astype(float)

            def sweet_risk_table_better(df, lift_thr=0.05, min_span=0.03, max_gap=0.02, min_support=3):
                rows = []
                exhaustion = _EXHAUSTION_VARS
                for v in [c for c in ["gap_pct","atr_usd","rvol","pm_vol_m","pm_dol_m","float_m","mcap_m","si_pct","fr_x","pmmc_pct","catalyst"] if c in df.columns]:
                    res = _rank_smooth(df[v], df["FT"], min_rows=20)
                    if res is None: 
                        continue
                    grid = res["grid"]; p = res["p_grid"]; p0 = res["p0"]; sup = res["support"]
                    # tail penalty for exhaustion vars
                    if v in exhaustion:
                        tail = np.clip((grid - 0.90) / 0.08, 0, 1)
                        pen  = 1.0 - 0.7 * tail
                        p    = p0 + (p - p0) * pen
                    i_peak = int(np.nanargmax(p)); peak_rank=float(grid[i_peak]); peak_p=float(p[i_peak])
                    thr = max(p0 + lift_thr, peak_p - 0.08)
                    mask = (p >= thr) & (sup >= min_support)
                    sweet_rank = _intervals_with_gap(mask, grid, max_gap=max_gap, min_span=min_span)
                    risk_mask  = (p <= p0 - lift_thr) & (sup >= min_support)
                    risk_rank  = _intervals_with_gap(risk_mask, grid, max_gap=max_gap, min_span=min_span)
                    # map to values
                    xv = pd.to_numeric(df[v], errors="coerce").dropna().to_numpy(float)
                    def r2v(ivals):
                        return [(float(np.nanquantile(xv, lo)), float(np.nanquantile(xv, hi))) for lo,hi in ivals]
                    sweet_vals = r2v(sweet_rank); risk_vals = r2v(risk_rank)
                    peak_val = float(np.nanquantile(xv, peak_rank)) if xv.size else float("nan")
                    rows.append({
                        "Variable": v,
                        "Base p(FT)": round(p0,3),
                        "Sweet rank": "; ".join([f"[{lo:.2f},{hi:.2f}]" for lo,hi in sweet_rank]) or "",
                        "Sweet values": "; ".join([f"[{_fmt_value(lo)},{_fmt_value(hi)}]" for lo,hi in sweet_vals]) or "",
                        "Risk rank": "; ".join([f"[{lo:.2f},{hi:.2f}]" for lo,hi in risk_rank]) or "",
                        "Risk values": "; ".join([f"[{_fmt_value(lo)},{_fmt_value(hi)}]" for lo,hi in risk_vals]) or "",
                        "Peak p(FT)": round(peak_p,3),
                        "Peak rank": round(peak_rank,3),
                        "Peak value": _fmt_value(peak_val)
                    })
                return pd.DataFrame(rows)

            tbl = sweet_risk_table_better(data, lift_thr=0.05, min_span=0.03, max_gap=0.02, min_support=3)
            st.dataframe(
                tbl, use_container_width=True, hide_index=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable"),
                    "Base p(FT)": st.column_config.NumberColumn("Base p(FT)", format="%.3f"),
                    "Sweet rank": st.column_config.TextColumn("Sweet spot (rank)"),
                    "Sweet values": st.column_config.TextColumn("Sweet spot (original units)"),
                    "Risk rank": st.column_config.TextColumn("Risk zone (rank)"),
                    "Risk values": st.column_config.TextColumn("Risk zone (original units)"),
                    "Peak p(FT)": st.column_config.NumberColumn("Peak p(FT)", format="%.3f"),
                    "Peak rank": st.column_config.NumberColumn("Peak rank", format="%.3f"),
                    "Peak value": st.column_config.TextColumn("Peak value"),
                }
            )
            st.download_button(
                "Download sweet/risk table (CSV)",
                tbl.to_csv(index=False).encode("utf-8"),
                "sweet_risk_table.csv",
                "text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Sweet-spot analysis failed: {e}")
