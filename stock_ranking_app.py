import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# ============================== Page & Styles ==============================
st.set_page_config(page_title="Premarket Stock Ranking — Derived from PMH BO Merged", layout="wide")
st.title("Premarket Stock Ranking — Data-derived rules from PMH BO Merged")

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

# ============================== Sidebar (settings) ==============================
st.sidebar.header("Learning / Scoring")

sb_bins     = st.sidebar.slider("Histogram bins (rank space)", 20, 120, 60, 5)
sb_lift     = st.sidebar.slider("Lift threshold over baseline for sweet/risk", 0.02, 0.25, 0.08, 0.01)
sb_support  = st.sidebar.slider("Min samples per rank-bin", 2, 50, 6, 1)
sb_gapmerge = st.sidebar.slider("Merge close intervals ≤ (rank width)", 0.00, 0.10, 0.02, 0.005)
sb_minspan  = st.sidebar.slider("Min interval span (rank)", 0.005, 0.10, 0.03, 0.005)

st.sidebar.markdown("---")
tail_strength = st.sidebar.slider("High-tail penalty (exhaustion vars)", 0.0, 1.0, 0.55, 0.05)
weight_dampen = st.sidebar.slider("Weight dampening (prevents domination)", 0.0, 1.0, 0.20, 0.05)

st.sidebar.markdown("---")
show_baseline = st.sidebar.checkbox("Curves: show baseline", True)
plot_all_curves = st.sidebar.checkbox("Curves: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or plot one variable",
    ["gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m","float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","catalyst"]
)

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}   # var -> model dict
if "WEIGHTS" not in st.session_state: st.session_state.WEIGHTS = {} # var -> weight (0..1)
if "BASEP"  not in st.session_state: st.session_state.BASEP = 0.5
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}

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

# ================= Predicted Day Volume (for checklist only) =================
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

# ================= Rank-hist learning (DB-derived) =================
EXHAUSTION_VARS = {"gap_pct","rvol","pm_dol_m","pmmc_pct","fr_x"}

def rank_hist_model(x: pd.Series, y: pd.Series, bins: int) -> Optional[Dict[str,Any]]:
    """Learn FT-rate curve in rank space (robust)."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 40 or y.nunique() != 2: return None

    ranks = x.rank(pct=True)   # 0..1
    edges = np.linspace(0,1,bins+1)
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, bins-1)

    total = np.bincount(idx, minlength=bins)
    ft    = np.bincount(idx[y==1], minlength=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(total>0, ft/total, np.nan)

    # Fill gaps by linear interpolation → robust, simple
    p_series = pd.Series(p).interpolate(limit_direction="both")
    p = p_series.fillna(p_series.mean()).to_numpy()

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    return {
        "edges": edges,
        "centers": (edges[:-1]+edges[1:])/2,
        "p": p,                    # FT rate per rank-bin
        "support": total,          # sample count per bin
        "p0": float(y.mean()),     # baseline FT rate
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

def auc_weight(x: pd.Series, y: pd.Series) -> float:
    """AUC-like separation → 0..1. Then dampened later."""
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x,y = x[mask], y[mask]
    if len(x) < 40 or y.nunique()!=2: return 0.0
    r = x.rank(method="average")
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return 0.0
    s1 = r[y==1].sum()
    auc = (s1 - n1*(n1+1)/2) / (n1*n0)
    return float(max(0.05, abs(auc-0.5)*2.0))  # 0..1 with floor

def tail_penalize(var_key: str, centers: np.ndarray, p: np.ndarray, p0: float, strength: float) -> np.ndarray:
    """Penalize extreme high ranks for exhaustion variables (ex: RVOL 7k+, Gap 300%+)."""
    if var_key not in EXHAUSTION_VARS or strength <= 0: return p
    # ramp penalty in top tail (rank > 0.9 → up to strength)
    tail = np.clip((centers - 0.90) / 0.08, 0, 1)  # 0→1 over 0.90..0.98
    penalty = 1.0 - strength * tail
    return p0 + (p - p0) * penalty

def find_intervals(var_key: str, model: Dict[str,Any], lift: float, min_support: int, gap_merge: float, min_span: float, strength: float):
    """Extract sweet/risk rank-intervals where FT rate is well above/below baseline."""
    centers = model["centers"]; p = model["p"].copy(); p0 = model["p0"]; sup = model["support"]
    p_tail = tail_penalize(var_key, centers, p, p0, strength)

    sweet_mask = (p_tail >= p0 + lift) & (sup >= min_support)
    risk_mask  = (p_tail <= p0 - lift) & (sup >= min_support)

    def merge(mask):
        idx = np.where(mask)[0]
        if idx.size == 0: return []
        intervals = []
        start = centers[idx[0]]; last = centers[idx[0]]
        for k in idx[1:]:
            if (centers[k] - last) <= gap_merge:
                last = centers[k]
            else:
                if (last - start) >= min_span: intervals.append((float(start), float(last)))
                start = last = centers[k]
        if (last - start) >= min_span: intervals.append((float(start), float(last)))
        return intervals

    sweet_r = merge(sweet_mask)
    risk_r  = merge(risk_mask)

    pr = model["quantiles"]["pr"]; vals = model["quantiles"]["vals"]
    def r2v(lo, hi):
        lo_v = float(np.interp(lo, pr, vals))
        hi_v = float(np.interp(hi, pr, vals))
        return lo_v, hi_v

    sweet_v = [r2v(a,b) for a,b in sweet_r]
    risk_v  = [r2v(a,b) for a,b in risk_r]
    return sweet_r, sweet_v, risk_r, risk_v

def value_to_prob(var_key: str, model: Dict[str,Any], x_val: float, tail_strength: float) -> float:
    """Map a raw value to the model's FT probability via rank lookup (+ tail adjustment)."""
    if model is None or not np.isfinite(x_val): return 0.5
    pr, vals = model["quantiles"]["pr"], model["quantiles"]["vals"]
    # value → rank (linear in quantile grid)
    if x_val <= vals.min(): r = 0.0
    elif x_val >= vals.max(): r = 1.0
    else:
        idx = np.searchsorted(vals, x_val)
        i0 = max(1, min(idx, len(vals)-1))
        x0, x1 = vals[i0-1], vals[i0]
        p0, p1 = pr[i0-1], pr[i0]
        t = (x_val - x0) / (x1 - x0) if x1 != x0 else 0.0
        r = float(p0 + t*(p1 - p0))

    centers = model["centers"]; p = model["p"]; base = model["p0"]
    j = int(np.clip(np.searchsorted(centers, r), 0, len(centers)-1))
    p_local = float(p[j])
    # tail adjust
    p_adj = tail_penalize(var_key, np.array([centers[j]]), np.array([p_local]), base, tail_strength)[0]
    return float(np.clip(p_adj, 1e-3, 1-1e-3))

# ============================== Upload & Learn (Main Pane) ==============================
st.markdown('<div class="section-title">Upload workbook (sheet: PMH BO Merged)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn rules from merged", use_container_width=True)

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

                # column mapping (robust-ish)
                col_ft    = _pick(raw, ["ft","FT"])
                col_gap   = _pick(raw, ["gap %","gap%","premarket gap","gap"])
                col_atr   = _pick(raw, ["atr","atr $","atr$","atr (usd)"])
                col_rvol  = _pick(raw, ["rvol @ bo","rvol","relative volume"])
                col_pmvol = _pick(raw, ["pm vol (m)","premarket vol (m)","pm volume (m)","pm shares (m)"])
                col_pmdol = _pick(raw, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
                col_float = _pick(raw, ["float m shares","public float (m)","float (m)","float"])
                col_mcap  = _pick(raw, ["marketcap m","market cap (m)","mcap m","mcap"])
                col_si    = _pick(raw, ["si","short interest %","short float %","short interest (float) %"])
                col_cat   = _pick(raw, ["catalyst","news","pr"])
                col_daily = _pick(raw, ["daily vol (m)","day volume (m)","volume (m)"])

                if col_ft is None:
                    st.error("No 'FT' column found in merged sheet.")
                else:
                    df = pd.DataFrame()
                    df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce")

                    if col_gap:   df["gap_pct"]  = pd.to_numeric(raw[col_gap],   errors="coerce")
                    if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                    if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                    if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                    if col_pmdol: df["pm_dol_m"] = pd.to_numeric(raw[col_pmdol], errors="coerce")
                    if col_float: df["float_m"]  = pd.to_numeric(raw[col_float], errors="coerce")
                    if col_mcap:  df["mcap_m"]   = pd.to_numeric(raw[col_mcap],  errors="coerce")
                    if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")
                    if col_cat:   df["catalyst"] = pd.to_numeric(raw[col_cat],   errors="coerce").clip(0,1)
                    if col_daily: df["daily_vol_m"] = pd.to_numeric(raw[col_daily], errors="coerce")

                    # derived — from DB
                    if {"pm_vol_m","float_m"}.issubset(df.columns):
                        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
                    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                        # PM Vol % of Predicted Day Volume
                        def _pred_row(r):
                            try:
                                return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                            except Exception:
                                return np.nan
                        pred = df.apply(_pred_row, axis=1)
                        df["pm_pct_pred"] = 100.0 * df["pm_vol_m"] / pred

                    df = df[df["FT"].notna()]
                    y = df["FT"].astype(float)
                    base_p = float(y.mean())

                    # learn models
                    candidates = [
                        "gap_pct","atr_usd","rvol","si_pct","pm_vol_m","pm_dol_m",
                        "float_m","mcap_m","fr_x","pmmc_pct","pm_pct_pred","catalyst"
                    ]
                    models, weights = {}, {}
                    for v in candidates:
                        if v in df.columns:
                            m = rank_hist_model(df[v], y, bins=sb_bins)
                            if m is not None:
                                # store curve + sweet/risk intervals (value space)
                                s_r, s_v, r_r, r_v = find_intervals(
                                    v, m, lift=sb_lift, min_support=sb_support,
                                    gap_merge=sb_gapmerge, min_span=sb_minspan, strength=tail_strength
                                )
                                m["sweet_rank"] = s_r
                                m["sweet_vals"] = s_v
                                m["risk_rank"]  = r_r
                                m["risk_vals"]  = r_v
                                models[v] = m
                                weights[v] = auc_weight(df[v], y)

                    # dampen & normalize weights (avoid domination)
                    if weights:
                        damp = {k: (1.0 - weight_dampen) * w for k, w in weights.items()}
                        s = sum(damp.values()) or 1.0
                        weights = {k: v/s for k, v in damp.items()}

                    st.session_state.MODELS  = models
                    st.session_state.WEIGHTS = weights
                    st.session_state.BASEP   = base_p
                    st.success(f"Learned {len(models)} variables · Baseline P(FT) ≈ {base_p:.2f}")
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank, tab_curves, tab_spots = st.tabs(["➕ Add Stock", "📊 Ranking", "📈 Curves", "🎯 Sweet Spots"])

# ============================== Add Stock ==============================
with tab_add:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
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
            # place the dilution slider right under PM $Vol (your preference):
            dilution_flag = st.select_slider("Dilution present?", options=[0,1], value=0,
                                             help="0 = none/negligible, 1 = present/overhang (penalizes log-odds)")

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived live
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted Day Vol & PM% of Predicted (for checklist)
        sigma_ln = 0.60
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        ci68_l, ci68_u = ci_from_logsigma(pred_vol_m, sigma_ln, 1.0)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

        models  = st.session_state.MODELS or {}
        weights = st.session_state.WEIGHTS or {}
        base_p  = float(st.session_state.BASEP or 0.5)

        var_vals = {
            "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
            "pm_vol_m": pm_vol_m, "pm_dol_m": pm_dol_m, "float_m": float_m, "mcap_m": mc_m,
            "fr_x": fr_x, "pmmc_pct": pmmc_pct, "pm_pct_pred": pm_pct_pred,
            "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
        }

        # per-variable probability from learned curves
        parts: Dict[str, Dict[str, float]] = {}
        z_sum = 0.0
        for k, x in var_vals.items():
            mdl = models.get(k)
            p = value_to_prob(k, mdl, x, tail_strength) if mdl else base_p
            w = float(weights.get(k, 0.0))
            parts[k] = {"x": x, "p": p, "w": w}
            if p <= 0 or p >= 1: p = float(np.clip(p, 1e-6, 1-1e-6))
            z_sum += w * math.log(p/(1-p))

        # dilution penalty (binary)
        z_sum += -0.90 * float(dilution_flag)  # you can mirror sidebar knob if you want
        z_sum = float(np.clip(z_sum, -12, 12))
        numeric_prob = 1.0 / (1.0 + math.exp(-z_sum))

        final_score = float(np.clip(numeric_prob*100.0, 0.0, 100.0))
        odds_name = odds_label(final_score)
        level = grade(final_score)
        verdict_pill = (
            '<span class="pill pill-good">Strong Setup</span>' if final_score >= 70 else
            '<span class="pill pill-warn">Constructive</span>' if final_score >= 55 else
            '<span class="pill pill-bad">Weak / Avoid</span>'
        )

        # checklist (Good/Caution/Risk) by comparing p to baseline +/- lift
        name_map = {
            "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL @ BO","si_pct":"Short Interest %",
            "pm_vol_m":"PM Volume (M)","pm_dol_m":"PM $Vol (M)","float_m":"Float (M)","mcap_m":"MarketCap (M)",
            "fr_x":"PM Float Rotation ×","pmmc_pct":"PM $Vol / MC %","pm_pct_pred":"PM Vol % of Pred",
            "catalyst":"Catalyst"
        }
        good, warn, risk = [], [], []
        for k, d in parts.items():
            nm = name_map.get(k, k)
            p = d["p"]; x = d["x"]
            p_pct = int(round(p*100))
            # label by data-derived lift
            if p >= base_p + sb_lift:
                good.append(f"{nm}: {_fmt_value(x)} — good (p≈{p_pct}%)")
            elif p <= base_p - sb_lift:
                risk.append(f"{nm}: {_fmt_value(x)} — risk (p≈{p_pct}%)")
            else:
                warn.append(f"{nm}: {_fmt_value(x)} — caution (p≈{p_pct}%)")

        # prepend predicted volume line
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

    # preview card
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

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
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
        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )
    else:
        st.info("Add at least one stock.")

# ============================== Curves ==============================
with tab_curves:
    st.markdown('<div class="section-title">Learned Curves (rank-space FT rate)</div>', unsafe_allow_html=True)
    models = st.session_state.MODELS or {}
    base_p = float(st.session_state.BASEP or 0.5)
    if not models:
        st.info("Upload + Learn first.")
    else:
        if plot_all_curves:
            learned_vars = list(models.keys())
            n = len(learned_vars)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.8, nrows*3.2))
            axes = np.atleast_2d(axes)
            for i, var in enumerate(learned_vars):
                ax = axes[i//ncols, i % ncols]
                m = models[var]
                centers = m["centers"]; p = m["p"]
                p = tail_penalize(var, centers, p, m["p0"], tail_strength)
                ax.plot(centers, p, lw=2)
                if show_baseline: ax.axhline(base_p, ls="--", lw=1, color="gray")
                ax.set_title(var)
                ax.set_xlabel("Rank (percentile of variable)")
                ax.set_ylabel("P(FT)")
            # hide unused axes
            for j in range(i+1, nrows*ncols):
                fig.delaxes(axes[j//ncols, j % ncols])
            st.pyplot(fig, clear_figure=True)
        else:
            var = sel_curve_var
            m = models.get(var)
            if m is None:
                st.warning(f"No curve learned for '{var}'.")
            else:
                centers = m["centers"]; p = m["p"]
                p = tail_penalize(var, centers, p, m["p0"], tail_strength)
                fig, ax = plt.subplots(figsize=(6.4, 3.2))
                ax.plot(centers, p, lw=2)
                if show_baseline: ax.axhline(base_p, ls="--", lw=1, color="gray")
                ax.set_xlabel("Rank (percentile of variable)")
                ax.set_ylabel("P(FT | rank)")
                ax.set_title(f"{var} — FT rate curve")
                st.pyplot(fig, clear_figure=True)

# ============================== Sweet Spots ==============================
with tab_spots:
    st.markdown('<div class="section-title">Sweet Spots & Danger Zones (value intervals)</div>', unsafe_allow_html=True)
    models = st.session_state.MODELS or {}
    if not models:
        st.info("Upload + Learn first.")
    else:
        rows = []
        for v, m in models.items():
            s_r, s_v, r_r, r_v = m.get("sweet_rank",[]), m.get("sweet_vals",[]), m.get("risk_rank",[]), m.get("risk_vals",[])
            rows.append({
                "Variable": v,
                "Baseline p(FT)": round(m["p0"],3),
                "Sweet (values)": "; ".join([f"[{_fmt_value(a)},{_fmt_value(b)}]" for a,b in s_v]) or "",
                "Risk (values)":  "; ".join([f"[{_fmt_value(a)},{_fmt_value(b)}]" for a,b in r_v]) or "",
                "N": m.get("n",0)
            })
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("Download sweet/risk intervals (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           "sweet_risk_intervals.csv", "text/csv", use_container_width=True)
