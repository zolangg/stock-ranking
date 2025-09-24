# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

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

# ============================== Sidebar ==============================
st.sidebar.header("Curves")
BINS = st.sidebar.slider("Curve bins (histogram)", min_value=2, max_value=10, value=2, step=1)
show_baseline = st.sidebar.checkbox("Curves: show baseline", True)
plot_all_curves = st.sidebar.checkbox("Curves: plot ALL variables", False)
sel_curve_var = st.sidebar.selectbox(
    "Or plot one variable",
    ["gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m","fr_x",
     "pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"]
)

# Threshold tuning floor for the L1 model (you asked to keep 0.65)
PRECISION_FLOOR = 0.65

# ============================== Session State ==============================
if "MODELS" not in st.session_state: st.session_state.MODELS = {}   # var -> curve dict (for plots only)
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}

# L1 model artifacts
if "L1_FEATURES" not in st.session_state: st.session_state.L1_FEATURES = []
if "L1_COEF" not in st.session_state: st.session_state.L1_COEF = None
if "L1_INTERCEPT" not in st.session_state: st.session_state.L1_INTERCEPT = 0.0
if "L1_THRESHOLD" not in st.session_state: st.session_state.L1_THRESHOLD = 0.50
if "L1_MEDIANS" not in st.session_state: st.session_state.L1_MEDIANS = {}
if "L1_METRICS" not in st.session_state: st.session_state.L1_METRICS = {}

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

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return s.replace("$","").replace("%","").replace("’","").replace("'","")

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns); nm = {_norm(c): c for c in cols}
    for cand in candidates:
        n = _norm(cand)
        if n in nm: return nm[n]
    for cand in candidates:
        n = _norm(cand)
        for k,v in nm.items():
            if n in k: return v
    return None

# ================= Predicted Day Volume (for PM% fallback) =================
def predict_day_volume_m_premarket(mcap_m: float, gap_pct: float, atr_usd: float) -> float:
    """ ln(Y) = 3.1435 + 0.1608*ln(MCap_M) + 0.6704*ln(Gap_%/100) − 0.3878*ln(ATR_$) """
    e = 1e-6
    mc  = max(float(mcap_m or 0.0), 0.0)
    gp  = max(float(gap_pct or 0.0), 0.0) / 100.0
    atr = max(float(atr_usd or 0.0), 0.0)
    ln_y = 3.1435 + 0.1608*math.log(mc + e) + 0.6704*math.log(gp + e) - 0.3878*math.log(atr + e)
    return math.exp(ln_y)

# ================= Curves (for plots only) =================
STRETCH_EPS = 0.10
def moving_average(y: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1: return y
    pad = w//2
    ypad = np.pad(y, (pad,pad), mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode='valid')

def stretch_curve_to_unit(p: np.ndarray, base_p: float) -> np.ndarray:
    eps = STRETCH_EPS
    p = np.asarray(p, dtype=float)
    pmin, pmax = float(np.nanmin(p)), float(np.nanmax(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return np.full_like(p, base_p)
    scale = max(1e-9, (pmax - pmin))
    p_stretched = eps + (1.0 - 2.0*eps) * (p - pmin) / scale
    return np.clip(p_stretched, 1e-6, 1.0 - 1e-6)

def _smooth_local_baseline(centers: np.ndarray, p_curve: np.ndarray, support: np.ndarray, bandwidth: float = 0.22) -> np.ndarray:
    c = centers.astype(float); p = p_curve.astype(float); n = (support.astype(float) + 1e-9)
    diffs = (c[:, None] - c[None, :]) / max(1e-6, bandwidth)
    w = np.exp(-0.5 * diffs**2) * n[None, :]
    num = (w * p[None, :]).sum(axis=1)
    den = w.sum(axis=1)
    pb = np.where(den > 0, num / den, np.nan)
    pb = pd.Series(pb).interpolate(limit_direction="both").fillna(np.nanmean(pb)).to_numpy()
    return pb

def rank_hist_model(x: pd.Series, y: pd.Series, bins: int) -> Optional[Dict[str,Any]]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 40 or y.nunique() != 2:
        return None

    ranks = x.rank(pct=True)
    B = int(bins)
    edges = np.linspace(0, 1, B + 1)
    idx = np.clip(np.searchsorted(edges, ranks, side="right")-1, 0, B-1)

    total = np.bincount(idx, minlength=B)
    ft    = np.bincount(idx[y==1], minlength=B)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_bin = np.where(total>0, ft/total, np.nan)

    if B > 2:
        p_series = pd.Series(p_bin).interpolate(limit_direction="both")
        p_fill   = p_series.fillna(p_series.mean()).to_numpy()
        p_smooth = moving_average(p_fill, w=3)
    else:
        p_smooth = p_bin

    centers = (edges[:-1] + edges[1:]) / 2.0
    p0_global = float(y.mean())
    p_base_var = float(np.average(p_smooth, weights=(total + 1e-9)))

    if B == 2:
        p_low, p_high = float(p_smooth[0]), float(p_smooth[1])
        p_line = p_base_var + (p_high - p_low) * (centers - 0.5)
        p_line = np.clip(p_line, 0.05, 0.95)
        p_ready = p_line
    else:
        p_ready = p_smooth

    eps_use = 0.08 if B == 2 else STRETCH_EPS
    pmin, pmax = float(np.min(p_ready)), float(np.max(p_ready))
    if np.isfinite(pmin) and np.isfinite(pmax) and pmax > pmin:
        scale = (pmax - pmin)
        p_use = eps_use + (1.0 - 2.0*eps_use) * (p_ready - pmin) / scale
    else:
        p_use = np.full_like(p_ready, p_base_var)
    p_use = np.clip(p_use, 1e-6, 1 - 1e-6)

    pb_curve = _smooth_local_baseline(centers, p_use, total, bandwidth=0.30 if B==2 else 0.22)

    pr = np.linspace(0,1,41)
    vals = np.quantile(x, pr)

    return {
        "edges": edges,
        "centers": centers,
        "support": total,
        "p_raw": p_use,
        "p0_global": p0_global,
        "p_base_var": p_base_var,
        "pb_curve": pb_curve,
        "quantiles": {"pr": pr, "vals": vals},
        "n": int(len(x))
    }

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn models", use_container_width=True)

# Utility to median-impute safely
def _median_impute(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    X = df_feat.replace([np.inf, -np.inf], np.nan).copy()
    med = {}
    for c in X.columns:
        if X[c].isna().all():
            med[c] = 0.0
            X[c] = 0.0
        else:
            m = float(X[c].median())
            if not np.isfinite(m): m = 0.0
            med[c] = m
            X[c] = X[c].fillna(m)
    return X, med

def _prob_to_grade_simple(p: float) -> str:
    # Simple ABCD mapping you asked for (in practice you can tweak)
    if p >= 0.75: return "A"
    if p >= 0.60: return "B"
    if p >= 0.45: return "C"
    return "D"

def _prob_to_odds_simple(p: float) -> str:
    if p >= 0.85: return "Very High Odds"
    if p >= 0.70: return "High Odds"
    if p >= 0.55: return "Moderate Odds"
    if p >= 0.40: return "Low Odds"
    return "Very Low Odds"

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

                # --- Column mapping ---
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

                # --- Build df with features + target ---
                df = pd.DataFrame()
                df["FT"] = pd.to_numeric(raw[col_ft], errors="coerce")

                if col_gap:   df["gap_pct"]  = pd.to_numeric(raw[col_gap],   errors="coerce")
                if col_atr:   df["atr_usd"]  = pd.to_numeric(raw[col_atr],   errors="coerce")
                if col_rvol:  df["rvol"]     = pd.to_numeric(raw[col_rvol],  errors="coerce")
                if col_pmvol: df["pm_vol_m"] = pd.to_numeric(raw[col_pmvol], errors="coerce")
                if col_pmdol: df["pm_dol_m"] = pd.to_numeric(raw[col_pmdol], errors="coerce")
                if col_float: df["float_m"]  = pd.to_numeric(raw[col_float],  errors="coerce")
                if col_mcap:  df["mcap_m"]   = pd.to_numeric(raw[col_mcap],  errors="coerce")
                if col_si:    df["si_pct"]   = pd.to_numeric(raw[col_si],    errors="coerce")
                if col_cat:   df["catalyst"] = pd.to_numeric(raw[col_cat],   errors="coerce").clip(0,1)
                if col_daily: df["daily_vol_m"] = pd.to_numeric(raw[col_daily], errors="coerce")

                # Derived (for features & for plots)
                if {"pm_vol_m","float_m"}.issubset(df.columns):
                    df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                    df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
                if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
                    df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"]
                # Predicted PM% fallback (if needed later)
                if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                    def _pred_row(r):
                        try:
                            return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                        except Exception:
                            return np.nan
                    pred = df.apply(_pred_row, axis=1)
                    df["pm_pct_pred"] = 100.0 * df.get("pm_vol_m", np.nan) / pred

                # --------- Fit "curves" models for plots (optional) ---------
                y_plot = df["FT"].astype(float)
                candidates = [
                    "gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m",
                    "fr_x","pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"
                ]
                models = {}
                for v in candidates:
                    if v in df.columns:
                        m = rank_hist_model(df[v], y_plot, bins=BINS)
                        if m is not None:
                            centers = m["centers"]
                            p_use = stretch_curve_to_unit(m["p_raw"], base_p=m["p_base_var"])
                            pb_curve = _smooth_local_baseline(centers, p_use, m["support"], bandwidth=0.22)
                            m["p"] = p_use
                            m["pb_curve"] = pb_curve
                            models[v] = m

                # --------- L1 Logistic Regression for SCORING ---------
                # Features: raw numeric
                feat = ["gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m",
                        "fr_x","pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"]
                # Prefer pm_pct_daily; if missing values, pm_pct_pred acts as spare later
                if "pm_pct_daily" not in df.columns and "pm_pct_pred" in df.columns:
                    # fine, still keep pm_pct_pred
                    pass

                df_train = df[df["FT"].notna()].copy()
                y = df_train["FT"].astype(int)
                X = df_train.reindex(columns=feat)  # may include columns that don't exist -> NaN

                # Impute medians robustly
                X, medians = _median_impute(X)

                # Sequential 70/30 split (first 70% train, last 30% test)
                n = len(X)
                cut = int(0.7 * n)
                X_train, X_test = X.iloc[:cut], X.iloc[cut:]
                y_train, y_test = y.iloc[:cut], y.iloc[cut:]

                # Fit L1 model (raw features, liblinear)
                clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=42, max_iter=500)
                clf.fit(X_train, y_train)

                # Tune threshold for highest recall with precision ≥ floor
                prob_test = clf.predict_proba(X_test)[:,1]
                prec, rec, thr = precision_recall_curve(y_test, prob_test)
                best_idx, best_rec = None, -1.0
                for i,(pval,rval) in enumerate(zip(prec, rec)):
                    if pval >= PRECISION_FLOOR and rval > best_rec:
                        best_idx, best_rec = i, rval
                if best_idx is not None:
                    tuned_thr = thr[min(best_idx, len(thr)-1)]
                else:
                    # Fallback: pick threshold that maximizes F1
                    f1s = 2*prec*rec/(prec+rec+1e-9)
                    ix = int(np.nanargmax(f1s))
                    tuned_thr = thr[min(ix, len(thr)-1)]

                # Report metrics at tuned threshold
                y_hat = (prob_test >= tuned_thr).astype(int)
                p_out = float(precision_score(y_test, y_hat))
                r_out = float(recall_score(y_test, y_hat))
                f_out = float(f1_score(y_test, y_hat))

                st.session_state.MODELS = models
                st.session_state.L1_FEATURES = list(X.columns)
                st.session_state.L1_COEF = clf.coef_[0].astype(float)
                st.session_state.L1_INTERCEPT = float(clf.intercept_[0])
                st.session_state.L1_THRESHOLD = float(tuned_thr)
                st.session_state.L1_MEDIANS = medians
                st.session_state.L1_METRICS = {"Precision": p_out, "Recall": r_out, "F1": f_out}

                st.success(
                    f"Learned L1 logistic (70/30). Tuned threshold={tuned_thr:.2f} (floor {PRECISION_FLOOR:.2f}). "
                    f"Val Precision={p_out:.2f}, Recall={r_out:.2f}, F1={f_out:.2f}."
                )

        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank, tab_curves = st.tabs(["➕ Add Stock", "📊 Ranking", "📈 Curves"])

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
            rvol     = input_float("RVOL", 0.0, min_value=0.0, decimals=2)
            pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
            pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
            dilution_flag = st.slider("Dilution present?", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                                      help="0 = none/negligible, 1 = strong overhang (penalizes log-odds)")
        with c3:
            catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"])

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        # Derived live
        fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
        pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

        # Predicted daily (fallback PM%)
        pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
        pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")
        pm_pct_daily = float("nan")  # generally unknown in premarket

        # Build feature vector in training order, with medians
        feat_order = st.session_state.get("L1_FEATURES", [])
        med = st.session_state.get("L1_MEDIANS", {})
        if not feat_order:
            st.error("Train the L1 model first (Upload & Learn).")
        else:
            live_row = {
                "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
                "float_m": float_m, "mcap_m": mc_m, "fr_x": fr_x, "pmmc_pct": pmmc_pct,
                "pm_pct_daily": pm_pct_daily, "pm_pct_pred": pm_pct_pred, "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
            }
            x_vals = []
            for k in feat_order:
                v = live_row.get(k, np.nan)
                if not np.isfinite(v):
                    v = float(med.get(k, 0.0))
                x_vals.append(float(v))
            x = np.array(x_vals, dtype=float)

            # L1 model prediction (logit), apply dilution penalty in logit space
            coef = st.session_state.get("L1_COEF", None)
            intercept = st.session_state.get("L1_INTERCEPT", 0.0)
            if coef is None:
                st.error("Model coefficients missing. Train the model again.")
            else:
                z = float(intercept + np.dot(coef, x))
                z += -0.90 * float(dilution_flag)  # dilution penalty
                z = float(np.clip(z, -12, 12))
                prob = float(1.0 / (1.0 + math.exp(-z)))

                tuned_thr = st.session_state.get("L1_THRESHOLD", 0.50)
                is_hit = (prob >= tuned_thr)

                final_score = float(np.clip(prob*100.0, 0.0, 100.0))
                odds_name = _prob_to_odds_simple(prob)
                level = _prob_to_grade_simple(prob)
                verdict_pill = (
                    '<span class="pill pill-good">Strong Setup</span>' if level=="A" else
                    '<span class="pill pill-warn">Constructive</span>' if level in ("B","C") else
                    '<span class="pill pill-bad">Weak / Avoid</span>'
                )

                # Simple checklist based on sign of contribution vs median
                name_map = {
                    "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
                    "float_m":"Float (M)","mcap_m":"MarketCap (M)","fr_x":"PM Float Rotation ×",
                    "pmmc_pct":"PM $Vol / MC %","pm_pct_daily":"PM Vol % of Daily",
                    "pm_pct_pred":"PM Vol % of Pred","catalyst":"Catalyst"
                }
                good, warn, risk = [], [], []
                eps_rel = 0.05  # ±5% band around median treated as neutral/caution
                for k, c in zip(feat_order, coef):
                    nm = name_map.get(k, k)
                    val = live_row.get(k, np.nan)
                    medk = float(med.get(k, 0.0))
                    if not np.isfinite(val):
                        val = medk
                    # Direction & distance from median
                    hi = medk * (1 + eps_rel)
                    lo = medk * (1 - eps_rel)
                    if c > 0:
                        if val > hi:   good.append(f"{nm}: {_fmt_value(val)}")
                        elif val < lo: risk.append(f"{nm}: {_fmt_value(val)}")
                        else:          warn.append(f"{nm}: {_fmt_value(val)}")
                    elif c < 0:
                        if val < lo:   good.append(f"{nm}: {_fmt_value(val)}")
                        elif val > hi: risk.append(f"{nm}: {_fmt_value(val)}")
                        else:          warn.append(f"{nm}: {_fmt_value(val)}")
                    else:
                        warn.append(f"{nm}: {_fmt_value(val)}")

                # Save row
                row = {
                    "Ticker": ticker,
                    "Odds": odds_name,
                    "Level": level,
                    "FinalScore": round(final_score, 2),
                    "PredVol_M": round(pred_vol_m, 2),
                    "VerdictPill": verdict_pill,
                    "GoodList": good, "WarnList": warn, "RiskList": risk,
                    "Hit@Thr": "Yes" if is_hit else "No",
                }
                st.session_state.rows.append(row)
                st.session_state.last = row
                st.success(
                    f"Saved {ticker} — Odds {row['Odds']} (Score {row['FinalScore']}) "
                    f"| Class@{tuned_thr:.2f}: {row['Hit@Thr']} "
                )

    # preview card
    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c,d,e = st.columns(5)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Final Score", f"{l.get('FinalScore',0):.2f}")
        c.metric("Grade", l.get('Level','—'))
        d.metric("Odds", l.get('Odds','—'))
        e.metric("PredVol (M)", f"{l.get('PredVol_M',0):.2f}")

        with st.expander("Premarket Checklist", expanded=True):
            st.markdown(f"**Verdict:** {l.get('VerdictPill','—')}", unsafe_allow_html=True)
            g,w,r = st.columns(3)
            def ul(items): return "<ul>"+"".join([f"<li>{x}</li>" for x in items])+"</ul>" if items else "<ul><li>—</li></ul>"
            with g: st.markdown("**Good**");    st.markdown(ul(l.get("GoodList",[])), unsafe_allow_html=True)
            with w: st.markdown("**Caution**"); st.markdown(ul(l.get("WarnList",[])), unsafe_allow_html=True)
            with r: st.markdown("**Risk**");    st.markdown(ul(l.get("RiskList",[])), unsafe_allow_html=True)

        # Show validation metrics
        m = st.session_state.get("L1_METRICS", {})
        if m:
            st.caption(f"Validation @ tuned threshold: Precision {m.get('Precision',0):.2f} • Recall {m.get('Recall',0):.2f} • F1 {m.get('F1',0):.2f}")

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)
    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows).sort_values("FinalScore", ascending=False).reset_index(drop=True)
        cols_to_show = ["Ticker","Odds","Level","FinalScore","PredVol_M","Hit@Thr"]
        for c in cols_to_show:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Odds","Level","Hit@Thr") else 0.0
        st.dataframe(
            df[cols_to_show], use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Odds": st.column_config.TextColumn("Odds"),
                "Level": st.column_config.TextColumn("Grade"),
                "FinalScore": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "PredVol_M": st.column_config.NumberColumn("Predicted Day Vol (M)", format="%.2f"),
                "Hit@Thr": st.column_config.TextColumn(f"≥ Thr ({st.session_state.get('L1_THRESHOLD',0.5):.2f})"),
            }
        )
        st.download_button(
            "Download CSV",
            df[cols_to_show].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )
    else:
        st.info("Add at least one stock.")

# ============================== Curves (for plots) ==============================
with tab_curves:
    st.markdown('<div class="section-title">Learned Curves (rank-space FT rate, local baselines dashed)</div>', unsafe_allow_html=True)
    models = st.session_state.MODELS or {}
    if not models:
        st.info("Upload + Learn first.")
    else:
        if plot_all_curves:
            learned_vars = list(models.keys())
            n = len(learned_vars)
            ncols = 3
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.6, nrows*3.2))
            axes = np.atleast_2d(axes)
            for i, var in enumerate(learned_vars):
                ax = axes[i//ncols, i % ncols]
                m = models[var]
                centers = m["centers"]; p = m["p"]
                ax.plot(centers, p, lw=2)
                pb_curve = m.get("pb_curve")
                if show_baseline:
                    if pb_curve is not None:
                        ax.plot(centers, pb_curve, ls="--", lw=1)
                    else:
                        ax.axhline(m.get("p_base_var", 0.5), ls="--", lw=1)
                ax.set_title(var, fontsize=11)
                ax.set_xlabel("Rank (percentile)", fontsize=10)
                ax.set_ylabel("P(FT)", fontsize=10)
                ax.tick_params(labelsize=9)
            total_axes = nrows * ncols
            for j in range(n, total_axes):
                fig.delaxes(axes[j//ncols, j % ncols])
            st.pyplot(fig, clear_figure=True)
        else:
            var = sel_curve_var
            m = models.get(var)
            if m is None:
                st.warning(f"No curve learned for '{var}'.")
            else:
                centers = m["centers"]; p = m["p"]
                fig, ax = plt.subplots(figsize=(6.4, 3.4))
                ax.plot(centers, p, lw=2)
                pb_curve = m.get("pb_curve")
                if show_baseline:
                    if pb_curve is not None:
                        ax.plot(centers, pb_curve, ls="--", lw=1)
                    else:
                        ax.axhline(m.get("p_base_var", 0.5), ls="--", lw=1)
                ax.set_xlabel("Rank (percentile of variable)", fontsize=10)
                ax.set_ylabel("P(FT | rank)", fontsize=10)
                ax.set_title(f"{var} — FT curve (baseline dashed)", fontsize=11)
                ax.tick_params(labelsize=9)
                st.pyplot(fig, clear_figure=True)
