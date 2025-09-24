# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from typing import Optional, Dict, Any, List, Tuple
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
st.sidebar.header("Settings")
PRECISION_FLOOR = st.sidebar.slider("Precision floor (tuning)", 0.00, 0.95, 0.65, 0.01)

# ============================== Session State ==============================
if "rows"   not in st.session_state: st.session_state.rows = []
if "last"   not in st.session_state: st.session_state.last = {}

# Logistic model artifacts
if "L1_FEATURES"  not in st.session_state: st.session_state.L1_FEATURES  = []
if "L1_COEF"      not in st.session_state: st.session_state.L1_COEF      = None
if "L1_INTERCEPT" not in st.session_state: st.session_state.L1_INTERCEPT = 0.0
if "L1_THRESHOLD" not in st.session_state: st.session_state.L1_THRESHOLD = 0.50
if "L1_MEDIANS"   not in st.session_state: st.session_state.L1_MEDIANS   = {}
if "L1_SCALE"     not in st.session_state: st.session_state.L1_SCALE     = {}
if "L1_METRICS"   not in st.session_state: st.session_state.L1_METRICS   = {}

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

# ================= Logistic regression (no sklearn model) =================
def _fit_logistic_ridge(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 400, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    IRLS/Newton for logistic regression with L2 penalty on weights (not intercept).
    Returns (coef_[k], intercept_)
    """
    n, k = X.shape
    Xb = np.concatenate([np.ones((n,1)), X], axis=1)
    w = np.zeros(k+1)

    R = np.eye(k+1) * l2
    R[0,0] = 0.0  # no penalty on intercept

    for _ in range(max_iter):
        z = Xb @ w
        z = np.clip(z, -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        if np.all(W < 1e-9):
            break
        WX = Xb * W[:, None]
        H  = Xb.T @ WX + R
        g  = Xb.T @ (y - p)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w + delta
        if np.linalg.norm(delta) < tol:
            w = w_new
            break
        w = w_new

    intercept_ = float(w[0])
    coef_ = w[1:].astype(float)
    return coef_, intercept_

def _predict_logistic(X: np.ndarray, coef_: np.ndarray, intercept_: float) -> np.ndarray:
    z = intercept_ + X @ coef_
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

# ================= Data utils =================
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

def _robust_scale(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float,float]]]:
    stats = {}
    Xs = X.copy()
    for c in Xs.columns:
        m = float(Xs[c].median())
        q1, q3 = float(Xs[c].quantile(0.25)), float(Xs[c].quantile(0.75))
        iqr = q3 - q1
        s = iqr if iqr > 1e-9 else (Xs[c].std() if np.isfinite(Xs[c].std()) and Xs[c].std()>1e-9 else 1.0)
        Xs[c] = (Xs[c] - m) / s
        stats[c] = (m, s)
    return Xs, stats

def tune_threshold_with_floor(y_true: np.ndarray, y_prob: np.ndarray, floor: float = 0.65) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Align sizes (thr len = len(prec)-1)
    thr_full  = np.concatenate(([0.0], thr, [1.0]))
    prec_full = np.concatenate(([prec[0]], prec, [prec[-1]]))
    rec_full  = np.concatenate(([rec[0]],  rec,  [rec[-1]]))

    best_thr, best_rec = None, -1.0
    for t, p, r in zip(thr_full, prec_full, rec_full):
        y_hat = (y_prob >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        if p >= floor and r > best_rec:
            best_rec, best_thr = r, t

    if best_thr is not None:
        return float(best_thr)

    # Fallback: best F1 with at least one positive
    best_f1, best_t = -1.0, None
    for t, p, r in zip(thr_full, prec_full, rec_full):
        y_hat = (y_prob >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        f1 = 0.0 if (p+r)==0 else 2*p*r/(p+r)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    if best_t is not None:
        return float(best_t)

    # Last resort: pick very high quantile to ensure ≥1 positive
    if y_prob.size:
        return float(np.quantile(y_prob, 1.0 - 1.0/max(1, y_prob.size)))

    return 0.5

def _prob_to_grade_simple(p: float) -> str:
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

# ============================== Upload & Learn ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
merged_sheet = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn model", use_container_width=True)

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

                if col_ft is None:
                    st.error("No 'FT' column found in sheet.")
                else:
                    # --- Build df with features + target ---
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

                    # Derived features
                    if {"pm_vol_m","float_m"}.issubset(df.columns):
                        df["fr_x"] = df["pm_vol_m"] / df["float_m"]
                    if {"pm_dol_m","mcap_m"}.issubset(df.columns):
                        df["pmmc_pct"] = 100.0 * df["pm_dol_m"] / df["mcap_m"]
                    if {"pm_vol_m","daily_vol_m"}.issubset(df.columns):
                        df["pm_pct_daily"] = 100.0 * df["pm_vol_m"] / df["daily_vol_m"]
                    if {"mcap_m","gap_pct","atr_usd","pm_vol_m"}.issubset(df.columns):
                        def _pred_row(r):
                            try:
                                return predict_day_volume_m_premarket(r["mcap_m"], r["gap_pct"], r["atr_usd"])
                            except Exception:
                                return np.nan
                        pred = df.apply(_pred_row, axis=1)
                        df["pm_pct_pred"] = 100.0 * df.get("pm_vol_m", np.nan) / pred

                    df_train = df[df["FT"].notna()].copy()
                    y = df_train["FT"].astype(int).values

                    feat = ["gap_pct","atr_usd","rvol","si_pct","float_m","mcap_m",
                            "fr_x","pmmc_pct","pm_pct_daily","pm_pct_pred","catalyst"]
                    X = df_train.reindex(columns=feat)

                    # Impute + robust scale
                    X_imp, medians = _median_impute(X)
                    X_scaled, scale_stats = _robust_scale(X_imp)
                    X_np = X_scaled.values

                    # Sequential 70/30 split (first 70% train, last 30% validation)
                    n = len(X_np)
                    if n < 10:
                        st.warning("Very few rows — model fit may be unstable.")
                    cut = int(0.7 * n)
                    X_tr, X_te = X_np[:cut], X_np[cut:]
                    y_tr, y_te = y[:cut], y[cut:]

                    # Fit ridge logistic
                    coef_, intercept_ = _fit_logistic_ridge(X_tr, y_tr, l2=1.0, max_iter=400, tol=1e-6)

                    # Tune threshold for highest recall subject to precision floor
                    prob_te = _predict_logistic(X_te, coef_, intercept_)
                    tuned_thr = tune_threshold_with_floor(y_te, prob_te, floor=PRECISION_FLOOR)

                    y_hat = (prob_te >= tuned_thr).astype(int)
                    p_out = float(precision_score(y_te, y_hat, zero_division=0))
                    r_out = float(recall_score(y_te, y_hat, zero_division=0))
                    f_out = float(f1_score(y_te, y_hat, zero_division=0))

                    # Save artifacts
                    st.session_state.L1_FEATURES  = list(X_imp.columns)
                    st.session_state.L1_COEF      = coef_
                    st.session_state.L1_INTERCEPT = intercept_
                    st.session_state.L1_THRESHOLD = tuned_thr
                    st.session_state.L1_MEDIANS   = medians
                    st.session_state.L1_SCALE     = scale_stats
                    st.session_state.L1_METRICS   = {"Precision": p_out, "Recall": r_out, "F1": f_out}

                    st.success(
                        f"Learned ridge logistic (70/30). Tuned threshold={tuned_thr:.2f} "
                        f"(floor {PRECISION_FLOOR:.2f}). Val Precision={p_out:.2f}, Recall={r_out:.2f}, F1={f_out:.2f}."
                    )
                    if len(y_te):
                        st.caption(
                            f"Train pos rate={y_tr.mean():.2f}, Val pos rate={y_te.mean():.2f} | "
                            f"Val prob range=[{prob_te.min():.3f}, {prob_te.max():.3f}]"
                        )
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add Stock", "📊 Ranking"])

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
        feat_order = st.session_state.get("L1_FEATURES", [])
        med        = st.session_state.get("L1_MEDIANS", {})
        scale_stats= st.session_state.get("L1_SCALE", {})
        coef       = st.session_state.get("L1_COEF", None)
        intercept  = st.session_state.get("L1_INTERCEPT", 0.0)
        tuned_thr  = st.session_state.get("L1_THRESHOLD", 0.50)

        if not feat_order or coef is None:
            st.error("Train the model first (Upload & Learn).")
        else:
            # Derived live
            fr_x = (pm_vol_m / float_m) if float_m > 0 else float("nan")
            pmmc_pct = (100.0 * pm_dol_m / mc_m) if mc_m > 0 else float("nan")

            pred_vol_m = predict_day_volume_m_premarket(mc_m, gap_pct, atr_usd)
            pm_pct_pred = (100.0 * pm_vol_m / pred_vol_m) if pred_vol_m > 0 else float("nan")

            live_row = {
                "gap_pct": gap_pct, "atr_usd": atr_usd, "rvol": rvol, "si_pct": si_pct,
                "float_m": float_m, "mcap_m": mc_m, "fr_x": fr_x, "pmmc_pct": pmmc_pct,
                "pm_pct_daily": float("nan"), "pm_pct_pred": pm_pct_pred,
                "catalyst": 1.0 if catalyst_flag=="Yes" else 0.0
            }

            # Median-impute + robust-scale with training stats
            x_vals = []
            for k in feat_order:
                v = live_row.get(k, np.nan)
                if not np.isfinite(v):
                    v = float(med.get(k, 0.0))
                m, s = scale_stats.get(k, (0.0, 1.0))
                v_scaled = (v - m) / (s if abs(s) > 1e-9 else 1.0)
                x_vals.append(float(v_scaled))
            x = np.array(x_vals, dtype=float)

            # Score
            z = float(intercept + np.dot(coef, x))
            z += -0.90 * float(dilution_flag)  # dilution penalty in logit space
            z = float(np.clip(z, -12, 12))
            prob = float(1.0 / (1.0 + math.exp(-z)))
            is_hit = (prob >= tuned_thr)

            final_score = float(np.clip(prob*100.0, 0.0, 100.0))
            odds_name = _prob_to_odds_simple(prob)
            level = _prob_to_grade_simple(prob)
            verdict_pill = (
                '<span class="pill pill-good">Strong Setup</span>' if level=="A" else
                '<span class="pill pill-warn">Constructive</span>' if level in ("B","C") else
                '<span class="pill pill-bad">Weak / Avoid</span>'
            )

            # Simple checklist using coefficient direction vs median ±5%
            name_map = {
                "gap_pct":"Gap %","atr_usd":"ATR $","rvol":"RVOL","si_pct":"Short Interest %",
                "float_m":"Float (M)","mcap_m":"MarketCap (M)","fr_x":"PM Float Rotation ×",
                "pmmc_pct":"PM $Vol / MC %","pm_pct_daily":"PM Vol % of Daily",
                "pm_pct_pred":"PM Vol % of Pred","catalyst":"Catalyst"
            }
            good, warn, risk = [], [], []
            eps_rel = 0.05
            for k, c in zip(feat_order, coef):
                nm = name_map.get(k, k)
                val = live_row.get(k, np.nan)
                medk = float(med.get(k, 0.0))
                if not np.isfinite(val): val = medk
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
