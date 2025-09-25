# app.py — Premarket FT Ranking (no DayVol, no Float)
# ---------------------------------------------------
# • Trains a single FT classifier (weighted L2 logistic) from your workbook.
# • Features used (if columns present): ln1p_pmvol, ln_gapf, catalyst,
#   ln_atr, ln_mcap, ln1p_rvol, ln1p_pmmc, ln1p_si
# • Grades only: A≥0.90, A+≥0.97, A++≥0.99. Below 0.90, B/C/D via warped quantiles.
# • UI: full-width inputs, catalyst directly under PM $Vol; trained / NOT trained badge.
# • Table: Ticker, Grade, FT Probability %. Delete top-12, CSV/Markdown download, Clear.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math, re
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ============================== Page & CSS ==============================
st.set_page_config(page_title="Premarket FT Ranking", layout="wide")
st.title("Premarket FT Ranking")

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Helvetica Neue", sans-serif; font-size:14.25px; }
  .section-title { font-weight: 700; font-size: 1.02rem; letter-spacing:.12px; margin: 4px 0 8px 0; }
  .block-divider { border-bottom: 1px solid #e5e7eb; margin: 12px 0 14px 0; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.12rem; }
  [data-testid="stMetric"] label { font-size: 0.82rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ============================== Session ==============================
if "ART" not in st.session_state: st.session_state.ART = {}   # FT artifacts
if "rows" not in st.session_state: st.session_state.rows = []
if "last" not in st.session_state: st.session_state.last = {}
if "flash" not in st.session_state: st.session_state.flash = None
if "TRAIN_PREDS" not in st.session_state: st.session_state.TRAIN_PREDS = np.array([])

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

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

def _nz(x, fallback=0.0):
    try:
        xx = float(x)
        return xx if np.isfinite(xx) else float(fallback)
    except: return float(fallback)

def _safe_log(x: float, eps: float = 1e-8) -> float:
    return math.log(max(_nz(x, 0.0), eps))

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

# ============================== Legend-driven mapping ==============================
_DEF = {
    "FT":     ["FT"],
    "GAP":    ["Gap %","Gap"],
    "ATR":    ["Daily ATR","ATR $","ATR","ATR (USD)","ATR$"],
    "RVOL":   ["RVOL @ BO","RVOL","Relative Volume"],
    "PMVOL":  ["PM Vol (M)","Premarket Vol (M)","PM Volume (M)"],
    "PM$":    ["PM $Vol (M)","PM Dollar Vol (M)","PM $ Volume (M)"],
    "MCAP":   ["MarketCap M","Market Cap (M)","MCap M"],
    "SI":     ["Short Interest %","Short Float %","Short Interest (Float) %"],
    "CAT":    ["Catalyst","News","PR"],
}
def _norm(s: str) -> str:
    s = re.sub(r"\s+"," ", str(s).strip().lower())
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
def _parse_catalyst_col(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    pos = {"1","true","yes","y","t"}
    neg = {"0","false","no","n","f",""}
    out = []
    for v in s.fillna(""):
        if v in pos: out.append(1.0)
        elif v in neg: out.append(0.0)
        elif "pr" in v or "news" in v or "catalyst" in v: out.append(1.0)
        else:
            try: out.append(float(v))
            except: out.append(0.0)
    return pd.Series(out, dtype=float).clip(0,1)

# ============================== Logistic utils ==============================
def logit_fit_weighted(Z: np.ndarray, y: np.ndarray, sample_w: np.ndarray,
                       l2: float = 1.0, max_iter: int = 160, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    n, k = Z.shape
    Xb = np.concatenate([np.ones((n,1)), Z], axis=1)
    w = np.zeros(k+1)
    R = np.eye(k+1); R[0,0] = 0.0; R *= l2
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0/(1.0 + np.exp(-np.clip(z, -35, 35)))
        W = p*(1-p)*sample_w
        if np.all(W < 1e-8): break
        WX = Xb * W[:, None]
        H = Xb.T @ WX + R
        g = Xb.T @ ((y - p) * sample_w)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, g, rcond=None)[0]
        w = w + delta
        if np.linalg.norm(delta) < tol: break
    return w[1:].astype(float), float(w[0])

def logit_inv(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0/(1.0 + np.exp(-z))

# ============================== Grade cuts (A/A+/A++) ==============================
def _inv_warp_p(sw: float, alpha: float = 2.6) -> float:
    sw = float(np.clip(sw, 0.0, 1.0))
    return float(1.0 - (1.0 - sw) ** (1.0 / max(1e-9, alpha)))

def make_grade_cuts() -> dict:
    A_cut, Ap_cut, App_cut = 0.90, 0.97, 0.99
    p_cal = st.session_state.get("TRAIN_PREDS", np.array([]))
    if p_cal.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.80, "C": 0.60}
    sub = np.asarray(p_cal[(p_cal < A_cut) & np.isfinite(p_cal)])
    if sub.size == 0:
        return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": 0.80, "C": 0.60}
    sub.sort()
    alpha = 2.6
    b_pos = _inv_warp_p(0.80, alpha)  # skew towards higher
    c_pos = _inv_warp_p(0.50, alpha)
    B_cut = float(np.quantile(sub, b_pos))
    C_cut = float(np.quantile(sub, c_pos))
    B_cut = min(B_cut, 0.89)
    if not np.isfinite(B_cut): B_cut = 0.80
    if not np.isfinite(C_cut): C_cut = 0.60
    if C_cut >= B_cut: C_cut = max(0.10, B_cut - 0.02)
    return {"App": App_cut, "Ap": Ap_cut, "A": A_cut, "B": B_cut, "C": C_cut}

def _prob_to_grade(p: float, cuts: Dict[str,float]) -> str:
    if p >= cuts.get("App",0.99): return "A++"
    if p >= cuts.get("Ap",0.97):  return "A+"
    if p >= cuts.get("A",0.90):   return "A"
    if p >= cuts.get("B",0.80):   return "B"
    if p >= cuts.get("C",0.60):   return "C"
    return "D"

# ============================== Upload & Learn (FT only) ==============================
st.markdown('<div class="section-title">Upload workbook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
sheet_name = st.text_input("Sheet name", "PMH BO Merged")
learn_btn = st.button("Learn FT model", use_container_width=True)

def _learn_ft(xls: pd.ExcelFile, sheet: str) -> None:
    raw = pd.read_excel(xls, sheet)

    # map columns
    col = {}
    for k, cands in _DEF.items():
        p = _pick(raw, cands)
        if p: col[k] = p
    if "FT" not in col:
        st.error("No 'FT' column found.")
        return

    df = pd.DataFrame()
    df["FT"] = (pd.to_numeric(raw[col["FT"]], errors="coerce").fillna(0.0) >= 0.5).astype(float)

    def _add(name, key):
        if key in col:
            df[name] = pd.to_numeric(raw[col[key]], errors="coerce")

    _add("gap_pct","GAP")
    _add("atr_usd","ATR")
    _add("rvol","RVOL")
    _add("pm_vol_m","PMVOL")
    _add("pm_dol_m","PM$")
    _add("mcap_m","MCAP")
    _add("si_pct","SI")
    if "CAT" in col: df["catalyst"] = _parse_catalyst_col(raw[col["CAT"]])
    else: df["catalyst"] = 0.0

    # Build FT features (no float, no dayvol, no pm-fraction)
    ft_defs: List[Tuple[str, callable]] = [
        ("ln1p_pmvol",  lambda r: _safe_log(_nz(r.get("pm_vol_m"),0.0) + 1.0)),
        ("ln_gapf",     lambda r: _safe_log(_nz(r.get("gap_pct"),0.0)/100.0)),
        ("catalyst",    lambda r: float(_nz(r.get("catalyst"),0.0))),
        ("ln_atr",      lambda r: _safe_log(r.get("atr_usd"))),
        ("ln_mcap",     lambda r: _safe_log(r.get("mcap_m"))),
        ("ln1p_rvol",   lambda r: _safe_log(_nz(r.get("rvol"),0.0) + 1.0)),
        ("ln1p_pmmc",   lambda r: _safe_log(_nz(r.get("pm_dol_m"),0.0)/max(1e-6,_nz(r.get("mcap_m"),0.0)) + 1.0)),
        ("ln1p_si",     lambda r: _safe_log(_nz(r.get("si_pct"),0.0) + 1.0)),
    ]
    # keep only those with available sources
    def _has_src(name: str) -> bool:
        src_map = {
            "ln1p_pmvol": "pm_vol_m", "ln_gapf": "gap_pct", "catalyst": "catalyst",
            "ln_atr": "atr_usd", "ln_mcap": "mcap_m", "ln1p_rvol": "rvol",
            "ln1p_pmmc": ("pm_dol_m","mcap_m"), "ln1p_si": "si_pct"
        }
        src = src_map[name]
        if isinstance(src, tuple): return all(s in df.columns for s in src)
        return src in df.columns
    ft_defs = [(n,f) for (n,f) in ft_defs if _has_src(n)]
    if not ft_defs:
        st.error("No usable FT features found.")
        return

    X = np.vstack([[f(df.iloc[i].to_dict()) for (n,f) in ft_defs] for i in range(len(df))]).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["FT"].to_numpy(dtype=float)

    # Standardize
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    Z = np.clip(Z, -3.0, 3.0)

    # Class weights
    p1 = float(np.mean(y)) if y.size else 0.5
    w1 = 0.5 / max(1e-9, p1) if p1 > 0 else 0.0
    w0 = 0.5 / max(1e-9, 1.0 - p1) if p1 < 1 else 0.0
    sample_w = np.where(y>0.5, w1, w0).astype(float)

    if Z.shape[0] >= 12 and np.unique(y).size == 2 and Z.shape[1] > 0:
        coef_z, bias = logit_fit_weighted(Z, y, sample_w, l2=1.0, max_iter=160, tol=1e-6)
        st.success(f"FT model trained (n={Z.shape[0]}; base FT≈{p1:.2f}).")
    else:
        coef_z, bias = np.zeros(Z.shape[1]), 0.0
        st.error("Unable to train FT (need ≥12 rows and both classes). Using neutral 50%.")

    p_cal = logit_inv(bias + Z @ coef_z)
    st.session_state.TRAIN_PREDS = p_cal
    st.session_state.GRADE_CUTS = make_grade_cuts()

    st.session_state.ART = {
        "feat_names":[n for (n,_) in ft_defs],
        "mu":mu, "sd":sd,
        "coef_z":coef_z, "bias":bias
    }
    st.success("Learning complete.")

if learn_btn:
    if not uploaded:
        st.error("Upload a workbook first.")
    else:
        try:
            xls = pd.ExcelFile(uploaded)
            if sheet_name not in xls.sheet_names:
                st.error(f"Sheet '{sheet_name}' not found. Available: {xls.sheet_names}")
            else:
                _learn_ft(xls, sheet_name)
        except Exception as e:
            st.error(f"Learning failed: {e}")

# ============================== Inference (FT only) ==============================
def predict_ft_prob(gap_pct: float, pm_vol_m: float, catalyst: float,
                    atr_usd: float, mcap_m: float, rvol: float, pm_dol_m: float,
                    si_pct: float) -> float:
    A = st.session_state.ART or {}
    names = A.get("feat_names") or []; mu = A.get("mu"); sd = A.get("sd")
    coef = A.get("coef_z"); bias = float(A.get("bias", 0.0))
    if not names or coef is None or mu is None or sd is None: return 0.50

    feat_map = {
        "ln1p_pmvol":  lambda: _safe_log(_nz(pm_vol_m,0.0) + 1.0),
        "ln_gapf":     lambda: _safe_log(_nz(gap_pct,0.0)/100.0),
        "catalyst":    lambda: float(_nz(catalyst,0.0)),
        "ln_atr":      lambda: _safe_log(atr_usd),
        "ln_mcap":     lambda: _safe_log(mcap_m),
        "ln1p_rvol":   lambda: _safe_log(_nz(rvol,0.0) + 1.0),
        "ln1p_pmmc":   lambda: _safe_log(_nz(pm_dol_m,0.0)/max(1e-6,_nz(mcap_m,0.0)) + 1.0),
        "ln1p_si":     lambda: _safe_log(_nz(si_pct,0.0) + 1.0),
    }
    vals = [float(feat_map[n]()) for n in names]
    Z = (np.array(vals) - mu) / sd
    Z = np.clip(Z, -3.0, 3.0)
    return float(np.clip(logit_inv(bias + np.dot(Z, coef)), 1e-3, 1-1e-3))

def _grade_for_prob(p: float) -> str:
    grade_cuts = st.session_state.get("GRADE_CUTS", make_grade_cuts())
    if p >= grade_cuts.get("App",0.99): return "A++"
    if p >= grade_cuts.get("Ap",0.97):  return "A+"
    if p >= grade_cuts.get("A",0.90):   return "A"
    if p >= grade_cuts.get("B",0.80):   return "B"
    if p >= grade_cuts.get("C",0.60):   return "C"
    return "D"

# ============================== Tabs ==============================
tab_add, tab_rank = st.tabs(["➕ Add / Score", "📊 Ranking"])

# ============================== Add / Score ==============================
with tab_add:
    # FT training status badge
    _ft_ok = bool(st.session_state.ART)
    st.markdown(
        f'<div style="margin:.25rem 0 1rem 0;">'
        f'  <span style="display:inline-block;padding:.18rem .55rem;border-radius:999px;'
        f'         border:1px solid {_ft_ok and "#16a34a" or "#ef4444"};'
        f'         background:{_ft_ok and "#ecfdf5" or "#fef2f2"};'
        f'         color:{_ft_ok and "#166534" or "#991b1b"}; font-weight:600; font-size:.82rem;">'
        f'    FT model: {_ft_ok and "trained" or "NOT trained"}'
        f'  </span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        # Full-width single-column inputs, catalyst directly under PM $Vol
        ticker   = st.text_input("Ticker", "").strip().upper()
        mcap_m   = input_float("Market Cap (Millions $)", 0.0, min_value=0.0, decimals=2)
        si_pct   = input_float("Short Interest (%)",       0.0, min_value=0.0, decimals=2)
        gap_pct  = input_float("Gap %",                    0.0, min_value=0.0, decimals=1)
        atr_usd  = input_float("ATR ($)",                  0.0, min_value=0.0, decimals=2)
        rvol     = input_float("RVOL",                     0.0, min_value=0.0, decimals=2)
        pm_vol_m = input_float("Premarket Volume (Millions)", 0.0, min_value=0.0, decimals=2)
        pm_dol_m = input_float("Premarket Dollar Volume (Millions $)", 0.0, min_value=0.0, decimals=2)
        catalyst_flag = st.selectbox("Catalyst?", ["No","Yes"], index=0)

        submitted = st.form_submit_button("Add / Score", use_container_width=True)

    if submitted and ticker:
        if not _ft_ok:
            st.error("Train the FT model first (Upload → Learn FT model).")
        else:
            cat = 1.0 if catalyst_flag=="Yes" else 0.0
            p = predict_ft_prob(
                gap_pct=gap_pct, pm_vol_m=pm_vol_m, catalyst=cat,
                atr_usd=atr_usd, mcap_m=mcap_m, rvol=rvol, pm_dol_m=pm_dol_m,
                si_pct=si_pct
            )
            grade = _grade_for_prob(p)

            row = {
                "Ticker": ticker,
                "Level": grade,
                "FT Probability %": round(100.0 * p, 2),
            }
            st.session_state.rows.append(row)
            st.session_state.last = row
            st.session_state.flash = f"Saved {ticker} — Grade {row['Level']} (P={row['FT Probability %']:.2f}%)"
            _rerun()

    l = st.session_state.last if isinstance(st.session_state.last, dict) else {}
    if l:
        st.markdown('<div class="block-divider"></div>', unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric("Last Ticker", l.get("Ticker","—"))
        b.metric("Grade", l.get("Level","—"))
        c.metric("FT Probability", f"{l.get('FT Probability %',0):.2f}%")

# ============================== Ranking ==============================
with tab_rank:
    st.markdown('<div class="section-title">Current Ranking</div>', unsafe_allow_html=True)

    if st.session_state.rows:
        df = pd.DataFrame(st.session_state.rows)
        if "FT Probability %" in df.columns:
            df = df.sort_values("FT Probability %", ascending=False).reset_index(drop=True)

        cols = ["Ticker","Level","FT Probability %"]
        for c in cols:
            if c not in df.columns: df[c] = "" if c in ("Ticker","Level") else 0.0

        st.dataframe(
            df[cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Level": st.column_config.TextColumn("Grade"),
                "FT Probability %": st.column_config.NumberColumn("FT Probability %", format="%.2f"),
            }
        )

        # Delete rows (top 12)
        st.markdown("#### Delete rows")
        del_cols = st.columns(4)
        head12 = df.head(12).reset_index(drop=True)
        for i, r in head12.iterrows():
            with del_cols[i % 4]:
                label = r.get("Ticker", f"Row {i+1}")
                if st.button(f"🗑️ {label}", key=f"del_{i}", use_container_width=True):
                    keep = df.drop(index=i).reset_index(drop=True)
                    st.session_state.rows = keep.to_dict(orient="records")
                    _rerun()

        # Downloads
        st.download_button(
            "Download CSV (Ranking)",
            df[cols].to_csv(index=False).encode("utf-8"),
            "ranking.csv", "text/csv", use_container_width=True
        )

        # Markdown table
        st.markdown("### 📋 Ranking (Markdown view)")
        md_text = df_to_markdown_table(df, cols)
        st.code(md_text, language="markdown")
        st.download_button(
            "Download Markdown",
            md_text.encode("utf-8"),
            "ranking.md", "text/markdown", use_container_width=True
        )

        c1, _ = st.columns([0.25,0.75])
        with c1:
            if st.button("Clear Ranking", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last = {}
                _rerun()
    else:
        st.info("No rows yet. Add a stock in the **Add / Score** tab.")
