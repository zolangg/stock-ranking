import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from itertools import combinations

# =========================== Page ===========================
st.set_page_config(page_title="Premarket Table + Model Stocks", layout="wide")
st.title("Premarket Table + Model Stocks")

# ======================= Markdown helper ====================
def df_to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a subset of columns as a Markdown table (2 decimals for numbers)."""
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "| (no data) |\n| --- |"
    sub = df.loc[:, keep].copy().fillna("")
    header = "| " + " | ".join(keep) + " |"
    sep    = "| " + " | ".join(["---"] * len(keep)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        cells = []
        for c in keep:
            v = row[c]
            if isinstance(v, (int, float, np.floating)):
                cells.append(f"{float(v):.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ======================= Session state ======================
if "rows" not in st.session_state: st.session_state.rows = []
if "last_warnings" not in st.session_state: st.session_state.last_warnings = []
if "model_groups" not in st.session_state: st.session_state.model_groups = {}  # for divergence tests
if "models" not in st.session_state: st.session_state.models = {}

# ================= Qualitative (short labels) ================
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
        "short": [
            "Full reversal", "Choppy reversal", "25–50% retrace",
            "Sideways (top 25%)", "Uptrend (deep PBs)", "Uptrend (mod PBs)",
            "Clean uptrend",
        ],
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
        "short": [
            "Fails levels", "Brief hold; loses", "Holds support; capped",
            "Breaks then loses", "Holds 1 major", "Holds several", "Above all",
        ],
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
        "short": [
            "Accel downtrend", "Downtrend", "Flattening",
            "Base", "Higher low", "BO from base", "Sustained uptrend",
        ],
    },
]

# ======================= Helpers (Upload tab) =================
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return (s.replace("$","").replace("%","").replace("("," ").replace(")"," ")
             .replace("’","").replace("'",""))

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    nm = {c: _norm(c) for c in cols}
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if n in nm[c]:
                return c
    return None

def parse_catalyst_series(s: pd.Series) -> pd.Series:
    """Parse Yes/No / booleans / 0/1 into 0/1 numeric; missing -> 0."""
    if s is None:
        return pd.Series([0.0]*0, dtype=float)
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return sn.fillna(0.0).clip(0, 1)
    sm = s.astype(str).str.strip().str.lower()
    true_vals = {"y","yes","true","t","1","✓","✔","x"}
    return sm.apply(lambda x: 1.0 if x in true_vals else 0.0)

def _to_float_strip_percent(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", "")
    if "%" in s:
        try:
            return float(s.replace("%", "").strip())
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def parse_percent_series(series: pd.Series) -> pd.Series:
    """
    Handles:
      - '90%' -> 90
      - 0.9 -> 90 (heuristic: if 90th percentile of abs <= 3, treat as fraction)
    """
    s = series.apply(_to_float_strip_percent)
    if not s.notna().any():
        return s
    abs_vals = s.abs()
    q90 = np.nanquantile(abs_vals, 0.90)
    if q90 <= 3.0:
        s = s.where(abs_vals > 3.0, s * 100.0)
    return s

def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize units to:
      - MarketCap_M$: millions of dollars
      - Float_M: millions of shares
      - ShortInt_%: percent (0..100)
      - Gap_%: percent (0..100)
      - ATR_$: dollars
      - RVOL: unitless
      - PM_Vol_M: millions of shares
      - PM_$Vol_M$: millions of dollars
    """
    out = df.copy()

    # percents
    if "Gap_%" in out.columns:
        out["Gap_%"] = parse_percent_series(out["Gap_%"])
    if "ShortInt_%" in out.columns:
        out["ShortInt_%"] = parse_percent_series(out["ShortInt_%"])

    def _med(name):
        s = pd.to_numeric(out[name], errors="coerce")
        return float(s.median()) if s.notna().any() else np.nan

    # Market cap -> M$
    if "MarketCap_M$" in out.columns:
        mc_med = _med("MarketCap_M$")
        if pd.notna(mc_med) and mc_med >= 1e6:
            out["MarketCap_M$"] = pd.to_numeric(out["MarketCap_M$"], errors="coerce") / 1_000_000.0

    # Float -> M shares
    if "Float_M" in out.columns:
        fl_med = _med("Float_M")
        if pd.notna(fl_med) and fl_med >= 1e6:
            out["Float_M"] = pd.to_numeric(out["Float_M"], errors="coerce") / 1_000_000.0

    # PM vol -> M shares
    if "PM_Vol_M" in out.columns:
        pv_med = _med("PM_Vol_M")
        if pd.notna(pv_med) and pv_med >= 1_000_000.0:
            out["PM_Vol_M"] = pd.to_numeric(out["PM_Vol_M"], errors="coerce") / 1_000_000.0

    # PM $vol -> M$
    if "PM_$Vol_M$" in out.columns:
        pd_med = _med("PM_$Vol_M$")
        if pd.notna(pd_med) and pd_med >= 10_000_000.0:
            out["PM_$Vol_M$"] = pd.to_numeric(out["PM_$Vol_M$"], errors="coerce") / 1_000_000.0

    return out

# ================= Build models + store raw groups =================
def build_models_and_groups(df: pd.DataFrame):
    """
    Returns:
      models: dict[name] -> 1-row DataFrame (medians + FR_x + PM$Vol/MC_% + Catalyst_%Yes)
      groups: dict[name] -> normalized full DataFrame with raw rows for that subgroup
    """
    models: dict[str, pd.DataFrame] = {}
    groups: dict[str, pd.DataFrame]  = {}

    col_ft    = pick_col(df, ["ft"])
    col_mc    = pick_col(df, ["market cap (m)","mcap m","mcap","marketcap m"])
    col_float = pick_col(df, ["float m shares","public float (m)","float (m)","float"])
    col_si    = pick_col(df, ["short interest %","short float %","si","short interest (float) %"])
    col_gap   = pick_col(df, ["gap %","gap%","premarket gap","gap"])
    col_atr   = pick_col(df, ["atr","atr $","atr$","atr (usd)"])
    col_rvol  = pick_col(df, ["rvol @ bo","rvol","relative volume"])
    col_pmvol = pick_col(df, ["pm vol (m)","pm volume (m)","premarket vol (m)","pm shares (m)"])
    col_pmdol = pick_col(df, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
    col_cat   = pick_col(df, ["catalyst","news","pr"])
    col_push  = pick_col(df, ["max push daily","max push %","maxpush","max push"])

    need_names = ["MarketCap","Float","SI","Gap%","ATR","RVOL","PM Vol (M)","PM $Vol (M)"]
    need_cols  = [col_mc, col_float, col_si, col_gap, col_atr, col_rvol, col_pmvol, col_pmdol]
    if not all(c is not None for c in need_cols):
        missing = [n for n, c in zip(need_names, need_cols) if c is None]
        st.error("Missing required columns: " + ", ".join(missing))
        return models, groups

    base = pd.DataFrame({
        "MarketCap_M$": pd.to_numeric(df[col_mc], errors="coerce"),
        "Float_M":      pd.to_numeric(df[col_float], errors="coerce"),
        "ShortInt_%":   df[col_si],
        "Gap_%":        df[col_gap],
        "ATR_$":        pd.to_numeric(df[col_atr], errors="coerce"),
        "RVOL":         pd.to_numeric(df[col_rvol], errors="coerce"),
        "PM_Vol_M":     pd.to_numeric(df[col_pmvol], errors="coerce"),
        "PM_$Vol_M$":   pd.to_numeric(df[col_pmdol], errors="coerce"),
    })
    base["Catalyst01"] = parse_catalyst_series(df[col_cat]) if col_cat else 0.0
    base = normalize_units(base)

    # FT masks
    if col_ft and col_ft in df.columns:
        ft_series = pd.to_numeric(df[col_ft], errors="coerce")
        mask_ft1 = (ft_series == 1)
        mask_ft0 = (ft_series == 0)
    else:
        mask_ft1 = pd.Series([False]*len(df))
        mask_ft0 = pd.Series([False]*len(df))

    # group DataFrames (retain rows)
    sub_ft1 = base[mask_ft1].copy() if mask_ft1.any() else pd.DataFrame(columns=base.columns)
    sub_ft0 = base[mask_ft0].copy() if mask_ft0.any() else pd.DataFrame(columns=base.columns)

    # Max Push group
    if col_push:
        push_vals = pd.to_numeric(df[col_push], errors="coerce")
        if push_vals.notna().any():
            pctl = 0.90 if push_vals.notna().sum() >= 40 else 0.75
            thr = push_vals.quantile(pctl)
            mask_push = (push_vals >= thr)
            sub_push = base[mask_push.fillna(False)].copy()
        else:
            sub_push = pd.DataFrame(columns=base.columns)
    else:
        sub_push = pd.DataFrame(columns=base.columns)

    # save raw groups for divergence
    if not sub_ft1.empty: groups["FT=1"] = sub_ft1
    if not sub_ft0.empty: groups["FT=0"] = sub_ft0
    if not sub_push.empty: groups["Max Push"] = sub_push

    def one_row_model(sub: pd.DataFrame, label: str) -> pd.DataFrame | None:
        sub_num = sub.dropna(how="all")
        if sub_num.empty:
            return None
        med = sub_num[[
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$"
        ]].median(numeric_only=True)

        fr_x = (med["PM_Vol_M"] / med["Float_M"]) if med.get("Float_M", 0) > 0 else 0.0
        pmmc_pct = (med["PM_$Vol_M$"] / med["MarketCap_M$"] * 100.0) if med.get("MarketCap_M$", 0) > 0 else 0.0
        cat_pct_yes = float(sub_num["Catalyst01"].mean() * 100.0) if "Catalyst01" in sub_num else 0.0

        return pd.DataFrame([{
            "Ticker": f"MODEL_{label}",
            "MarketCap_M$": float(med.get("MarketCap_M$", 0.0)),
            "Float_M":      float(med.get("Float_M", 0.0)),
            "ShortInt_%":   float(med.get("ShortInt_%", 0.0)),
            "Gap_%":        float(med.get("Gap_%", 0.0)),
            "ATR_$":        float(med.get("ATR_$", 0.0)),
            "RVOL":         float(med.get("RVOL", 0.0)),
            "PM_Vol_M":     float(med.get("PM_Vol_M", 0.0)),
            "PM_$Vol_M$":   float(med.get("PM_$Vol_M$", 0.0)),
            "FR_x":         float(fr_x),
            "PM$Vol/MC_%":  float(pmmc_pct),
            "Catalyst_%Yes": float(cat_pct_yes),
        }])

    m_ft1  = one_row_model(sub_ft1, "FT1")
    m_ft0  = one_row_model(sub_ft0, "FT0")
    m_push = one_row_model(sub_push, "MaxPush")

    if m_ft1 is not None:  models["FT=1"] = m_ft1
    if m_ft0 is not None:  models["FT=0"] = m_ft0
    if m_push is not None: models["Max Push"] = m_push

    return models, groups

# =================== Stats: KW (permutation) & Cliff's δ ===================
def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1-based)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # tie groups
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # average of ranks i+1..j+1
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def kruskal_wallis_H_permutation(groups_list, perms: int = 1999, random_state: int = 42):
    """
    Compute Kruskal–Wallis H and a permutation p-value by shuffling group labels.
    groups_list: list of 1D numeric arrays (k groups)
    """
    rng = np.random.default_rng(random_state)
    # Filter valid groups
    clean = [np.asarray(g, dtype=float) for g in groups_list if len(pd.Series(g).dropna()) >= 2]
    clean = [pd.Series(g).dropna().to_numpy(dtype=float) for g in clean]
    k = len(clean)
    if k < 2:
        return np.nan, np.nan

    # Observed H
    all_vals = np.concatenate(clean, axis=0)
    ranks = rankdata_average_ties(all_vals)
    sizes = [len(g) for g in clean]
    idx_start = np.cumsum([0] + sizes[:-1])
    Ri_bar = []
    for i, n in enumerate(sizes):
        r = ranks[idx_start[i]: idx_start[i] + n]
        Ri_bar.append(r.mean())
    N = len(all_vals)
    H_obs = (12.0 / (N * (N + 1))) * sum(n * (Ri**2) for n, Ri in zip(sizes, Ri_bar)) - 3.0 * (N + 1)

    # Permutation p-value
    p_count = 1  # add-one smoothing
    for _ in range(perms):
        perm = rng.permutation(N)
        ranks_perm = ranks[perm]  # ranks are fixed; shuffling group labels equivalent
        Ri_bar_perm = []
        start = 0
        for n in sizes:
            r = ranks_perm[start:start+n]
            Ri_bar_perm.append(r.mean())
            start += n
        H_perm = (12.0 / (N * (N + 1))) * sum(n * (Ri**2) for n, Ri in zip(sizes, Ri_bar_perm)) - 3.0 * (N + 1)
        if H_perm >= H_obs - 1e-12:
            p_count += 1
    p_perm = p_count / (perms + 1)
    return float(H_obs), float(p_perm)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (a vs b). Range [-1,1]."""
    a = pd.Series(a).dropna().to_numpy(dtype=float)
    b = pd.Series(b).dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Efficient pairwise compare using broadcasting (may be fine for small samples)
    A = a[:, None]
    comp = (A > b).sum() - (A < b).sum()
    return float(comp) / float(len(a) * len(b))

# =================== Robust divergence renderer (statistical) ===================
def render_divergence_tables_stat(models: dict, groups: dict, perms: int = 4999):
    """
    Shows significant rows using:
      - Kruskal–Wallis H with permutation p-value
      - Max pairwise |Cliff’s δ|
    Significance rule (tunable below). Falls back to Top-K practical differences
    by effect size if nothing passes the gate.
    """
    try:
        if not isinstance(models, dict) or not models:
            return

        st.markdown("### 🔎 Divergence across Model Stocks (statistical)")

        model_names = [name for name in models.keys() if name in groups and not groups[name].empty]
        if len(model_names) < 2:
            st.info("Need at least two model groups with data (e.g., FT=1 and Max Push) to compare.")
            return

        # ---- Tunables ----
        THRESH_P = 0.20          # was 0.10
        THRESH_DELTA = 0.40      # was 0.50
        MIN_N = 8                # skip vars where any subgroup has < 8 non-NA
        TOP_K = 8                # fallback: show top-K by effect size

        var_list = [
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst01"
        ]

        def _series_for(gdf: pd.DataFrame, var: str) -> pd.Series:
            if var in gdf.columns:
                return pd.to_numeric(gdf[var], errors="coerce")
            if var == "FR_x" and {"PM_Vol_M","Float_M"}.issubset(gdf.columns):
                a = pd.to_numeric(gdf["PM_Vol_M"], errors="coerce")
                b = pd.to_numeric(gdf["Float_M"], errors="coerce").replace(0, np.nan)
                return (a / b).replace([np.inf, -np.inf], np.nan)
            if var == "PM$Vol/MC_%" and {"PM_$Vol_M$","MarketCap_M$"}.issubset(gdf.columns):
                a = pd.to_numeric(gdf["PM_$Vol_M$"], errors="coerce")
                b = pd.to_numeric(gdf["MarketCap_M$"], errors="coerce").replace(0, np.nan)
                return (a / b * 100.0).replace([np.inf, -np.inf], np.nan)
            return pd.Series(dtype=float)

        # --- collect stats per var ---
        candidates = []
        for var in var_list:
            data_by_group = {}
            medians_by_group = {}
            sizes = []
            for name in model_names:
                s = _series_for(groups[name], var).dropna()
                if len(s) > 0:
                    data_by_group[name] = s.to_numpy(float)
                    medians_by_group[name] = float(s.median())
                    sizes.append(len(s))
                else:
                    sizes.append(0)

            if len([n for n in data_by_group]) < 2:
                continue
            if any(n < MIN_N for n in sizes):
                # skip underpowered variable
                continue

            # KW (permutation)
            groups_list = [data_by_group[n] for n in data_by_group.keys()]
            H, p_perm = kruskal_wallis_H_permutation(groups_list, perms=perms)

            # Pairwise Cliff’s deltas
            deltas = []
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    a_name, b_name = model_names[i], model_names[j]
                    if a_name in data_by_group and b_name in data_by_group:
                        d = cliffs_delta(data_by_group[a_name], data_by_group[b_name])
                        deltas.append((a_name, b_name, d))
            if deltas:
                best = max(deltas, key=lambda t: abs(t[2]) if pd.notna(t[2]) else -1)
                pair = f"{best[0]} vs {best[1]}"
                max_abs_delta = abs(best[2]) if pd.notna(best[2]) else np.nan
            else:
                pair, max_abs_delta = "", np.nan

            meds = list(medians_by_group.values())
            vmin, vmax = float(np.nanmin(meds)), float(np.nanmax(meds))
            rng  = float(vmax - vmin)
            fold = float(vmax / vmin) if vmin > 0 else np.nan

            significant = (pd.notna(p_perm) and p_perm < THRESH_P) or (pd.notna(max_abs_delta) and max_abs_delta >= THRESH_DELTA)

            row = {
                "Variable": var, "H": H, "p_perm": p_perm, "Max|Cliffδ|": max_abs_delta, "Pair": pair,
                "Min": vmin, "Max": vmax, "Range": rng, "Fold": fold, "significant": significant
            }
            for name in model_names:
                if name in medians_by_group:
                    row[name] = medians_by_group[name]
            candidates.append(row)

        if not candidates:
            st.info("No variables with enough data per subgroup (min N unmet).")
            return

        dfc = pd.DataFrame(candidates).replace([np.inf, -np.inf], np.nan)

        sig_df = dfc[dfc["significant"] == True].copy()
        if sig_df.empty:
            st.warning("No variables met the relaxed rule. Showing Top differences by effect size (|Cliff’s δ|).")
            show_df = dfc.sort_values(["Max|Cliffδ|","Range","Fold"], ascending=[False, False, False]).head(TOP_K).copy()
        else:
            show_df = sig_df.sort_values(["p_perm","Max|Cliffδ|"], ascending=[True, False]).copy()

        display_cols = ["Variable"] + [m for m in model_names if m in show_df.columns] + \
                       ["Min","Max","Range","Fold","H","p_perm","Max|Cliffδ|","Pair"]
        display_cols = [c for c in display_cols if c in show_df.columns]

        col_cfg = {"Variable": st.column_config.TextColumn("Variable"),
                   "Pair": st.column_config.TextColumn("Strongest Pair")}
        num_cols = set(show_df.columns) - {"Variable","Pair","significant"}
        for c in num_cols:
            col_cfg[c] = st.column_config.NumberColumn(c if c not in {"Max|Cliffδ|"} else "Max |Cliff’s δ|", format="%.2f")

        st.dataframe(show_df[display_cols], use_container_width=True, hide_index=True, column_config=col_cfg)
        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(show_df, display_cols), language="markdown")

    except Exception as e:
        st.error("Divergence view failed. See details below.")
        st.exception(e)

# =========================== Tabs ===========================
tab_manual, tab_models = st.tabs(["📝 Manual Table", "📥 Upload & Model Stocks"])

# =============================================================================
# TAB 1 — Manual Table (inputs + derived + qualitative tags + sanity flags)
# =============================================================================
with tab_manual:
    st.subheader("Add Stock")
    with st.form("add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1.1, 1.1, 1.1])

        with c1:
            ticker  = st.text_input("Ticker", "").strip().upper()
            mc_m    = st.number_input("Market Cap (M$)", 0.0, step=0.01, format="%.2f")
            float_m = st.number_input("Public Float (M)", 0.0, step=0.01, format="%.2f")
            si_pct  = st.number_input("Short Interest (%)", 0.0, step=0.01, format="%.2f")

        with c2:
            gap_pct = st.number_input("Gap %", 0.0, step=0.1, format="%.1f",
                                      help="Enter as percent (e.g., 45 for +45%, not 0.45).")
            atr_usd = st.number_input("ATR ($)", 0.0, step=0.01, format="%.2f")
            rvol    = st.number_input("RVOL", 0.0, step=0.01, format="%.2f")

        with c3:
            pm_vol  = st.number_input("Premarket Volume (M)", 0.0, step=0.01, format="%.2f")
            pm_dol  = st.number_input("Premarket Dollar Vol (M$)", 0.0, step=0.01, format="%.2f")
            catalyst = st.slider("Catalyst (−1…+1)", -1.0, 1.0, 0.0, 0.05)
            dilution = st.slider("Dilution (−1…+1)", -1.0, 1.0, 0.0, 0.05)

        st.markdown("---")
        st.subheader("Qualitative Context")
        q_cols = st.columns(3)
        qual_tags = {}
        for i, crit in enumerate(QUAL_CRITERIA):
            with q_cols[i % 3]:
                choice = st.radio(
                    crit["question"],
                    options=list(enumerate(crit["options"], 1)),
                    format_func=lambda x: f"{x[0]}. {x[1]}",
                    key=f"qual_{crit['name']}"
                )
                lvl = int(choice[0])
                qual_tags[crit["name"]] = f"{lvl}-{crit['short'][lvl-1]}"

        submitted = st.form_submit_button("Add", use_container_width=True)

    if submitted and ticker:
        fr_x = (pm_vol/float_m) if float_m > 0 else 0.0
        pmmc_pct = (pm_dol/mc_m*100.0) if mc_m > 0 else 0.0
        row = {
            "Ticker": ticker,
            "MarketCap_M$": mc_m,
            "Float_M": float_m,
            "ShortInt_%": si_pct,
            "Gap_%": gap_pct,
            "ATR_$": atr_usd,
            "RVOL": rvol,
            "PM_Vol_M": pm_vol,
            "PM_$Vol_M$": pm_dol,
            "Catalyst": catalyst,
            "Dilution": dilution,
            "FR_x": fr_x,
            "PM$Vol/MC_%": pmmc_pct,
            "GapStruct": qual_tags.get("GapStruct",""),
            "LevelStruct": qual_tags.get("LevelStruct",""),
            "Monthly": qual_tags.get("Monthly",""),
        }
        st.session_state.rows.append(row)

        # ---- Sanity flags for manual entry ----
        flags = []
        if gap_pct > 0 and gap_pct < 1.0:
            flags.append("Gap % looks < 1 — did you mean percent (e.g., 45) not fraction (0.45)?")
        if si_pct > 0 and si_pct < 1.0:
            flags.append("Short Interest % looks < 1 — enter as percent (e.g., 12.5) not 0.125.")
        if float_m > 0 and pm_vol/float_m > 30:
            flags.append(f"PM Float Rotation = {pm_vol/float_m:.1f}× — unusually high, check units.")
        if mc_m > 0 and (pm_dol/mc_m*100.0) > 200:
            flags.append(f"PM $Vol / MC = {pm_dol/mc_m*100.0:.1f}% — unusually high, check units.")
        st.session_state.last_warnings = flags

        do_rerun()

    st.subheader("Table")
    if st.session_state.rows:
        if st.session_state.last_warnings:
            for f in st.session_state.last_warnings:
                st.warning(f)

        df = pd.DataFrame(st.session_state.rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "MarketCap_M$": st.column_config.NumberColumn("Market Cap (M$)", format="%.2f"),
                "Float_M": st.column_config.NumberColumn("Public Float (M)", format="%.2f"),
                "ShortInt_%": st.column_config.NumberColumn("Short Interest (%)", format="%.2f"),
                "Gap_%": st.column_config.NumberColumn("Gap %", format="%.2f"),
                "ATR_$": st.column_config.NumberColumn("ATR ($)", format="%.2f"),
                "RVOL": st.column_config.NumberColumn("RVOL", format="%.2f"),
                "PM_Vol_M": st.column_config.NumberColumn("Premarket Vol (M)", format="%.2f"),
                "PM_$Vol_M$": st.column_config.NumberColumn("Premarket $Vol (M$)", format="%.2f"),
                "Catalyst": st.column_config.NumberColumn("Catalyst", format="%.2f"),
                "Dilution": st.column_config.NumberColumn("Dilution", format="%.2f"),
                "FR_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.2f"),
                "PM$Vol/MC_%": st.column_config.NumberColumn("PM $Vol / MC (%)", format="%.2f"),
                "GapStruct": st.column_config.TextColumn("GapStruct"),
                "LevelStruct": st.column_config.TextColumn("LevelStruct"),
                "Monthly": st.column_config.TextColumn("Monthly"),
            }
        )

        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "premarket_table.csv", "text/csv",
                use_container_width=True
            )
        with c2:
            if st.button("Clear Table", use_container_width=True):
                st.session_state.rows = []
                st.session_state.last_warnings = []
                do_rerun()

        st.markdown("### 📋 Table (Markdown)")
        cols = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                "PM_Vol_M","PM_$Vol_M$","Catalyst","Dilution","FR_x","PM$Vol/MC_%",
                "GapStruct","LevelStruct","Monthly"]
        st.code(df_to_markdown_table(df, cols), language="markdown")
    else:
        st.info("Add a stock above to populate the table.")

# =============================================================================
# TAB 2 — Upload & Model Stocks + Statistical Divergence
# =============================================================================
with tab_models:
    st.subheader("Upload workbook")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    sheet_main = st.text_input("Data sheet name", "PMH BO Merged")
    build_btn = st.button("Build Model Stocks", use_container_width=True)

    if build_btn:
        if not uploaded:
            st.error("Upload a workbook first.")
        else:
            try:
                xls = pd.ExcelFile(uploaded)
                if sheet_main not in xls.sheet_names:
                    st.error(f"Sheet '{sheet_main}' not found. Available: {xls.sheet_names}")
                else:
                    raw = pd.read_excel(xls, sheet_main)
                    models, groups = build_models_and_groups(raw)
                    st.session_state.models = models
                    st.session_state.model_groups = groups

                    if not models:
                        st.warning("No model tables could be built. Check required columns and FT/Max Push presence.")
                    else:
                        st.success(f"Built {len(models)} model table(s).")

                        col_cfg = {
                            "Ticker": st.column_config.TextColumn("Ticker"),
                            "MarketCap_M$": st.column_config.NumberColumn("Market Cap (M$)", format="%.2f"),
                            "Float_M": st.column_config.NumberColumn("Public Float (M)", format="%.2f"),
                            "ShortInt_%": st.column_config.NumberColumn("Short Interest (%)", format="%.2f"),
                            "Gap_%": st.column_config.NumberColumn("Gap %", format="%.2f"),
                            "ATR_$": st.column_config.NumberColumn("ATR ($)", format="%.2f"),
                            "RVOL": st.column_config.NumberColumn("RVOL", format="%.2f"),
                            "PM_Vol_M": st.column_config.NumberColumn("Premarket Vol (M)", format="%.2f"),
                            "PM_$Vol_M$": st.column_config.NumberColumn("Premarket $Vol (M$)", format="%.2f"),
                            "FR_x": st.column_config.NumberColumn("PM Float Rotation (×)", format="%.2f"),
                            "PM$Vol/MC_%": st.column_config.NumberColumn("PM $Vol / MC (%)", format="%.2f"),
                            "Catalyst_%Yes": st.column_config.NumberColumn("Catalyst (%Yes)", format="%.2f"),
                        }
                        cols_order = ["Ticker","MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
                                      "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst_%Yes"]

                        for name, dfm in models.items():
                            st.markdown(f"### Model Stock — {name}")
                            st.dataframe(dfm, use_container_width=True, hide_index=True, column_config=col_cfg)

                            c1, c2 = st.columns([0.35, 0.65])
                            with c1:
                                st.download_button(
                                    f"Download CSV ({name})",
                                    dfm.to_csv(index=False).encode("utf-8"),
                                    f"model_{_norm(name).replace(' ','_')}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                            with c2:
                                st.markdown("**Markdown**")
                                st.code(df_to_markdown_table(dfm, cols_order), language="markdown")

                        # Statistical divergence (only significant rows)
                        render_divergence_tables_stat(st.session_state.models, st.session_state.model_groups, perms=1999)

            except Exception as e:
                st.error(f"Failed to build models: {e}")
