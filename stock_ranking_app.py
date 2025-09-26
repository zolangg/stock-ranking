import streamlit as st
import pandas as pd
import re

# =========================== Page ===========================
st.set_page_config(page_title="Model Stock Tables (DB Upload)", layout="wide")
st.title("Model Stock Tables (DB Upload)")

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
            if isinstance(v, (int, float)):
                cells.append(f"{float(v):.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ==================== Column mapping helpers =================
def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    return (s.replace("$","")
             .replace("%","")
             .replace("("," ")
             .replace(")"," ")
             .replace("’","")
             .replace("'",""))

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    nm = {c: _norm(c) for c in cols}
    # exact normalized match
    for cand in candidates:
        n = _norm(cand)
        for c in cols:
            if nm[c] == n:
                return c
    # substring fallback
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
    # Try numeric first
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return sn.fillna(0.0).clip(0, 1)
    # Fallback: text parsing
    sm = s.astype(str).str.strip().str.lower()
    true_vals = {"y","yes","true","t","1","✓","✔","x"}
    return sm.apply(lambda x: 1.0 if x in true_vals else 0.0)

# ========================= UI ===============================
st.subheader("Upload workbook")
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
sheet_main = st.text_input("Data sheet name", "PMH BO Merged")

build_btn = st.button("Build Model Stocks", use_container_width=True)

# ======================= Build models =======================
def build_model_rows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build 1-row model tables for:
      - FT1: FT == 1
      - FT0: FT == 0 (if present)
      - MaxPush: top 10% by 'Max Push Daily' (if present; fallback to top 25% for very small data)
    Returns dict of name -> DataFrame(one row).
    """
    out: dict[str, pd.DataFrame] = {}

    # Map required columns
    col_ft    = pick_col(df, ["ft"])
    col_mc    = pick_col(df, ["market cap (m)","mcap m","mcap","marketcap m"])
    col_float = pick_col(df, ["float m shares","public float (m)","float (m)","float"])
    col_si    = pick_col(df, ["short interest %","short float %","si","short interest (float) %"])
    col_gap   = pick_col(df, ["gap %","gap%","premarket gap","gap"])
    col_atr   = pick_col(df, ["atr","atr $","atr$","atr (usd)"])
    col_rvol  = pick_col(df, ["rvol @ bo","rvol","relative volume"])
    col_pmvol = pick_col(df, ["pm vol (m)","pm volume (m)","premarket vol (m)","pm shares (m)"])
    col_pmdol = pick_col(df, ["pm $vol (m)","pm dollar vol (m)","pm $ volume (m)","pm $vol"])
    col_cat   = pick_col(df, ["catalyst","news","pr"])  # Yes/No field
    col_push  = pick_col(df, ["max push daily","max push %","maxpush","max push"])

    need_names = ["MarketCap","Float","SI","Gap%","ATR","RVOL","PM Vol (M)","PM $Vol (M)"]
    need_cols  = [col_mc, col_float, col_si, col_gap, col_atr, col_rvol, col_pmvol, col_pmdol]
    if not all(c is not None for c in need_cols):
        missing = [n for n, c in zip(need_names, need_cols) if c is None]
        st.error("Missing required columns: " + ", ".join(missing))
        return out

    # Build numeric base frame with unified names
    base = pd.DataFrame({
        "MarketCap_M$": pd.to_numeric(df[col_mc], errors="coerce"),
        "Float_M":      pd.to_numeric(df[col_float], errors="coerce"),
        "ShortInt_%":   pd.to_numeric(df[col_si], errors="coerce"),
        "Gap_%":        pd.to_numeric(df[col_gap], errors="coerce"),
        "ATR_$":        pd.to_numeric(df[col_atr], errors="coerce"),
        "RVOL":         pd.to_numeric(df[col_rvol], errors="coerce"),
        "PM_Vol_M":     pd.to_numeric(df[col_pmvol], errors="coerce"),
        "PM_$Vol_M$":   pd.to_numeric(df[col_pmdol], errors="coerce"),
    })

    # Catalyst as 0/1 (Yes/No). If missing, all zeros.
    if col_cat:
        base["Catalyst01"] = parse_catalyst_series(df[col_cat])
    else:
        base["Catalyst01"] = 0.0

    # FT masks
    if col_ft and col_ft in df.columns:
        ft_series = pd.to_numeric(df[col_ft], errors="coerce")
        mask_ft1 = (ft_series == 1)
        mask_ft0 = (ft_series == 0)
    else:
        mask_ft1 = pd.Series([False]*len(df))
        mask_ft0 = pd.Series([False]*len(df))

    # Helper to compute a 1-row model from a subset (by median; Catalyst as %Yes)
    def model_from_subset(sub: pd.DataFrame, label: str) -> pd.DataFrame | None:
        sub_num = sub.dropna(how="all")
        if sub_num.empty:
            return None
        med = sub_num[[
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL","PM_Vol_M","PM_$Vol_M$"
        ]].median(numeric_only=True)

        # Derived from medians (robust for small N)
        fr_x = (med["PM_Vol_M"] / med["Float_M"]) if med.get("Float_M", 0) > 0 else 0.0
        pmmc_pct = (med["PM_$Vol_M$"] / med["MarketCap_M$"] * 100.0) if med.get("MarketCap_M$", 0) > 0 else 0.0

        cat_pct_yes = float(sub_num["Catalyst01"].mean() * 100.0) if "Catalyst01" in sub_num else 0.0

        row = pd.DataFrame([{
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
        return row

    # Build FT=1 model
    sub_ft1 = base[mask_ft1] if mask_ft1.any() else pd.DataFrame(columns=base.columns)
    m_ft1 = model_from_subset(sub_ft1, "FT1")
    if m_ft1 is not None:
        out["FT=1"] = m_ft1

    # Build FT=0 model (if present)
    sub_ft0 = base[mask_ft0] if mask_ft0.any() else pd.DataFrame(columns=base.columns)
    m_ft0 = model_from_subset(sub_ft0, "FT0")
    if m_ft0 is not None:
        out["FT=0"] = m_ft0

    # Build MaxPush model (top decile or, if very small, top quartile)
    if col_push:
        push_vals = pd.to_numeric(df[col_push], errors="coerce")
        if push_vals.notna().any():
            # prefer 90th percentile; if sample too small, fall back to 75th to ensure some rows
            pctl = 0.90 if push_vals.notna().sum() >= 40 else 0.75
            thr = push_vals.quantile(pctl)
            mask_push = (push_vals >= thr)
            sub_push = base[mask_push.fillna(False)]
            m_push = model_from_subset(sub_push, "MaxPush")
            if m_push is not None:
                out["Max Push"] = m_push

    return out

# ======================== Run build =========================
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
                models = build_model_rows(raw)
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

        except Exception as e:
            st.error(f"Failed to build models: {e}")
