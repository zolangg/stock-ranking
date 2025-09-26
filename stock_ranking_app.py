import numpy as np
import pandas as pd
import streamlit as st

def render_divergence_tables(models: dict):
    """Robust divergence view with full error surfacing."""
    try:
        if not isinstance(models, dict) or not models:
            return

        st.markdown("### 🔎 Divergence across Model Stocks")

        # only keep valid, non-empty model dataframes
        model_names = [
            name for name, dfm in models.items()
            if isinstance(dfm, pd.DataFrame) and not dfm.empty
        ]
        if len(model_names) < 2:
            st.info("Need at least two model tables (e.g., FT=1 and Max Push) to compare.")
            return

        var_list = [
            "MarketCap_M$","Float_M","ShortInt_%","Gap_%","ATR_$","RVOL",
            "PM_Vol_M","PM_$Vol_M$","FR_x","PM$Vol/MC_%","Catalyst_%Yes"
        ]

        def _to_float(x):
            try:
                return float(pd.to_numeric(x, errors="coerce"))
            except Exception:
                return np.nan

        comp_rows = []
        for var in var_list:
            row = {"Variable": var}
            vals = []
            for mname in model_names:
                v = np.nan
                dfm = models.get(mname)
                if isinstance(dfm, pd.DataFrame) and not dfm.empty and (var in dfm.columns):
                    v = _to_float(dfm[var].iloc[0])
                row[mname] = v
                vals.append(v)

            s = pd.Series(vals, index=model_names, dtype="float64").replace([np.inf, -np.inf], np.nan)
            valid = s.dropna()

            if len(valid) >= 2:
                vmin = float(valid.min())
                vmax = float(valid.max())
                frng = float(vmax - vmin)
                fold = float(vmax / vmin) if vmin > 0 else np.nan  # avoid inf

                pct_vars = {"ShortInt_%","Gap_%","PM$Vol/MC_%","Catalyst_%Yes"}
                if var in pct_vars:
                    significant = (abs(frng) >= 15.0) or (pd.notna(fold) and fold >= 1.5)
                elif var in {"FR_x","RVOL"}:
                    significant = pd.notna(fold) and (fold >= 1.5)
                else:
                    significant = pd.notna(fold) and (fold >= 2.0)

                row.update({
                    "Min": vmin, "Max": vmax, "Range": frng,
                    "Fold": fold, "Significant": "Yes" if significant else ""
                })
            else:
                row.update({"Min": np.nan, "Max": np.nan, "Range": np.nan, "Fold": np.nan, "Significant": ""})

            comp_rows.append(row)

        comp_df = pd.DataFrame(comp_rows).replace([np.inf, -np.inf], np.nan)

        # ensure metric columns exist (even if all NaN)
        for col in ["Min","Max","Range","Fold","Significant"]:
            if col not in comp_df.columns:
                comp_df[col] = np.nan

        # UI toggle
        show_all = st.checkbox("Show all variables (not only significant)", value=False, key="show_all_divergence")
        view_df = comp_df if show_all else comp_df[comp_df["Significant"] == "Yes"]

        if view_df.empty:
            st.info("No significant divergences based on current thresholds.")
            return

        # Build display columns strictly from what's present
        display_cols = ["Variable"] + model_names + ["Min","Max","Range","Fold","Significant"]
        display_cols = [c for c in display_cols if c in view_df.columns]

        # Column config only for actually present columns
        col_cfg = {"Variable": st.column_config.TextColumn("Variable")}
        for mname in model_names:
            if mname in view_df.columns:
                col_cfg[mname] = st.column_config.NumberColumn(mname, format="%.2f")
        if "Min" in view_df.columns:   col_cfg["Min"]   = st.column_config.NumberColumn("Min", format="%.2f")
        if "Max" in view_df.columns:   col_cfg["Max"]   = st.column_config.NumberColumn("Max", format="%.2f")
        if "Range" in view_df.columns: col_cfg["Range"] = st.column_config.NumberColumn("Range", format="%.2f")
        if "Fold" in view_df.columns:  col_cfg["Fold"]  = st.column_config.NumberColumn("Fold (×)", format="%.2f")
        if "Significant" in view_df.columns:
            col_cfg["Significant"] = st.column_config.TextColumn("Flag")

        # Display
        st.dataframe(view_df[display_cols], use_container_width=True, hide_index=True, column_config=col_cfg)

        # Markdown export (your helper already guards missing cols)
        from math import isnan
        st.markdown("**Markdown**")
        st.code(df_to_markdown_table(view_df, display_cols), language="markdown")

        # Optional: small debug expander to inspect raw comp_df if needed
        with st.expander("Debug: raw divergence table", expanded=False):
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Divergence view failed. See details below.")
        st.exception(e)
