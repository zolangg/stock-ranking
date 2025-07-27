uimport streamlit as st
import pandas as pd

st.header("Premarket Stock Ranking")

# 1:1 Catalyst-Liste aus dem PDF – KEINE Eigenkreationen
CATALYSTS = [
    # NEWS CATALYSTS
    {"name": "Unusually good/bad earnings report (surprise!)", "score": 1.0},
    {"name": "Better/worse than expected guidance reported", "score": 0.9},
    {"name": "Market can’t put a ceiling on earnings (growth stocks)", "score": 0.9},
    {"name": "New/Delayed product announced", "score": 0.8},
    {"name": "Announced positive/negative results of an ongoing study", "score": 0.8},
    {"name": "Announced positive/negative results of an independent study", "score": 0.9},
    {"name": "Announced positive/negative results of a completed study", "score": 0.9},
    {"name": "Move into a hot new sector", "score": 0.7},
    {"name": "Gained market share", "score": 0.7},
    {"name": "Collaboration/Dissociation with an established company", "score": 0.7},
    {"name": "Announced offering/reverse-split/supply with dilution impact", "score": -0.8},
    {"name": "Favorable/unfavorable government regulatory announcement", "score": 1.0},
    {"name": "New/cancellation of a large contract for the company", "score": 0.8},
    {"name": "Getting new funding", "score": 0.5},
    {"name": "Cost cuts", "score": 0.4},
    {"name": "Improved net margin", "score": 0.4},
    {"name": "Analyst upgrades/downgrades", "score": 0.5},
    {"name": "Macroeconomic news", "score": 0.6},
    {"name": "Management changes", "score": 0.3},
    {"name": "Dividend announcements", "score": 0.2},
    {"name": "Report alleging misconduct by the company", "score": -0.5},
    {"name": "Showcased at a prestigious event to potential investors, partners, and industry colleagues", "score": 0.3},
    {"name": "Honored for excellence in performance or product innovation", "score": 0.2},
    {"name": "Paid-off debt", "score": 0.3},
    {"name": "Anchor/Sympathy Play", "score": 0.2},

    # TECHNICAL CATALYSTS
    {"name": "Anchored/2-Day VWAP", "score": 0.4},
    {"name": "Moving Average", "score": 0.3},
    {"name": "Candlestick Pattern", "score": 0.3},
    {"name": "Chart Pattern", "score": 0.4},
    {"name": "Trendline", "score": 0.3},
    {"name": "Volume", "score": 0.5},
    {"name": "Moving higher with unusually strong Options Volume", "score": 0.5},
    {"name": "Break-out/-down Angle", "score": 0.4},
    {"name": "Support/Resistance Levels", "score": 0.3},

    # PRICE CATALYSTS
    {"name": "Overnight gap of ±3% and min. 10% of Average Daily Volume", "score": 0.7},
    {"name": "Held bid in an up-trending stock visible on Level 2 and Time, Sales", "score": 0.6},
    {"name": "Gap fill", "score": 0.3},
    {"name": "Recent History (2Y)", "score": 0.2},
    {"name": "Full/Half/Quarter Price Level", "score": 0.2},
    {"name": "Overextension", "score": 0.3},
    {"name": "Beaten Down Stock", "score": 0.3},
    {"name": "All-time/52W High/Low Break", "score": 0.7},
    {"name": "Breakout", "score": 0.7},
]

# Hauptkriterien (wie vorhin, ggf. Gewichte leicht angepasst)
CRITERIA = [
    {
        "name": "RVOL",
        "question": "Wie hoch ist das Relative Volume (RVOL)?",
        "options": [
            "1 – Unterdurchschnittlich (<1)",
            "2 – Niedrig (1–2)",
            "3 – Durchschnittlich (2–5)",
            "4 – Hoch (5–10)",
            "5 – Extrem hoch (>10)",
        ],
        "weight": 0.17,
    },
    {
        "name": "ATR",
        "question": "Wie groß ist die ATR ($)?",
        "options": [
            "1 – Sehr eng (<0.1 $)",
            "2 – Eng (0.1–0.2 $)",
            "3 – Mittel (0.2–0.5 $)",
            "4 – Hoch (0.5–1.0 $)",
            "5 – Sehr groß (>1.0 $)",
        ],
        "weight": 0.08,
    },
    {
        "name": "Float",
        "question": "Wie niedrig ist der Float?",
        "options": [
            "1 – Sehr hoch (>100 Mio.)",
            "2 – Hoch (50–100 Mio.)",
            "3 – Mittel (25–50 Mio.)",
            "4 – Niedrig (10–25 Mio.)",
            "5 – Sehr niedrig (<10 Mio.)",
        ],
        "weight": 0.09,
    },
    {
        "name": "PreMarket",
        "question": "Wie ist die Pre-Market Struktur?",
        "options": [
            "1 – Flatline, keine Bewegung",
            "2 – Chaotisch, keine klaren Levels",
            "3 – Gap und Volumen, keine echte Struktur",
            "4 – Guter Move, aber leichte Unsauberkeiten",
            "5 – Klarer, starker Move, klare Trigger",
        ],
        "weight": 0.09,
    },
    {
        "name": "Technicals",
        "question": "Wie sind die Technicals (Pattern, Overhead, Struktur)?",
        "options": [
            "1 – Kein klares Setup, viel Overhead",
            "2 – Viele Widerstände, durchwachsen",
            "3 – Leichte Overhead-Levels, mittelmäßig",
            "4 – Klares Muster, wenig Overhead",
            "5 – Kein Overhead, Gap-Up, Blue-Sky",
        ],
        "weight": 0.09,
    },
    {
        "name": "Monthly",
        "question": "Wie sieht der Monthly/Weekly-Kontext aus?",
        "options": [
            "1 – Kein Kontext, Flat/Random",
            "2 – Alter Downtrend, kein Volumen",
            "3 – Downtrend mit Volumen, viele alte Levels",
            "4 – Großer Volumenanstieg, leichte alte Widerstände",
            "5 – Dead-Chart, frischer Volumenpeak, Breakout",
        ],
        "weight": 0.06,
    },
    {
        "name": "VolProfile",
        "question": "Wie ist das Volume Profile?",
        "options": [
            "1 – Flat, keine Struktur",
            "2 – Chaotisch, viele Cluster",
            "3 – Viele Cluster, einige Level als Widerstand",
            "4 – Gute Cluster, wenig Overhead",
            "5 – Klares Volumencluster, keine Overhead-Resistance",
        ],
        "weight": 0.06,
    },
    {
        "name": "Spread",
        "question": "Wie eng ist der Bid-Ask-Spread?",
        "options": [
            "1 – Sehr groß (>3%)",
            "2 – Groß (2–3%)",
            "3 – Mittel (1–2%)",
            "4 – Eng (0.5–1%)",
            "5 – Extrem eng (<0.5%)"
        ],
        "weight": 0.08,
    },
    {
        "name": "FloatPct",
        "question": "Wie viel Prozent des Floats wurden Premarket bereits gehandelt?",
        "options": [
            "1 – <2%",
            "2 – 2–5%",
            "3 – 5–10%",
            "4 – 10–20%",
            "5 – >20%"
        ],
        "weight": 0.08,
    },
]

if "stock_scores" not in st.session_state:
    st.session_state.stock_scores = []

with st.form(key="stock_form", clear_on_submit=True):
    ticker = st.text_input("Stock-Ticker", max_chars=10).strip().upper()
    criteria_points = {}
    for crit in CRITERIA:
        idx = st.radio(crit["question"], options=list(enumerate(crit["options"], 1)),
                       format_func=lambda x: x[1], key=crit["name"])
        criteria_points[crit["name"]] = idx[0]
    
    # --- Catalyst Bewertung als Multiselect ---
    selected_catalysts = st.multiselect(
        "Wähle alle News/Technicals/Price-Katalysatoren (Mehrfachauswahl möglich):",
        options=[cat["name"] for cat in CATALYSTS]
    )
    # Score addieren (Deckelung: maximal 1.0)
    catalyst_score = sum(cat["score"] for cat in CATALYSTS if cat["name"] in selected_catalysts)
    catalyst_score = min(max(catalyst_score, 0), 1.0)
    catalyst_points = catalyst_score * 5   # auf 1–5 Skala
    
    criteria_points["Catalyst"] = catalyst_points
    submit = st.form_submit_button("Stock bewerten & speichern")

if submit and ticker:
    weighted_score = (
        sum(
            criteria_points[crit['name']] * crit['weight'] for crit in CRITERIA
        ) +
        catalyst_points * 0.20  # Catalyst als Extra-Gewichtung
    ) / (1 + 0.20) # normieren, damit max Score ≈ 5 bleibt
    stock_entry = {
        "Ticker": ticker,
        **criteria_points,
        "Catalyst_Types": ', '.join(selected_catalysts) if selected_catalysts else "None",
        "Score": round(weighted_score, 2)
    }
    st.session_state.stock_scores.append(stock_entry)
    st.success(f"Stock {ticker} gespeichert!")

st.write("---")
st.header("Ranking")

def heat_level(score):
    if score >= 4.35:
        return "A+"
    elif score >= 4.0:
        return "A"
    elif score >= 3.7:
        return "B"
    elif score >= 3.3:
        return "C"
    else:
        return "D"

def color_level(val):
    color_map = {
        "A+": "background-color: #fa7268; color: white",
        "A":  "background-color: #ffe156; color: black",
        "B":  "background-color: #6ee7b7; color: black",
        "C":  "background-color: #60a5fa; color: white",
        "D":  "background-color: #e5e7eb; color: black"
    }
    return color_map.get(val, "")

if st.session_state.stock_scores:
    df = pd.DataFrame(st.session_state.stock_scores)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Level"] = df["Score"].apply(heat_level)
    df["Score"] = df["Score"].astype(float).round(2)

    # Für die Anzeige: Gestylte Tabelle
    styled = df.style.format({"Score": "{:.2f}"}).applymap(color_level, subset=["Level"])
    st.dataframe(styled, use_container_width=True)

    # Für den Download: Unstylierten DataFrame nehmen!
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Tabelle als CSV herunterladen",
        data=csv,
        file_name="stock_ranking.csv",
        mime="text/csv"
    )
else:
    st.info("Noch keine Stocks bewertet. Fülle den Fragebogen oben aus!")