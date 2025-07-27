import streamlit as st
import pandas as pd

st.set_page_config(page_title="Stock Ranking Questionnaire", layout="wide")

# Kriterien mit Frage, Skalenbeschreibung und Gewichtung
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
        "weight": 0.2,
    },
    {
        "name": "ProjVol",
        "question": "Wie hoch ist das projizierte Tagesvolumen (Projected Volume)?",
        "options": [
            "1 – Sehr gering (<2 Mio.)",
            "2 – Gering (2–5 Mio.)",
            "3 – Mittel (5–10 Mio.)",
            "4 – Hoch (10–20 Mio.)",
            "5 – Extrem hoch (>20 Mio.)",
        ],
        "weight": 0.2,
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
        "weight": 0.1,
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
        "weight": 0.1,
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
        "weight": 0.1,
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
        "weight": 0.1,
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
        "weight": 0.1,
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
        "weight": 0.05,
    },
]
NEWS_BONUS = 1

# Memory für bereits bewertete Stocks
if "stock_scores" not in st.session_state:
    st.session_state.stock_scores = []

with st.form(key="stock_form", clear_on_submit=True):
    ticker = st.text_input("Stock-Ticker", max_chars=10).strip().upper()
    criteria_points = {}
    for crit in CRITERIA:
        idx = st.radio(crit["question"], options=list(enumerate(crit["options"], 1)),
                       format_func=lambda x: x[1], key=crit["name"])
        criteria_points[crit["name"]] = idx[0]  # Die Punkte (1–5)
    news = st.checkbox("Gibt es relevante News/Katalysator?", key="news")
    submit = st.form_submit_button("Stock bewerten & speichern")

if submit and ticker:
    weighted_score = sum(
        criteria_points[crit['name']] * crit['weight'] for crit in CRITERIA
    ) + (NEWS_BONUS if news else 0)
    stock_entry = {
        "Ticker": ticker,
        **criteria_points,
        "News": "Ja" if news else "Nein",
        "Score": round(weighted_score, 2)
    }
    st.session_state.stock_scores.append(stock_entry)
    st.success(f"Stock {ticker} gespeichert!")

st.write("---")
st.header("Bisherige Bewertungen & Ranking")
if st.session_state.stock_scores:
    df = pd.DataFrame(st.session_state.stock_scores)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
    # Optional: Download-Link
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "Tabelle als CSV herunterladen",
        data=csv,
        file_name="stock_ranking.csv",
        mime="text/csv"
    )
else:
    st.info("Noch keine Stocks bewertet. Fülle den Fragebogen oben aus!")
