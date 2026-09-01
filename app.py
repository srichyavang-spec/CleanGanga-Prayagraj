from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CleanGanga – Prayagraj",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
APP_DATA_DIR = DATA_DIR / "app"

STATION_FILE = DATA_DIR / "station_summary.csv"
HOTSPOT_FILE = DATA_DIR / "hotspot_ranking.csv"
ASSESSMENT_FILE = DATA_DIR / "prayagraj_assessment.csv"
PROTOTYPE_FILE = DATA_DIR / "prototype_response.json"


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    station_df = pd.read_csv(STATION_FILE)
    hotspot_df = pd.read_csv(HOTSPOT_FILE)

    assessment_df = None

    if ASSESSMENT_FILE.exists():
        assessment_df = pd.read_csv(ASSESSMENT_FILE)

    return station_df, hotspot_df, assessment_df


@st.cache_data
def load_prototype_response():

    if not PROTOTYPE_FILE.exists():
        return None

    try:
        with open(PROTOTYPE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return None


# ============================================================
# 4. CHECK FILES
# ============================================================

if not STATION_FILE.exists() or not HOTSPOT_FILE.exists():

    st.error(
        "Required project data files are missing. "
        "Please make sure the data folder contains "
        "station_summary.csv and hotspot_ranking.csv."
    )

    st.stop()


station_df, hotspot_df, assessment_df = load_data()
prototype_response = load_prototype_response()


# ============================================================
# 5. IDENTIFY HOTSPOT SCORE COLUMN
# ============================================================

if "hotspot_score_baseline" in hotspot_df.columns:

    score_column = "hotspot_score_baseline"

elif "hotspot_score" in hotspot_df.columns:

    score_column = "hotspot_score"

else:

    st.error("Hotspot score column was not found.")
    st.stop()


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.title("🌊 CleanGanga – Prayagraj")

st.subheader(
    "AI-Powered Water Quality Decision Support"
)

st.write(
    "Analyze water-quality evidence, identify potential pollution "
    "hotspots, and generate grounded decision-support insights."
)

st.divider()


# ============================================================
# 7. DASHBOARD METRICS
# ============================================================

station_count = hotspot_df["Station"].nunique()

if assessment_df is not None:
    observation_count = len(assessment_df)
else:
    observation_count = "N/A"

highest_score = pd.to_numeric(
    hotspot_df[score_column],
    errors="coerce"
).max()

highest_station = hotspot_df.loc[
    hotspot_df[score_column].idxmax(),
    "Station"
]


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Stations Monitored",
        station_count
    )

with col2:
    st.metric(
        "Observations",
        observation_count
    )

with col3:
    st.metric(
        "Highest Hotspot Score",
        f"{highest_score:.2f}"
    )

with col4:
    st.metric(
        "Top Hotspot",
        str(highest_station)[:25]
    )


st.divider()


# ============================================================
# 8. HOTSPOT RANKING
# ============================================================

st.header("🔥 Pollution Hotspot Ranking")

ranking_display = hotspot_df.copy()

ranking_display = ranking_display.sort_values(
    score_column,
    ascending=False
)

display_columns = ["Station", score_column]

if "rank" in ranking_display.columns:
    display_columns.insert(0, "rank")

ranking_display = ranking_display[display_columns]

ranking_display = ranking_display.rename(
    columns={
        score_column: "Hotspot Score"
    }
)

st.dataframe(
    ranking_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 9. STATION SELECTION
# ============================================================

st.divider()

st.header("📍 Station Analysis")

station_names = hotspot_df["Station"].dropna().tolist()

selected_station = st.selectbox(
    "Select a monitoring station",
    station_names
)


# ============================================================
# 10. FIND SELECTED STATION
# ============================================================

station_match = station_df[
    station_df["Station"] == selected_station
]

hotspot_match = hotspot_df[
    hotspot_df["Station"] == selected_station
]

if station_match.empty:

    st.warning(
        "Detailed information for this station is not available."
    )

else:

    station = station_match.iloc[0]
    hotspot = hotspot_match.iloc[0]


    # ========================================================
    # 11. STATION METRICS
    # ========================================================

    st.subheader("Water Quality Evidence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if "mean_bod" in station.index:

            st.metric(
                "Mean BOD",
                f"{station['mean_bod']:.2f} mg/L"
            )

    with col2:

        if "mean_fc" in station.index:

            st.metric(
                "Mean Fecal Coliform",
                f"{station['mean_fc']:.0f}"
            )

    with col3:

        if "persistence" in station.index:

            st.metric(
                "Persistence",
                f"{station['persistence']:.2f}"
            )

    with col4:

        st.metric(
            "Hotspot Score",
            f"{hotspot[score_column]:.2f}"
        )


    # ========================================================
    # 12. LOCATION
    # ========================================================

    st.subheader("📍 Location")

    latitude = station.get("latitude")
    longitude = station.get("longitude")

    if pd.notna(latitude) and pd.notna(longitude):

        location_df = pd.DataFrame(
            {
                "latitude": [latitude],
                "longitude": [longitude]
            }
        )

        st.map(
            location_df,
            latitude="latitude",
            longitude="longitude"
        )

        st.write(
            f"**Coordinates:** {latitude:.6f}, {longitude:.6f}"
        )


    # ========================================================
    # 13. ADDITIONAL EVIDENCE
    # ========================================================

    st.subheader("📊 Station Evidence")

    evidence_columns = [
        "Station",
        "observations",
        "persistence",
        "mean_bod",
        "mean_fc",
        "anomaly_rate"
    ]

    available_columns = [
        column
        for column in evidence_columns
        if column in station.index
    ]

    evidence_df = pd.DataFrame(
        [station[available_columns]]
    )

    st.dataframe(
        evidence_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 14. AI DECISION SUPPORT
# ============================================================

st.divider()

st.header("🤖 AI Decision Support")

st.write(
    "Ask a question about the selected station. "
    "The final version will connect this interface to "
    "IBM Granite + RAG."
)

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: Why is this station considered "
        "a potential pollution hotspot?"
    ),
    height=100
)


# ============================================================
# 15. CURRENT PROTOTYPE RESPONSE
# ============================================================

if st.button("Generate Decision Support", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        st.subheader("💡 Decision-Support Response")

        # ----------------------------------------------------
        # Try existing prototype response
        # ----------------------------------------------------

        response_text = None

        if isinstance(prototype_response, dict):

            possible_keys = [
                "response",
                "answer",
                "generated_response",
                "explanation"
            ]

            for key in possible_keys:

                if key in prototype_response:

                    response_text = prototype_response[key]
                    break


        # ----------------------------------------------------
        # Fallback response using verified project evidence
        # ----------------------------------------------------

        if not response_text:

            selected_score = float(
                hotspot[score_column]
            )

            mean_bod = station.get("mean_bod")
            mean_fc = station.get("mean_fc")
            persistence = station.get("persistence")

            response_text = (
                f"{selected_station} has a hotspot score of "
                f"{selected_score:.2f} in the project's baseline "
                f"hotspot analysis. "
            )

            if pd.notna(mean_bod):

                response_text += (
                    f"The recorded mean BOD is approximately "
                    f"{mean_bod:.2f} mg/L. "
                )

            if pd.notna(mean_fc):

                response_text += (
                    f"The recorded mean fecal coliform level is "
                    f"approximately {mean_fc:.0f} MPN/100mL. "
                )

            if pd.notna(persistence):

                response_text += (
                    f"The station's persistence value is "
                    f"{persistence:.2f}. "
                )

            response_text += (
                "These values provide evidence for prioritizing "
                "this station for further investigation. "
                "They should not be interpreted as an official "
                "regulatory determination."
            )


        st.write(response_text)


# ============================================================
# 16. RESPONSIBLE AI NOTICE
# ============================================================

st.divider()

with st.expander("⚠️ Responsible AI & Limitations"):

    st.write(
        """
        **Important limitations**

        - The analysis is based on the project's available dataset.
        - Hotspot scores are decision-support indicators, not official
          regulatory classifications.
        - AI-generated explanations should remain grounded in verified
          project evidence.
        - The system should not invent measurements or unsupported causes.
        - Additional field investigation may be required before taking
          regulatory or operational action.
        """
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "CleanGanga-Prayagraj | AI-assisted water-quality "
    "decision support | Prototype"
)