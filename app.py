from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load local environment variables without exposing secrets in the code.
load_dotenv()

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CleanGanga – Prayagraj",
    page_icon="🌊",
    layout="wide",
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
APP_DIR = DATA_DIR / "app"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"

ASSESSMENT_FILE = DATA_DIR / "prayagraj_assessment.csv"
STATION_FILE = DATA_DIR / "station_summary.csv"
HOTSPOT_FILE = DATA_DIR / "hotspot_ranking.csv"

APP_STATION_FILE = APP_DIR / "station_dashboard.csv"

# Prefer the NB06 UI-ready table when available.
# Otherwise use the original NB02 station summary + hotspot ranking.

# ============================================================
# 3. IBM GRANITE CONFIGURATION
# ============================================================

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv(
    "WATSONX_URL",
    "https://us-south.ml.cloud.ibm.com",
)
GRANITE_MODEL_ID = os.getenv(
    "GRANITE_MODEL_ID",
    "ibm/granite-3-3-8b-instruct",
)

IBM_GRANITE_READY = bool(
    WATSONX_APIKEY and WATSONX_PROJECT_ID
)

# ============================================================
# 4. LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():
    if not STATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing required file: {STATION_FILE}"
        )

    if not HOTSPOT_FILE.exists():
        raise FileNotFoundError(
            f"Missing required file: {HOTSPOT_FILE}"
        )

    station_df = pd.read_csv(STATION_FILE)
    hotspot_df = pd.read_csv(HOTSPOT_FILE)

    assessment_df = None
    if ASSESSMENT_FILE.exists():
        assessment_df = pd.read_csv(ASSESSMENT_FILE)

    return station_df, hotspot_df, assessment_df


try:
    station_df, hotspot_df, assessment_df = load_project_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# ============================================================
# 5. NORMALIZE/VALIDATE HOTSPOT DATA
# ============================================================

if "hotspot_score_baseline" in hotspot_df.columns:
    score_column = "hotspot_score_baseline"
elif "hotspot_score" in hotspot_df.columns:
    score_column = "hotspot_score"
else:
    st.error(
        "The hotspot ranking file does not contain "
        "'hotspot_score_baseline' or 'hotspot_score'."
    )
    st.stop()

if "Station" not in station_df.columns:
    st.error("station_summary.csv must contain a 'Station' column.")
    st.stop()

if "Station" not in hotspot_df.columns:
    st.error("hotspot_ranking.csv must contain a 'Station' column.")
    st.stop()

# ============================================================
# 6. RAG KNOWLEDGE INDEX
# ============================================================

@st.cache_resource
def build_rag_index():
    if not KNOWLEDGE_DIR.exists():
        return None, None, []

    files = sorted(
        list(KNOWLEDGE_DIR.glob("*.txt"))
        + list(KNOWLEDGE_DIR.glob("*.md"))
    )

    chunks = []

    def chunk_text(text, chunk_size=1200, overlap=200):
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        result = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            result.append(text[start:end])

            if end == len(text):
                break

            start = max(0, end - overlap)

        return result

    for path in files:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for chunk_id, chunk in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "source": path.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    if not chunks:
        return None, None, []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(
        [item["text"] for item in chunks]
    )

    return vectorizer, matrix, chunks


vectorizer, knowledge_matrix, knowledge_chunks = build_rag_index()


def retrieve_knowledge(query, top_k=3):
    if (
        vectorizer is None
        or knowledge_matrix is None
        or not knowledge_chunks
    ):
        return []

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        knowledge_matrix,
    )[0]

    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            **knowledge_chunks[i],
            "score": float(scores[i]),
        }
        for i in indices
    ]


# ============================================================
# 7. IBM GRANITE ADAPTER
# ============================================================

@st.cache_resource
def get_granite_model():
    if not IBM_GRANITE_READY:
        return None

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_APIKEY,
        )

        return ModelInference(
            model_id=GRANITE_MODEL_ID,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                "max_new_tokens": 350,
                "temperature": 0.2,
            },
        )

    except Exception as exc:
        st.warning(
            "IBM Granite could not be initialized. "
            f"Reason: {type(exc).__name__}"
        )
        return None


granite_model = get_granite_model()


# ============================================================
# 8. STATION EVIDENCE
# ============================================================

def get_station_evidence(station_name):
    station_match = station_df[
        station_df["Station"] == station_name
    ]

    hotspot_match = hotspot_df[
        hotspot_df["Station"] == station_name
    ]

    if station_match.empty:
        return None

    station_row = station_match.iloc[0]
    hotspot_row = (
        hotspot_match.iloc[0]
        if not hotspot_match.empty
        else None
    )

    def value(row, columns):
        if row is None:
            return None

        for column in columns:
            if column in row.index:
                item = row[column]

                if pd.isna(item):
                    return None

                return item

        return None

    return {
        "station": station_name,
        "latitude": value(
            station_row,
            ["latitude", "Latitude"],
        ),
        "longitude": value(
            station_row,
            ["longitude", "Longitude"],
        ),
        "observations": value(
            station_row,
            ["observations"],
        ),
        "persistence": value(
            station_row,
            ["persistence"],
        ),
        "mean_bod": value(
            station_row,
            ["mean_bod"],
        ),
        "max_bod": value(
            station_row,
            ["max_bod"],
        ),
        "mean_fc": value(
            station_row,
            ["mean_fc"],
        ),
        "max_fc": value(
            station_row,
            ["max_fc"],
        ),
        "anomaly_rate": value(
            station_row,
            ["anomaly_rate"],
        ),
        "hotspot_score": value(
            hotspot_row,
            [score_column],
        ),
        "hotspot_rank": value(
            hotspot_row,
            ["rank"],
        ),
    }


# ============================================================
# 9. GROUNDED PROMPT
# ============================================================

def build_grounded_prompt(
    evidence,
    user_question,
    retrieved_docs,
):
    metric_names = [
        "observations",
        "persistence",
        "mean_bod",
        "max_bod",
        "mean_fc",
        "max_fc",
        "anomaly_rate",
    ]

    metric_lines = []

    for name in metric_names:
        value = evidence.get(name)

        if value is not None:
            metric_lines.append(
                f"- {name}: {value}"
            )

    metrics = "\n".join(metric_lines)
    if not metrics:
        metrics = "- No additional station metrics available."

    source_blocks = []

    for doc in retrieved_docs:
        source_blocks.append(
            f"[Source: {doc['source']} | "
            f"Chunk: {doc['chunk_id']}]\n"
            f"{doc['text']}"
        )

    source_text = "\n\n".join(source_blocks)

    if not source_text:
        source_text = "No verified reference evidence was retrieved."

    return f"""
You are an environmental decision-support assistant
for the CleanGanga-Prayagraj project.

USER QUESTION:
{user_question}

PROJECT-COMPUTED EVIDENCE:

Station: {evidence["station"]}
Hotspot score: {evidence["hotspot_score"]}
Hotspot rank: {evidence["hotspot_rank"]}

Station metrics:
{metrics}

RETRIEVED REFERENCE EVIDENCE:

{source_text}

INSTRUCTIONS:

1. Answer only from the supplied evidence.
2. Clearly distinguish project-computed measurements from
   retrieved reference information.
3. Mention the retrieved source when using reference information.
4. Never invent measurements, dates, sources, or causes.
5. Do not claim a pollution source unless the supplied evidence
   explicitly supports it.
6. Do not claim causation.
7. Do not present the hotspot score as an official regulatory
   classification.
8. Communicate uncertainty and limitations.
9. If the evidence is insufficient, explicitly say so.
10. Keep the answer concise and useful for decision support.

Return a grounded explanation.
""".strip()


def generate_granite_response(prompt):
    if granite_model is None:
        return None

    try:
        return granite_model.generate_text(prompt=prompt)
    except Exception as exc:
        st.error(
            "IBM Granite request failed: "
            f"{type(exc).__name__}"
        )
        return None


# ============================================================
# 10. PAGE HEADER
# ============================================================

st.title("🌊 CleanGanga – Prayagraj")
st.subheader("AI-Powered Water Quality Decision Support")

st.write(
    "Explore station-level water-quality evidence, identify "
    "potential pollution hotspots, and ask grounded questions "
    "using IBM Granite + RAG."
)

st.divider()

# ============================================================
# 11. DASHBOARD
# ============================================================

station_count = hotspot_df["Station"].nunique()

observation_count = (
    len(assessment_df)
    if assessment_df is not None
    else "N/A"
)

numeric_scores = pd.to_numeric(
    hotspot_df[score_column],
    errors="coerce",
)

highest_score = numeric_scores.max()

if numeric_scores.notna().any():
    highest_station = hotspot_df.loc[
        numeric_scores.idxmax(),
        "Station",
    ]
else:
    highest_station = "N/A"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stations Monitored", station_count)

with col2:
    st.metric("Observations", observation_count)

with col3:
    st.metric(
        "Highest Hotspot Score",
        f"{highest_score:.2f}",
    )

with col4:
    st.metric(
        "Top Hotspot",
        str(highest_station)[:25],
    )

# ============================================================
# 12. HOTSPOT RANKING
# ============================================================

st.divider()
st.header("🔥 Pollution Hotspot Ranking")

ranking = hotspot_df.copy()

ranking[score_column] = pd.to_numeric(
    ranking[score_column],
    errors="coerce",
)

ranking = ranking.sort_values(
    score_column,
    ascending=False,
)

display_columns = ["Station", score_column]

if "rank" in ranking.columns:
    display_columns.insert(0, "rank")

ranking_display = ranking[display_columns].rename(
    columns={
        score_column: "Hotspot Score",
        "rank": "Rank",
    }
)

st.dataframe(
    ranking_display,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 13. STATION ANALYSIS
# ============================================================

st.divider()
st.header("📍 Station Analysis")

station_names = (
    hotspot_df["Station"]
    .dropna()
    .astype(str)
    .tolist()
)

selected_station = st.selectbox(
    "Select a monitoring station",
    station_names,
)

evidence = get_station_evidence(selected_station)

if evidence is None:
    st.warning(
        "Detailed evidence for this station is unavailable."
    )
    st.stop()

# ============================================================
# 14. EVIDENCE METRICS
# ============================================================

st.subheader("Water Quality Evidence")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if evidence["mean_bod"] is not None:
        st.metric(
            "Mean BOD",
            f"{float(evidence['mean_bod']):.2f} mg/L",
        )

with col2:
    if evidence["mean_fc"] is not None:
        st.metric(
            "Mean Fecal Coliform",
            f"{float(evidence['mean_fc']):.0f}",
        )

with col3:
    if evidence["persistence"] is not None:
        st.metric(
            "Persistence",
            f"{float(evidence['persistence']):.2f}",
        )

with col4:
    if evidence["hotspot_score"] is not None:
        st.metric(
            "Hotspot Score",
            f"{float(evidence['hotspot_score']):.2f}",
        )

# ============================================================
# 15. LOCATION
# ============================================================

if (
    evidence["latitude"] is not None
    and evidence["longitude"] is not None
):
    st.subheader("🗺️ Station Location")

    location_df = pd.DataFrame(
        {
            "latitude": [float(evidence["latitude"])],
            "longitude": [float(evidence["longitude"])],
        }
    )

    st.map(location_df)

    st.caption(
        f"Coordinates: "
        f"{float(evidence['latitude']):.6f}, "
        f"{float(evidence['longitude']):.6f}"
    )

# ============================================================
# 16. DETAILED EVIDENCE TABLE
# ============================================================

st.subheader("📊 Detailed Station Evidence")

evidence_table = pd.DataFrame(
    [
        {
            "Station": evidence["station"],
            "Observations": evidence["observations"],
            "Persistence": evidence["persistence"],
            "Mean BOD": evidence["mean_bod"],
            "Max BOD": evidence["max_bod"],
            "Mean Fecal Coliform": evidence["mean_fc"],
            "Max Fecal Coliform": evidence["max_fc"],
            "Anomaly Rate": evidence["anomaly_rate"],
            "Hotspot Score": evidence["hotspot_score"],
            "Hotspot Rank": evidence["hotspot_rank"],
        }
    ]
)

st.dataframe(
    evidence_table,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 17. AI DECISION SUPPORT
# ============================================================

st.divider()
st.header("🤖 AI Decision Support")

if IBM_GRANITE_READY:
    st.success(
        f"IBM Granite configured: {GRANITE_MODEL_ID}"
    )
else:
    st.info(
        "IBM Granite credentials are not configured yet. "
        "The dashboard and retrieval layer can still be inspected."
    )

question = st.text_area(
    "Ask a question about this station",
    placeholder=(
        "Example: Why is this station considered a "
        "potential pollution hotspot?"
    ),
    height=100,
)

if st.button(
    "Generate Decision Support",
    type="primary",
):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        retrieval_query = (
            f"{question} "
            f"{selected_station} "
            "water quality BOD fecal coliform "
            "hotspot monitoring"
        )

        retrieved_docs = retrieve_knowledge(
            retrieval_query,
            top_k=3,
        )

        prompt = build_grounded_prompt(
            evidence,
            question,
            retrieved_docs,
        )

        answer = generate_granite_response(prompt)

        if answer:
            st.subheader("💡 IBM Granite Response")
            st.write(answer)

        elif not IBM_GRANITE_READY:
            st.warning(
                "Granite is not configured. "
                "The retrieved evidence is shown below."
            )

        if retrieved_docs:
            st.subheader("📚 Retrieved Evidence")

            for doc in retrieved_docs:
                with st.expander(
                    f"{doc['source']} | "
                    f"chunk {doc['chunk_id']} | "
                    f"score {doc['score']:.3f}"
                ):
                    st.write(doc["text"])
        else:
            st.warning(
                "No knowledge-base documents were retrieved. "
                "Add verified .txt or .md references to "
                "data/knowledge_base/."
            )

# ============================================================
# 18. RESPONSIBLE AI
# ============================================================

st.divider()

with st.expander("⚠️ Responsible AI & Project Limitations"):
    st.markdown(
        """
        - Measurements are taken from the project dataset.
        - Hotspot scores come from the deterministic hotspot analysis.
        - The hotspot score is a decision-support indicator, not an
          official regulatory determination.
        - The Logistic Regression target is rule-derived and should not
          be treated as independent scientific ground truth.
        - The system does not establish causation or identify responsible
          parties.
        - AI explanations must remain grounded in supplied evidence.
        - Retrieved sources are displayed for transparency.
        - Important decisions should be supported by appropriate
          environmental/field investigation.
        - API keys must never be committed to Git.
        """
    )

st.caption(
    "CleanGanga-Prayagraj | IBM Granite + RAG | Streamlit Prototype"
)
