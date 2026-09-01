# 🌊 CleanGanga – Prayagraj
### AI-Assisted Water Pollution Hotspot Decision-Support System

> An explainable, evidence-driven prototype for identifying potential water-quality hotspots in Prayagraj and generating grounded decision-support explanations using AI.

---

## 📌 Overview

CleanGanga-Prayagraj is an AI + Sustainability project developed as part of the **1M1B AI for Sustainability Virtual Internship**, in collaboration with **IBM SkillsBuild & AICTE**.

The project focuses on a practical environmental problem:

> **How can historical water-quality data be transformed into an interpretable system that helps identify potential pollution hotspots and communicate the evidence behind those findings?**

Instead of directly asking a Large Language Model (LLM) to determine whether a location is polluted, this project follows a more reliable approach:

```text
Raw Water-Quality Data
        ↓
Data Quality & Cleaning
        ↓
Prayagraj Filtering
        ↓
Water-Quality Assessment
        ↓
Station-Level Analysis
        ↓
Hotspot Detection
        ↓
Hotspot Ranking
        ↓
Evidence Generation
        ↓
Retrieval-Grounded AI
        ↓
Decision-Support Explanation
        ↓
Interactive Streamlit Prototype
```

The central design principle is:

> **AI should explain evidence, not replace evidence.**

---

# 🎯 Project Objectives

The project was developed with the following objectives:

- Analyze historical water-quality observations for Prayagraj.
- Identify monitoring stations with potentially concerning pollution patterns.
- Develop an interpretable hotspot scoring methodology.
- Analyze persistence and unusual observations.
- Establish a classical machine-learning baseline.
- Build an evidence layer for AI-assisted explanations.
- Explore Retrieval-Augmented Generation (RAG) with IBM Granite.
- Evaluate AI responses and failure cases.
- Incorporate responsible-AI safeguards.
- Build an interactive Streamlit prototype.
- Document the complete development process and limitations.

---

# 🌍 Sustainability & SDG Alignment

## Primary SDG: SDG 6 – Clean Water and Sanitation

The project directly supports **SDG 6**, which focuses on ensuring availability and sustainable management of water and sanitation for all.

CleanGanga-Prayagraj contributes to this goal by exploring how data analysis and AI can assist with:

- water-quality monitoring,
- identification of potential pollution hotspots,
- evidence-based environmental analysis,
- faster interpretation of monitoring data,
- communication of environmental insights.

### Potential stakeholders

The system could be useful as a prototype for:

- environmental analysts,
- water-quality monitoring teams,
- researchers,
- sustainability teams,
- policy and planning teams,
- students and educators.

The prototype is **not intended to replace official environmental monitoring or regulatory decision-making**.

---

# 🧩 Problem Statement

Water-quality monitoring produces large amounts of observations across different locations and time periods.

However, raw monitoring data does not automatically provide an easy way to answer questions such as:

- Which stations appear most concerning?
- Are concerning measurements persistent?
- Which pollution indicators contribute to the concern?
- Is an observation an isolated anomaly or part of a broader pattern?
- What evidence supports a station's ranking?
- How can this evidence be communicated clearly to a human decision-maker?

A system that combines transparent data analysis with evidence-grounded AI can help convert raw observations into more understandable decision-support information.

---

# 💡 Proposed Solution

CleanGanga-Prayagraj uses a layered approach.

### Layer 1 — Data

Historical water-quality observations are cleaned and prepared.

### Layer 2 — Analytical reasoning

Water-quality measurements are analyzed using transparent rules and station-level aggregation.

### Layer 3 — Hotspot scoring

Stations are compared using multiple signals such as:

- threshold exceedance,
- relative station performance,
- persistence,
- severity,
- anomaly behavior.

### Layer 4 — Evidence

The numerical results are preserved as structured evidence.

### Layer 5 — AI

The AI layer uses the computed evidence and retrieved knowledge to generate human-readable explanations.

### Layer 6 — Application

A Streamlit interface presents the results interactively.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │   Historical Dataset    │
                    │     CPCB Water Data     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Data Quality Layer     │
                    │ Cleaning + Validation    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Prayagraj Assessment  │
                    │ BOD + Faecal Coliform   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Analytical Layer      │
                    │ Station-level analysis  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Hotspot Engine       │
                    │ Persistence + Severity  │
                    │ Relative comparison     │
                    │ Anomaly analysis         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Evidence Layer      │
                    │ Structured station data │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Knowledge Base   │      │ Station Evidence │
          │ Standards/docs   │      │ Metrics/results  │
          └────────┬─────────┘      └────────┬─────────┘
                   │                         │
                   └────────────┬────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ AI / RAG Decision Layer │
                    │    IBM Granite          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Streamlit Prototype   │
                    │ Dashboard + Explanation │
                    └─────────────────────────┘
```

---

# 📊 Dataset

The project uses historical river water-quality monitoring data from the **Central Pollution Control Board (CPCB)** dataset for Uttar Pradesh.

### Dataset summary

| Property | Value |
|---|---:|
| Full dataset observations | 1,414 |
| Full dataset columns | 23 |
| Prayagraj/Allahabad observations | 116 |
| Monitoring stations | 6 |
| Primary analyzed indicators | BOD, Faecal Coliform |

The Prayagraj subset contains six monitoring stations.

The project identified missing measurements during the data-quality stage and handled them explicitly rather than silently treating missing values as valid observations.

---

# 🧪 Water-Quality Indicators

## Biological Oxygen Demand (BOD)

BOD is used as an indicator associated with organic pollution.

Higher BOD measurements can indicate increased oxygen demand from biological processes in the water.

## Faecal Coliform

Faecal Coliform is used as an indicator associated with contamination from faecal sources.

The availability of these indicators was not uniform across the dataset, which became an important consideration during preprocessing and analysis.

---

# 🔍 Notebook 01 — Data Quality & Preparation

File:

```text
notebooks/01_data_quality.ipynb
```

The first notebook established the foundation for the rest of the project.

Work included:

- inspecting the original dataset,
- understanding the columns,
- checking missing values,
- identifying usable measurements,
- filtering the Prayagraj/Allahabad region,
- identifying monitoring stations,
- examining measurement availability,
- applying the selected assessment logic,
- creating the cleaned assessment dataset.

### Main output

```text
data/prayagraj_assessment.csv
```

This became the primary handoff artifact between the data-preparation and analytical stages.

---

# 📈 Notebook 02 — Hotspot Analysis

File:

```text
notebooks/02_hotspot_analysis.ipynb
```

The second stage moved from cleaned observations to station-level analysis.

The notebook examined:

- station-level water-quality behavior,
- BOD measurements,
- Faecal Coliform measurements,
- threshold-based evidence,
- persistence,
- severity,
- anomaly behavior,
- station comparisons,
- hotspot ranking.

The project deliberately avoided immediately jumping to a complex AI model.

Instead, the analytical baseline was built first.

> **Baseline first → measure → inspect limitations → add AI.**

This makes the final AI layer easier to interpret and evaluate.

---

# 🔥 Hotspot Detection

A major design decision was to avoid defining a hotspot from a single measurement.

The project combines multiple signals.

## 1. Absolute exceedance

Checks whether measurements exceed the selected benchmark.

```text
Is the measurement concerning in absolute terms?
```

## 2. Relative station comparison

Compares stations against one another.

```text
Is this station performing worse relative to
the other monitored stations?
```

## 3. Persistence

Measures how consistently concerning behavior occurs.

```text
Is this a recurring pattern rather than
a single observation?
```

## 4. Severity

Captures the magnitude of concerning measurements.

## 5. Anomaly behavior

Helps distinguish unusual observations from persistent behavior.

Therefore:

```text
Absolute Evidence
       +
Relative Evidence
       +
Persistence
       +
Severity / Anomaly Information
       ↓
Hotspot Evidence
       ↓
Station Ranking
```

---

# 📁 Analytical Outputs

The analysis produces reusable artifacts instead of keeping all results only in notebook memory.

Important outputs include:

```text
data/prayagraj_assessment.csv
data/station_summary.csv
data/hotspot_ranking.csv
```

### `prayagraj_assessment.csv`

Contains cleaned Prayagraj observations and assessment-related fields.

### `station_summary.csv`

Contains station-level analytical metrics.

### `hotspot_ranking.csv`

Contains the ranked hotspot results used by later stages of the project.

This separation makes the pipeline easier to reproduce and allows downstream components to consume structured results.

---

# 🤖 Machine-Learning Baseline

A Logistic Regression baseline was also explored.

### Target

```text
Pollution_Level
```

### Features

The model deliberately used:

- Latitude
- Longitude
- Month
- Observation order

Current BOD and Faecal Coliform values were excluded from the feature set to avoid direct target leakage.

### Model configuration

```text
StandardScaler
+
LogisticRegression
```

The baseline achieved:

```text
Accuracy = 1.000
```

on a very small six-sample test set.

### Important interpretation

This result **must not be interpreted as proof of strong generalization**.

The usable dataset for this experiment was extremely small, and latitude/longitude are strongly associated with station identity.

Therefore, the unusually high test accuracy is more informative as an engineering observation than as evidence of a production-ready predictive model.

This is an intentional part of the project:

> A high metric is not automatically a good model.

The experiment helped demonstrate why dataset size, feature structure, spatial dependence, and evaluation design matter.

---

# 🧠 Notebook 03 — Granite RAG Decision Support

File:

```text
notebooks/03_granite_rag_decision_support.ipynb
```

The third stage introduced the AI decision-support layer.

The important architectural decision was:

> **IBM Granite should receive computed evidence rather than independently determining the hotspot.**

For example, the AI can receive structured information such as:

```text
Station
Hotspot score
Persistence
Mean BOD
Mean Faecal Coliform
Anomaly rate
Supporting observations
Retrieved environmental information
```

The AI then converts this evidence into an understandable explanation.

Conceptually:

```text
Structured Station Evidence
          +
Retrieved Knowledge
          ↓
     IBM Granite
          ↓
Grounded Explanation
```

---

# 📚 Retrieval-Augmented Generation

The project uses the RAG concept to ground AI responses in relevant information.

The intended knowledge base includes sources such as:

- water-quality standards,
- river-water classification references,
- relevant environmental guidance,
- location-specific information,
- project methodology documentation.

The purpose of retrieval is to provide relevant context before generation.

Instead of:

```text
Question → LLM → Answer
```

the intended architecture is:

```text
Question
   ↓
Retrieve Relevant Evidence
   ↓
Combine With Computed Station Metrics
   ↓
IBM Granite
   ↓
Grounded Explanation
```

---

# 🛡️ Responsible AI

Responsible AI is a core part of the project.

The internship guidelines explicitly require attention to fairness, transparency, ethics, and privacy.

The project therefore considers:

## Transparency

The AI is given the numerical evidence it is expected to interpret.

## Grounding

Responses should be based on supplied measurements and retrieved knowledge.

## No unsupported causation

The system should not claim:

> "This factory caused the pollution."

unless appropriate evidence exists.

A hotspot ranking does not establish causality.

## Uncertainty

The hotspot score is a **decision-support indicator**, not an official regulatory determination.

## Human oversight

The prototype supports human analysis rather than replacing environmental experts.

## Privacy

The analyzed environmental dataset does not require personal user information for the core analytical workflow.

---

# 🧪 Notebook 04 — RAG Evaluation & Responsible AI

File:

```text
notebooks/04_rag_evaluation_responsible_ai.ipynb
```

The fourth stage focused on evaluating the AI component rather than assuming that generated responses were correct.

Evaluation artifacts are maintained in:

```text
evaluation/
```

These include:

```text
evaluation_questions.csv
failure_log.csv
granite_results.csv
ground_truth.csv
manual_scores.csv
```

The evaluation considers questions such as:

- Is the response grounded?
- Does it correctly interpret the supplied evidence?
- Does it answer the actual question?
- Does it introduce unsupported claims?
- Does it follow responsible-AI constraints?
- What happens when information is unavailable?

This turns the AI component from a simple demo into something that can be inspected and evaluated.

---

# 🧩 Notebook 05 — Final Decision-Support Prototype

File:

```text
notebooks/05_final_decision_support_prototype.ipynb
```

This stage combined the analytical and AI components into a more complete decision-support workflow.

The prototype works with structured project outputs rather than requiring the entire analytical pipeline to be recomputed for every AI interaction.

The workflow is:

```text
Station Selection
      ↓
Retrieve Station Evidence
      ↓
Display Analytical Metrics
      ↓
Retrieve Supporting Context
      ↓
Generate Explanation
      ↓
Present Evidence + Explanation
```

---

# 🔗 Notebook 06 — Application Integration

The final notebook stage focused on connecting the developed components into the application workflow.

The goal was to move from:

```text
Individual notebooks
```

to:

```text
Reusable analytical outputs
        +
AI decision-support logic
        +
Application interface
```

This allowed the project to be demonstrated as an end-to-end system rather than as disconnected experiments.

---

# 🖥️ Streamlit Prototype

The project includes a Streamlit-based application.

The interface is designed to demonstrate:

- station selection,
- hotspot ranking,
- station-level metrics,
- analytical evidence,
- AI-assisted explanation,
- decision-support information.

The application provides a practical demonstration of how the analytical pipeline could be exposed to a user.

### Prototype flow

```text
Open Application
       ↓
View Hotspot Ranking
       ↓
Select Station
       ↓
Inspect Evidence
       ↓
Request Explanation
       ↓
Review AI Response
       ↓
Review Limitations / Disclaimer
```

Screenshots of the working prototype are included as part of the project demonstration.

---

# 🧰 Technology Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Logistic Regression
- StandardScaler

### Generative AI

- IBM Granite
- Retrieval-Augmented Generation
- Prompt engineering

### Application

- Streamlit

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 📂 Repository Structure

```text
CleanGanga-Prayagraj/
│
├── data/
│   ├── Prayagraj_Data.csv
│   ├── prayagraj_assessment.csv
│   ├── station_summary.csv
│   ├── hotspot_ranking.csv
│   └── prototype_response.json
│
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_hotspot_analysis.ipynb
│   ├── 03_granite_rag_decision_support.ipynb
│   ├── 04_rag_evaluation_responsible_ai.ipynb
│   ├── 05_final_decision_support_prototype.ipynb
│   └── 06_application_integration.ipynb
│
├── evaluation/
│   ├── evaluation_questions.csv
│   ├── failure_log.csv
│   ├── granite_results.csv
│   ├── ground_truth.csv
│   └── manual_scores.csv
│
├── src/
│   └── ...
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔐 Security & Secrets

API credentials are **not stored in the repository**.

Environment variables are used for sensitive credentials.

Example:

```text
.env
```

is excluded through `.gitignore`.

A real deployment should use a secure secrets-management mechanism rather than hard-coding credentials in Python files.

---

# ⚙️ Running the Project

## 1. Clone the repository

```bash
git clone <repository-url>
cd CleanGanga-Prayagraj
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a local `.env` file with the required API credentials.

**Do not commit this file.**

## 5. Run the Streamlit application

```powershell
streamlit run streamlit_app.py
```

---

# 📏 Evaluation Philosophy

The project follows a simple engineering philosophy:

```text
Build
  ↓
Measure
  ↓
Inspect
  ↓
Explain
  ↓
Improve
```

The project does not treat "the application runs" as the only success criterion.

Important questions include:

- Does the analytical result make sense?
- Can the result be explained?
- Can the AI response be grounded?
- What happens when evidence is missing?
- What happens when the model produces unsupported information?
- What are the limitations of the dataset?
- Can another person reproduce the workflow?

---

# ❗ Limitations

This is a prototype, not a production environmental monitoring system.

## Dataset limitations

The analysis is based on the available historical dataset.

## Missing measurements

Some indicators contain missing values or are unavailable in the selected subset.

## Temporal limitations

Historical observations should not automatically be interpreted as current water-quality conditions.

## Spatial limitations

Six monitoring stations cannot represent every location within the wider river system.

## Causal limitations

The hotspot ranking identifies patterns in available measurements.

It does **not** prove the source or cause of pollution.

## Machine-learning limitations

The Logistic Regression experiment used a very small dataset, so its high test accuracy should not be interpreted as production-level predictive performance.

## Generative-AI limitations

LLMs can produce incorrect or unsupported statements.

Therefore, generated explanations should always be interpreted alongside the underlying evidence.

## Regulatory limitation

The hotspot score created in this project is a prototype analytical indicator.

It is **not an official regulatory classification**.

---

# 🚀 Future Improvements

If the system were developed further, possible improvements include:

### Data

- integrate newer monitoring observations,
- increase the number of monitoring stations,
- include additional water-quality indicators,
- improve temporal coverage,
- integrate additional environmental variables.

### Analytics

- improve hotspot scoring validation,
- perform stronger spatial validation,
- explore temporal forecasting,
- improve anomaly detection,
- compare additional baseline models.

### RAG

- expand and verify the knowledge base,
- improve retrieval quality,
- add citation tracking,
- implement stronger retrieval evaluation,
- add better out-of-scope detection.

### AI reliability

- structured output validation,
- retry handling,
- API timeout handling,
- rate-limit handling,
- fallback behavior,
- response validation.

### Application

- interactive maps,
- station comparison,
- trend visualization,
- automated reports,
- user roles,
- deployment infrastructure.

---

# 💭 Key Engineering Decisions

## Why not use AI for everything?

Because the underlying measurements can be analyzed deterministically.

There is little value in asking an LLM to calculate a hotspot score when Python can calculate it transparently and reproducibly.

Therefore:

```text
Deterministic computation → numbers
AI → explanation and language-based interaction
```

---

## Why build the baseline first?

Starting with a transparent baseline makes it possible to understand the data before introducing model complexity.

This also makes it easier to identify whether AI actually adds value.

---

## Why use structured evidence?

Providing the AI with computed station-level metrics reduces the opportunity for the model to invent analytical values.

---

## Why evaluate failures?

A system that works only when everything goes perfectly is not robust.

Failure analysis therefore became part of the project rather than an afterthought.

---

# 🧠 Lessons Learned

This project was not only about building an AI application.

It involved learning how to:

- inspect real-world datasets,
- handle missing data,
- build reproducible analytical pipelines,
- design interpretable scoring methods,
- understand data leakage,
- interpret ML metrics critically,
- separate deterministic logic from probabilistic AI,
- design RAG workflows,
- think about hallucination and grounding,
- evaluate AI outputs,
- handle responsible-AI concerns,
- build an application around analytical results,
- manage API credentials securely,
- use Git throughout iterative development,
- document engineering decisions.

One of the most important lessons was:

> **A complicated AI system is not automatically a better system.**

A transparent analytical foundation can be more valuable than asking a large language model to perform every part of the problem.

---

# 🏆 What This Project Demonstrates

CleanGanga-Prayagraj demonstrates an end-to-end AI + sustainability workflow:

```text
Real-World Problem
        ↓
Data Understanding
        ↓
Data Quality
        ↓
Exploratory Analysis
        ↓
Transparent Analytical Baseline
        ↓
Hotspot Detection
        ↓
Machine-Learning Baseline
        ↓
Evidence Generation
        ↓
RAG
        ↓
IBM Granite
        ↓
Responsible AI
        ↓
Evaluation
        ↓
Streamlit Prototype
```

The project therefore goes beyond simply creating an LLM chatbot.

It demonstrates how AI can be placed **on top of a measurable analytical foundation** to make environmental information easier to understand and act upon.

---

# 📸 Prototype Demonstration

The final project includes screenshots demonstrating the working application.

The demonstration covers:

1. Streamlit application startup.
2. Hotspot/monitoring information.
3. Station-level analytical evidence.
4. AI-assisted explanation.
5. Decision-support presentation.

---

# 🌱 Expected Impact

If developed with larger, current, and operational datasets, a system based on this architecture could help environmental teams:

- prioritize stations for investigation,
- understand persistent pollution patterns,
- identify unusual observations,
- communicate analytical evidence faster,
- reduce the effort required to interpret large monitoring datasets.

The system is intended to **assist human decision-makers**, not replace them.

---

# 📌 Project Status

### Completed

- [x] Dataset understanding
- [x] Data-quality analysis
- [x] Prayagraj filtering
- [x] Water-quality assessment
- [x] Station-level analysis
- [x] Hotspot methodology
- [x] Hotspot ranking
- [x] ML baseline
- [x] Analytical output generation
- [x] RAG architecture
- [x] AI decision-support workflow
- [x] Responsible-AI considerations
- [x] AI evaluation artifacts
- [x] Streamlit prototype
- [x] Prototype screenshots
- [x] Git-based project organization

### Finalization

- [ ] Final repository cleanup
- [ ] Final README review
- [ ] Final presentation
- [ ] Final submission package

---

# 👨‍💻 Project Author

**Sri Chyavan Gorti**

Developed as part of the:

**1M1B AI for Sustainability Virtual Internship**

in collaboration with:

**IBM SkillsBuild & AICTE**

---

# ⚠️ Disclaimer

CleanGanga-Prayagraj is an educational and research-oriented prototype.

The analysis is based on the available historical dataset and should not be interpreted as a current official assessment of river-water quality.

Hotspot scores are analytical indicators created for this project and are not official regulatory classifications.

AI-generated explanations should be reviewed against the underlying evidence and should not replace qualified environmental expertise or official monitoring procedures.

---

# ⭐ Final Principle

> **Build → Measure → Explain → Evaluate → Improve**

The goal of CleanGanga-Prayagraj is not simply to demonstrate that an AI model can generate an answer.

The goal is to demonstrate how **data, analytical reasoning, retrieval, AI, evaluation, and responsible design can work together to support better environmental decision-making.**