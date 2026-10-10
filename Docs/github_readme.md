# Sepsis-EWS: Real-Time, Explainable Sepsis Deterioration Early-Warning System

[![CI/CD Pipeline](https://github.com/your-username/Sepsis-EWS/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Sepsis-EWS/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-ff5722)](https://feast.dev/)

> **Portfolio & Engineering Prototype Notice:** Sepsis-EWS is an advanced MLOps and systems engineering demonstration designed to solve real-time latency, explainability, and drift challenges in clinical telemetry. **It is built entirely on public, de-identified research data (PhysioNet 2019 Sepsis Challenge dataset) and is NOT a cleared or validated medical device.** See [PRODUCTION_NOTES.md](PRODUCTION_NOTES.md) for full scope boundaries.

---

## 📌 1. Executive Summary & Problem Statement

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. In acute care settings, patient deterioration frequently manifests as a slow, multivariate drift across multiple vital signs simultaneously—a pattern extremely difficult for clinicians to synthesize manually during high-stress shifts. 

Traditional rule-based scoring systems (e.g., NEWS2, qSOFA) rely on fixed static thresholds that cause severe **alarm fatigue** (a recognized National Patient Safety Goal). Meanwhile, commercial EHR AI models have struggled with real-world transportability and calibration drift. 

**Sepsis-EWS** provides a production-grade blueprint for real-time clinical decision support featuring:
1. **Streaming Ingestion & Validation:** Simulated IoT/FHIR telemetry ingested via Redpanda with Pydantic v2 strict contracts.
2. **Online Feature Store Architecture:** Feast + Redis delivering sub-millisecond feature retrieval with zero train/serve skew.
3. **Cost-Sensitive Inference & Fast Explainability:** LightGBM optimized via Optuna with sub-12ms SHAP TreeExplainer attribution.
4. **SWADT (Novel Drift Monitor):** A custom drift-detection mechanism combining per-feature PSI and live SHAP-importance velocity.
5. **Structurally Enforced Safety:** API schema design that makes treatment/dosage generation architecturally impossible (`extra="forbid"`).

---

## 🏗️ 2. System Architecture & Data Flow

### 2.1 End-to-End Data Flow (ERD / Architecture Overview)

```ascii
+---------------------------+
| Patient Simulator / FHIR  |
+---------------------------+
              |
              v (Producer)
+---------------------------+
|  Redpanda (vitals.raw)    |
+---------------------------+
              |
              v (Consumer & Validator)
+---------------------------+      (Invalid)      +-------------------+
|    Pydantic v2 Contract   | ------------------> | Dead-Letter Queue |
+---------------------------+                     +-------------------+
              | (Valid)
              v
+---------------------------+
| Polars / Pandas Causal    | ---> Rolling Means, Stds, Shock Index
| Feature Engineering Engine|
+---------------------------+
              |
              v (Online Push)
+---------------------------+
| Feast + Redis Online Store| <------------------+
+---------------------------+                    |
              |                                  |
              v (Read Online Features)           |
+---------------------------+                    |
| FastAPI Inference Service |                    |
|  - LightGBM Classifier    |                    |
|  - SHAP TreeExplainer     |                    |
+---------------------------+                    |
      |                 |                        |
      v                 v (Shadow Evaluation)    |
+-------------+   +-------------------+          |
| SQLite Audit|   | SWADT Drift Monitor | --------+
|   Trail     |   +-------------------+
+-------------+          |
                         v (Drift Alert / Retrain Trigger)
                 +-------------------+
                 | Streamlit Live UI | ---> ICU Census Grid, Risk Tiers, SHAP Panel
                 +-------------------+
```

---

## 📊 3. Data Flow Diagrams (DFD Levels 0, 1, and 2)

### DFD Level 0 (Context Diagram)
```ascii
      +-------------------------------------------------+
      |             External Actors / Data              |
      |   - PhysioNet 2019 Raw Patient Records (.psv)   |
      |   - Clinical Staff / ICU Nurses                 |
      +-------------------------------------------------+
                               |
                               | Telemetry & Audit / UI Actions
                               v
                   +-----------------------+
                   |                       |
                   |      SEPSIS-EWS       |
                   |    System Boundary    |
                   |                       |
                   +-----------------------+
                               |
                               | Risk Scores, SHAP Charts, Census Grid
                               v
      +-------------------------------------------------+
      |                 Authorized User                 |
      |          (Nurse / Clinician / Admin)            |
      +-------------------------------------------------+
```

### DFD Level 1 (Major Subsystems)
```ascii
 [Raw Files] ---> (1.0 Ingestion & Validation) 
                           |
                           v (Clean Events)
                 (2.0 Feature Engineering) 
                           |
                           v (Rolling Metrics)
                 (3.0 Feast Feature Store) 
                           |
            +--------------+--------------+
            |                             |
            v                             v
(4.0 Inference & SHAP)        (5.0 SWADT Drift Engine)
            |                             |
            v                             v
(6.0 Audit & Logging)         (7.0 Streamlit Dashboard)
```

### DFD Level 2 (Detailed Component Breakdown: Inference & Drift)
```ascii
 [Feast Online Store] 
         |
         +---> Read Entity Rows 
         |
         v
 [FastAPI Predict Endpoint] ---> Authenticate JWT / RBAC
         |
         +---> Fetch Model (LightGBM)
         |     |
         |     v Compute Probability
         |     +---> Check Risk Tier (Stable / At-Risk / Deteriorating)
         |
         +---> Fetch Explainer (SHAP KMeans background)
               |
               v Calculate Feature Attributions (~11.46ms)
               |
               +---> Return JSON Payload
                       |
                       +---> Append Record to SQLite Audit Log (`audit.db`)
                       |
                       +---> Feed SWADT Shadow Monitor (PSI vs SHAP Velocity)
```

---

## 📐 4. Class Diagrams & UML Workflows

### 4.1 Core Class Structure
```ascii
+-----------------------+       +------------------------+       +------------------------+
|      AuthManager      |       |      FeatureStore      |       |    InferenceEngine     |
+-----------------------+       +------------------------+       +------------------------+
| - secret_key: str     |       | - repo_path: str       |       | - model: LGBMClassifier|
| - algorithm: str      |       | - store: Redis         |       | - explainer: TreeExpl. |
+-----------------------+       +------------------------+       +------------------------+
| + verify_password()   |       | + get_online_features()|       | + predict_proba()      |
| + create_token()      |       | + push_to_online()     |       | + compute_shap()       |
+-----------------------+       +------------------------+       +------------------------+
            ^                               ^                                ^
            |                               |                                |
            +-------------------------------+--------------------------------+
                                            |
                                            v
                                +------------------------+
                                |      FastAPI App       |
                                +------------------------+
                                | + post /v1/auth/token  |
                                | + get  /v1/census      |
                                | + post /v1/predict     |
                                | + get  /v1/health      |
                                +------------------------+
```

### 4.2 UML Sequence Diagram (Prediction & Audit Workflow)
```ascii
Streamlit UI       FastAPI (/v1/predict)      Feast/Redis      LightGBM / SHAP      SQLite Audit
     |                      |                      |                 |                 |
     |---(1) POST Request-->|                      |                 |                 |
     |    (JWT Bearer)      |                      |                 |                 |
     |                      |---(2) Get Features-->|                 |                 |
     |                      |<--(3) Return Data----|                 |                 |
     |                      |                      |                 |                 |
     |                      |---(4) Model Predict------------------->|                 |
     |                      |<--(5) Prob & Tier----------------------|                 |
     |                      |                      |                 |                 |
     |                      |---(6) SHAP Explainer------------------>|                 |
     |                      |<--(7) Top Features---------------------|                 |
     |                      |                      |                 |                 |
     |                      |---(8) Log Audit Record---------------------------------->|
     |<--(9) JSON Response--|                      |                 |                 |
```

---

## 🛠️ 5. Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Streaming Ingestion** | Redpanda (Kafka API-compatible) | Simulates live ICU monitor telemetry as a partitioned event stream |
| **Data Validation** | Pydantic v2 contracts | Schema & range validation; dead-letter routing for invalid rows |
| **Feature Engineering** | Polars (streaming) / pandas | Causal rolling-window features: multi-resolution means/std, shock index |
| **Feature Store** | Feast + Redis | Sub-millisecond feature retrieval; eliminates train/serve skew |
| **Offline Model** | LightGBM + Optuna | Cost-sensitive gradient-boosted classifier (tuned `scale_pos_weight`) |
| **Explainability** | SHAP TreeExplainer | Per-prediction feature attribution (~11.46ms measured latency) |
| **Drift Monitoring** | SWADT (Custom) & River | Fuses per-feature PSI drift with live SHAP-importance velocity |
| **Serving API** | FastAPI, JWT Auth, RBAC | Structurally advisory-only schema (`extra="forbid"`, no treatment field) |
| **Audit Trail** | SQLite (append-only) | Full predict -> disposition -> acknowledgment record for governance |
| **Experiment Tracking** | MLflow + PostgreSQL | Run/metric/artifact tracking and model registry |
| **Orchestration** | Docker Compose (8 services) | Redpanda, Redis, Postgres, MLflow, pipeline, API, drift monitor, UI |
| **CI/CD** | GitHub Actions | Automated test suite + Docker build verification on every push |
| **Dashboard** | Streamlit | Live ICU census, risk-tier sorted patient grid, SHAP explanation panel |

---

## 🚀 6. Setup & Installation Instructions

### Prerequisites
* Python 3.10 or higher
* Docker & Docker Compose installed and running
* PowerShell (Windows) or Bash (Linux/macOS)

### Step 1: Clone the Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/Sepsis-EWS.git
cd Sepsis-EWS

python -m venv venv
# On Windows (PowerShell):
& venv\Scripts\Activate.ps1
# On Unix/macOS:
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Start Infrastructure via Docker Compose
Bring up Redis, Redpanda, PostgreSQL, and MLflow services:
```bash
docker compose -f infra/docker-compose.yml up -d
```

### Step 3: Run the Local Services (Development Mode)

1. **Start the FastAPI Inference Backend on Port 8001:**
   ```bash
   $env:PYTHONPATH="E:\SDEWS" # (Or export PYTHONPATH on Unix)
   python -B -m uvicorn api.main:app --port 8001 --reload
   ```

2. **Start the Ingestion Consumer (in a new terminal):**
   ```bash
   & venv\Scripts\Activate.ps1
   $env:PYTHONPATH="E:\SDEWS"
   python pipeline/ingestion/consumer.py
   ```

3. **Start the Feature Pipeline Transformer (in a new terminal):**
   ```bash
   & venv\Scripts\Activate.ps1
   $env:PYTHONPATH="E:\SDEWS"
   python pipeline/features/transforms.py
   ```

4. **Launch the Streamlit Dashboard (in a new terminal):**
   ```bash
   & venv\Scripts\Activate.ps1
   streamlit run dashboard/app.py
   ```
   * Open your browser at `http://localhost:8502` to view the live ICU census grid and Feast-backed risk analytics.

---

## 🧪 7. Running Tests & Validation

Run the comprehensive test suite covering validation contracts, feature calculation, API routes, and model inference:
```pytest
pytest -v --cov=api --cov=pipeline
```

---
*Clinical Disclaimer: This software is a research prototype intended solely for portfolio demonstration and engineering evaluation. It must never be used for clinical diagnosis, patient triage, or healthcare decision-making.*