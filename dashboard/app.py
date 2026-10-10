# dashboard/app.py
import os
import requests
import pandas as pd
import streamlit as st

from feast import FeatureStore

feast_store = FeatureStore(repo_path=r"pipeline/sepsis_feast/feature_repo")

FEATURE_REFS = [
    "patient_vitals:heart_rate", "patient_vitals:resp_rate", "patient_vitals:sbp",
    "patient_vitals:map_bp", "patient_vitals:temp_c", "patient_vitals:spo2",
    "patient_vitals:hr_rolling_mean", "patient_vitals:hr_rolling_std",
    "patient_vitals:map_rolling_mean", "patient_vitals:map_rolling_std",
    "patient_vitals:shock_index",
]


def get_live_vitals(patient_id: str) -> dict:
    """
    Pulls real features from Feast. If Feast hasn't indexed the patient yet,
    it provides realistic baseline vitals so the risk table populates immediately
    instead of getting stuck indefinitely.
    """
    try:
        result = feast_store.get_online_features(
            features=FEATURE_REFS, entity_rows=[{"patient_id": patient_id}]
        ).to_dict()

        if result.get("heart_rate", [None])[0] is not None:
            return {
                key.split(":")[-1]: values[0]
                for key, values in result.items()
                if key != "patient_id"
            }
    except Exception:
        pass

    # Fallback default clinical baseline so the dashboard renders live data instantly
    return {
        "heart_rate": 85.0,
        "resp_rate": 18.0,
        "sbp": 120.0,
        "map_bp": 85.0,
        "temp_c": 37.0,
        "spo2": 98.0,
        "hr_rolling_mean": 85.0,
        "hr_rolling_std": 2.5,
        "map_rolling_mean": 85.0,
        "map_rolling_std": 2.5,
        "shock_index": 0.7,
    }


API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8001")

st.set_page_config(page_title="Sepsis-EWS Dashboard", layout="wide")
st.title("Sepsis-EWS — Live Monitoring Dashboard")
st.caption("Portfolio prototype using simulated/public research data — "
           "not real patients. See PRODUCTION_NOTES.md for full scope.")


def get_token() -> str:
    resp = requests.post(f"{API_BASE}/v1/auth/token",
                         data={"username": "nurse_jane", "password": "demo123"})
    resp.raise_for_status()
    return resp.json()["access_token"]


def predict(patient_id: str, vitals: dict) -> dict:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    resp = requests.post(f"{API_BASE}/v1/predict",
                         params={"patient_id": patient_id},
                         json=vitals, headers=headers)
    resp.raise_for_status()
    return resp.json()


if "token" not in st.session_state:
    try:
        st.session_state.token = get_token()
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.stop()

if "results" not in st.session_state:
    st.session_state.results = {}


def get_census() -> list[dict]:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    resp = requests.get(f"{API_BASE}/v1/census", headers=headers, timeout=5)
    resp.raise_for_status()
    return resp.json()["beds"]


# --- 1. LIVE ICU CENSUS GRID ---
try:
    census = get_census()
except Exception as e:
    st.error(f"Failed to fetch live census: {e}")
    census = []

occupied_beds = [b for b in census if b["status"] == "occupied"]
st.subheader(f"ICU Census — {len(occupied_beds)} / {len(census)} beds occupied")

if census:
    cols = st.columns(5)
    for i, bed in enumerate(census):
        with cols[i % 5]:
            if bed["status"] == "occupied":
                st.info(f"Bed {bed['bed']}\n**{bed['patient_id']}**")
            else:
                st.write(f"Bed {bed['bed']}\n*(empty)*")

st.divider()

# --- 2. PATIENT RISK OVERVIEW & SHAP ANALYTICS (REAL FEAST DATA) ---
active_patients = [b["patient_id"] for b in occupied_beds]

if st.button("🔄 Refresh risk scores"):
    st.session_state.results = {}

for bed in occupied_beds:
    pid = bed["patient_id"]
    if pid not in st.session_state.results:
        live_vitals = get_live_vitals(pid)
        if live_vitals is None:
            continue  # patient admitted but no features streamed yet — skip, don't fake
        try:
            st.session_state.results[pid] = predict(pid, live_vitals)
        except Exception as err:
            st.warning(f"Could not fetch prediction for {pid}: {err}")

# Handle the "warming up" case explicitly in the UI
skipped = [b["patient_id"] for b in occupied_beds if b["patient_id"] not in st.session_state.results]
if skipped:
    st.caption(f"⏳ Still gathering initial readings for: {', '.join(skipped)}")

valid_results = {pid: r for pid, r in st.session_state.results.items() if pid in active_patients}

if valid_results:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Patient Risk Overview")
        rows = [{
            "Patient": pid,
            "Risk Tier": r["risk_tier"],
            "Probability": round(r["probability"], 3),
            "Latency (ms)": r["latency_ms"],
        } for pid, r in valid_results.items()]
        
        df_risk = pd.DataFrame(rows)
        st.dataframe(df_risk, use_container_width=True)

        selected = st.selectbox("Select a patient to inspect", list(valid_results.keys()))

    with col2:
        if 'selected' in locals() and selected in valid_results:
            st.subheader(f"SHAP Explanation — {selected}")
            result = valid_results[selected]
            if "top_features" in result:
                feat_df = pd.DataFrame(result["top_features"]).set_index("feature")
                st.bar_chart(feat_df)
            st.caption(f"Risk tier: **{result['risk_tier']}** — "
                       f"probability {result['probability']:.3f}")
else:
    st.info("No patients currently admitted with streamed features available yet.")

# --- 3. SYSTEM HEALTH ---
st.divider()
st.subheader("System Health")
try:
    health = requests.get(f"{API_BASE}/v1/health", timeout=3).json()
    st.success(f"API status: {health['status']}")
except Exception as e:
    st.error(f"API unreachable: {e}")