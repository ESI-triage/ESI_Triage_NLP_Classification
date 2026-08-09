import streamlit as st
import numpy as np
import joblib
import requests

st.set_page_config(page_title="ED Triage Assistant", layout="centered")
st.title("Emergency Department Multimodal Triage System")
st.write("Predicts Emergency Severity Index (ESI) tiers using Bio-ClinicalBERT semantic embeddings + XGBoost.")
st.caption("⚠️ Research/educational prototype only. Not validated for clinical decision-making.")

BERT_API_URL = "https://esi-triage-bioclinicalbert-api-1042542474746.us-central1.run.app"  # <-- update this


@st.cache_resource
def load_xgb_model():
    return joblib.load("final_best_xgb_triage_model.pkl")


xgb_model = load_xgb_model()


def get_bert_embedding(text: str) -> np.ndarray:
    try:
        response = requests.post(BERT_API_URL, json={"text": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"]).reshape(1, -1)
    except requests.exceptions.Timeout:
        st.error("The embedding service timed out. It may be cold-starting — try again in a moment.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the embedding service: {e}")
        st.stop()


with st.form("triage_form"):
    clinical_note = st.text_area(
        "Clinical Triage Note",
        "54yo F reports fever with mild shortness of breath on exertion...",
    )

    col1, col2 = st.columns(2)
    with col1:
        sbp = st.number_input("Systolic BP (mmHg)", value=120)
        dbp = st.number_input("Diastolic BP (mmHg)", value=80)
        hr = st.number_input("Heart Rate (bpm)", value=75)
        rr = st.number_input("Respiratory Rate", value=16)
    with col2:
        temp = st.number_input("Temperature (\u00b0C)", value=37.0)
        spo2 = st.number_input("SpO2 (%)", value=98.0)
        pain = st.slider("Pain Score (0-10)", 0, 10, 2)

    submit = st.form_submit_button("Predict ESI Level")

if submit:
    with st.spinner("Analyzing clinical note..."):
        text_embedding = get_bert_embedding(clinical_note)

        vitals = np.array([[sbp, dbp, hr, rr, temp, spo2, pain]])
        combined = np.concatenate([text_embedding, vitals], axis=1)

        expected_features = xgb_model.n_features_in_
        if combined.shape[1] != expected_features:
            st.error(f"Feature mismatch: got {combined.shape[1]}, model expects {expected_features}")
            st.stop()

        probs = xgb_model.predict_proba(combined)[0]
        pred_class = xgb_model.classes_[np.argmax(probs)]

    st.success(f"Predicted Acuity Tier: **ESI {pred_class}**")
    st.bar_chart({f"ESI {xgb_model.classes_[i]}": float(probs[i]) for i in range(len(probs))})
