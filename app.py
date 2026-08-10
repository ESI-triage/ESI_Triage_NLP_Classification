import streamlit as st
import numpy as np
import joblib
import requests

st.set_page_config(page_title="ED Triage Assistant", layout="centered")
st.title("Emergency Department Multimodal Triage System")
st.write("Predicts Emergency Severity Index (ESI) tiers using Bio-ClinicalBERT semantic embeddings + XGBoost.")
st.caption(
    ":red[Warning: Research/educational prototype only. Not validated for clinical decision-making.]"
)

BERT_API_URL = "https://esi-triage-bioclinicalbert-api-1042542474746.us-central1.run.app/embed"


@st.cache_resource
def load_xgb_model():
    return joblib.load("final_best_xgb_triage_model.pkl")


xgb_model = load_xgb_model()


def get_bert_embedding(text: str) -> np.ndarray:
    try:
        response = requests.post(
            BERT_API_URL, json={"text": text}, timeout=120)
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"]).reshape(1, -1)
    except requests.exceptions.Timeout:
        st.error("The embedding service timed out. Try again in a moment.")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the embedding service: {e}")
        st.stop()

# Defining a clear form function which will clear all the values entered in the fields


def clear_form():
    keys_to_clear = [
        "clinical_note", "age", "sex", "sbp", "dbp", "hr", "rr",
        "temp", "spo2", "pain", "wbc", "hb", "plt", "na", "k", "cr", "glu", "trop"
    ]
    for key in keys_to_clear:
        if key == "clinical_note":
            st.session_state[key] = ""
        else:
            st.session_state[key] = None


with st.form("triage_form"):
    clinical_note = st.text_area(
        "Clinical Triage Note *",
        value="",
        placeholder="e.g., 54yo F reports fever with mild shortness of breath on exertion...",
        key="clinical_note",
    )

    st.subheader("1. Patient Demographics")
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        age = st.number_input(
            "Age *",
            min_value=18,
            max_value=100,
            value=None,
            placeholder="e.g., 54",
            key="age",
        )
    with demo_col2:
        sex = st.selectbox(
            "Sex *",
            options=["Female", "Male"],
            index=None,
            placeholder="Select sex...",
            key="sex",
        )
        sex_val = 1 if sex == "Male" else 0 if sex == "Female" else None

    st.subheader("2. Vital Signs")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        sbp = st.number_input(
            "Systolic BP (mmHg) *",
            value=None,
            placeholder="e.g., 120.0",
            key="sbp",
        )
        temp = st.number_input(
            "Temperature (°C) *",
            value=None,
            placeholder="e.g., 37.0",
            key="temp",
        )
        hr = st.number_input(
            "Heart Rate (bpm) *",
            value=None,
            placeholder="e.g., 75.0",
            key="hr",
        )
        rr = st.number_input(
            "Respiratory Rate *",
            value=None,
            placeholder="e.g., 16.0",
            key="rr",
        )
    with v_col2:

        dbp = st.number_input(
            "Diastolic BP (mmHg) *",
            value=None,
            placeholder="e.g., 80.0",
            key="dbp",
        )
        spo2 = st.number_input(
            "SpO2 (%) *",
            value=None,
            placeholder="e.g., 98.0",
            key="spo2",
        )
        pain = st.number_input(
            "Pain Score (0-10) *",
            min_value=0,
            max_value=10,
            value=None,
            placeholder="e.g., 2",
            key="pain",
        )

    st.subheader("3. Core Laboratory Results (Optional)")
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        wbc = st.number_input(
            "WBC (10^3/µL)", value=None, placeholder="e.g., 7.50", key="wbc")
        hb = st.number_input(
            "Hemoglobin (g/dL)", value=None, placeholder="e.g., 14.00", key="hb")
        plt = st.number_input(
            "Platelet Count (10^3/µL)", value=None, placeholder="e.g., 250.00", key="plt")
        na = st.number_input(
            "Sodium (mEq/L)", value=None, placeholder="e.g., 140.00", key="na")
    with l_col2:
        k = st.number_input(
            "Potassium (mEq/L)", value=None, placeholder="e.g., 4.10", key="k")
        cr = st.number_input(
            "Creatinine (mg/dL)", value=None, placeholder="e.g., 0.90", key="cr")
        glu = st.number_input(
            "Glucose (mg/dL)", value=None, placeholder="e.g., 100.00", key="glu")
        trop = st.number_input(
            "Troponin (ng/mL)", value=None, placeholder="e.g., 0.02", key="trop")

    # add the predict ESI and the clear field buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submit = st.form_submit_button("Predict ESI Level", type="primary")
    with btn_col2:
        clear = st.form_submit_button("Clear All Fields", on_click=clear_form)

if submit:
    # Verify mandatory fields
    mandatory_fields = {
        "Clinical Triage Note": clinical_note.strip(),
        "Age": age,
        "Sex": sex,
        "Systolic BP": sbp,
        "Diastolic BP": dbp,
        "Heart Rate": hr,
        "Respiratory Rate": rr,
        "Temperature": temp,
        "SpO2": spo2,
        "Pain Score": pain,
    }

    missing_fields = [label for label,
                      val in mandatory_fields.items() if val is None or val == ""]

    if missing_fields:
        st.error(
            f"Please fill out all mandatory fields (*) before predicting: {', '.join(missing_fields)}"
        )
        st.stop()

    # Convert optional lab inputs: if empty, pass np.nan to XGBoost
    lab_vals = [
        np.nan if val is None else val
        for val in [wbc, hb, plt, na, k, cr, glu, trop]
    ]

    with st.spinner("Analyzing clinical data..."):
        text_embedding = get_bert_embedding(clinical_note)

        # 17 tabular features: 2 demographics + 7 vitals + 8 core labs
        tabular_features = np.array([[
            age, sex_val, sbp, dbp, hr, rr, temp, spo2, pain,
            *lab_vals
        ]])

        combined = np.concatenate([text_embedding, tabular_features], axis=1)

        expected_features = xgb_model.n_features_in_
        if combined.shape[1] != expected_features:
            st.error(
                f"Feature mismatch: got {combined.shape[1]}, model expects {expected_features}"
            )
            st.stop()

        probs = xgb_model.predict_proba(combined)[0]
        pred_class = xgb_model.classes_[np.argmax(probs)]

    st.success(f"Predicted Acuity Tier: **ESI {pred_class + 1}**")
    st.bar_chart({
        f"ESI {xgb_model.classes_[i] + 1}": float(probs[i])
        for i in range(len(probs))
    })
