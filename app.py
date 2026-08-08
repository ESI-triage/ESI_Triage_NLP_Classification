import streamlit as st
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel

# Set page title and layout
st.set_page_config(page_title="ED Triage Assistant", layout="centered")

st.title("Emergency Department Multimodal Triage System")
st.write("Predicts Emergency Severity Index (ESI) tiers using Bio-ClinicalBERT semantic embeddings + XGBoost.")

# Load Bio-ClinicalBERT and XGBoost (Cached so it loads only once into RAM)
@st.cache_resource
def load_model():
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()
    
    xgb = joblib.load("final_best_xgb_triage_model.pkl")
    return tokenizer, bert_model, xgb

tokenizer, raw_bert, xgb_model = load_model()

# Input Form
with st.form("triage_form"):
    clinical_note = st.text_area("Clinical Triage Note", "54yo F reports fever with mild shortness of breath on exertion...")
    
    col1, col2 = st.columns(2)
    with col1:
        sbp = st.number_input("Systolic BP (mmHg)", value=120)
        dbp = st.number_input("Diastolic BP (mmHg)", value=80)
        hr = st.number_input("Heart Rate (bpm)", value=75)
        rr = st.number_input("Respiratory Rate", value=16)
    with col2:
        temp = st.number_input("Temperature (°C)", value=37.0)
        spo2 = st.number_input("SpO2 (%)", value=98.0)
        pain = st.slider("Pain Score (0-10)", 0, 10, 2)
        
    submit = st.form_submit_button("Predict ESI Level")

# Prediction Logic
if submit:
    # Extract BERT embedding
    encoded = tokenizer([clinical_note], truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = raw_bert(**encoded)
        text_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    # Concatenate with Vitals
    vitals = np.array([[sbp, dbp, hr, rr, temp, spo2, pain]])
    combined = np.concatenate([text_embedding, vitals], axis=1)

    # Predict Probabilities
    probs = xgb_model.predict_proba(combined)[0]
    pred_class = np.argmax(probs) + 1
    
    st.success(f"Predicted Acuity Tier: **ESI {pred_class}**")
    st.bar_chart({f"ESI {i+1}": float(probs[i]) for i in range(5)})