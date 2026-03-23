import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from brugada_pipeline import preprocess_ecg_signal, extract_ecg_features


# 1. Page Configuration
st.set_page_config(page_title="Brugada Detector", layout="wide")

# --- SMART PATH RESOLUTION (UPDATED FOR GITHUB) ---
# Get the directory where app.py is sitting on the GitHub server
BASE_DIR = Path(__file__).resolve().parent

# Check three common locations to find the project root
if (BASE_DIR / "files").exists():
    # Scenario 1: Everything is in the main folder
    PROJECT_ROOT = BASE_DIR
elif (BASE_DIR / "brugada_project" / "files").exists():
    # Scenario 2: Everything is inside a 'brugada_project' subfolder
    PROJECT_ROOT = BASE_DIR / "brugada_project"
else:
    # Scenario 3: Fallback (default to current directory)
    PROJECT_ROOT = BASE_DIR

# Final Paths - using / operator for cross-platform compatibility
MODEL_PATH = PROJECT_ROOT / "saved_model" / "best_brugada_model.pkl"
files_folder = str(PROJECT_ROOT / "files")

# 2. Model Loading
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Model file missing! Expected at: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Version Mismatch Error: {e}")
        st.info("Try retraining your model with the latest scikit-learn version.")
        return None

model = load_model()
# --- SIDEBAR: Medical & DOSM Context ---
with st.sidebar:
    st.header("📋 Medical Reference")
    st.markdown("""
    **Brugada Type 1:** Characterized by a 'coved' ST-segment elevation ≥2mm in leads V1-V3.
    
    **Why this matters in Malaysia:**
    According to **DOSM**, heart disease is the #1 cause of death. Brugada is a leading cause of Sudden Unexplained Nocturnal Death Syndrome (SUNDS) in Southeast Asia.
    """)
    st.divider()
    st.warning("⚠️ For screening support only. Consult a cardiologist for clinical diagnosis.")
    st.info(f"📁 **Data Source:** {PROJECT_ROOT.name}")

# 2. Main UI Header
st.title("🫀 Brugada Syndrome AI Detector")
st.write("Early detection of cardiac abnormalities using ML-based ECG analysis.")

# 3. Input Section
patient_id = st.text_input("Enter Patient ID:", value="188981")

if st.button("Run Diagnostic Analysis"):
    if model is None:
        st.error("Model not loaded. Check path settings.")
    else:
        try:
        # 4. Feature Extraction
            with st.spinner('Analyzing ECG signals...'):
                features = extract_ecg_features(patient_id, files_folder)
                input_df = pd.DataFrame([features])
            
                # Prediction Logic
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

            # 5. Dashboard Layout
            st.divider()
            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.subheader("Diagnostic Result")
                if prediction == 1:
                    st.error(f"### 🚨 HIGH RISK DETECTED")
                    st.write(f"Confidence Score: **{probability:.2%}**")
                else:
                    st.success(f"### ✅ LOW RISK / NORMAL")
                    st.write(f"Confidence Score: **{(1-probability):.2%}**")
            
            # --- Risk Assessment Gauge ---
                st.write("#### Risk Probability Gauge")
                st.progress(float(probability))
                st.caption("Lower Risk 👈 [Probability Scale] 👉 Higher Risk")

            with col2:
                st.subheader("Key Signal Metrics (V1-V3)")
                # Clean up the chart labels for the judges
                chart_data = input_df[['V1_std', 'V2_std', 'V3_std']].T
                chart_data.columns = ['Standard Deviation']
                st.bar_chart(chart_data)
                st.caption("Standard Deviation across precordial leads V1, V2, and V3.")

        # 6. Technical Insights
            with st.expander("View Raw Extracted Features"):
                st.dataframe(input_df)

        # 7. Downloadable Report
            report_text = f"Patient ID: {patient_id}\nResult: {'High Risk' if prediction == 1 else 'Low Risk'}\nProbability: {probability:.2%}"
            st.download_button("📥 Download Summary Report", report_text, file_name=f"Brugada_Report_{patient_id}.txt")
        

        except Exception as e:
            st.error(f"Diagnostic Error: {e}")
            st.info("Verify the Patient ID exists in the 'files' folder.")
# Footer - National Context
st.divider()
st.markdown("""
<div style="text-align: center; color: grey;">
    Developed for the International Data Science Challenge 2026 by Horas Asal Jalan
</div>
""", unsafe_allow_html=True)