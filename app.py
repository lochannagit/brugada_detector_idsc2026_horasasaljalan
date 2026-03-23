import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys
from pathlib import Path
import traceback

# Add the current directory to path to import brugada_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import with error handling
try:
    from brugada_pipeline import preprocess_ecg_signal, extract_ecg_features
except ImportError as e:
    st.error(f"Failed to import brugada_pipeline: {e}")
    st.stop()

# 1. Page Configuration
st.set_page_config(page_title="Brugada Detector", layout="wide")

# --- SMART PATH RESOLUTION ---
BASE_DIR = Path(__file__).resolve().parent

# Try different possible folder structures
def find_files_folder(base_dir):
    """Find the files folder in various possible locations"""
    possible_paths = [
        base_dir / "files",
        base_dir / "brugada_project" / "files",
        base_dir / "brugada-huca-dataset" / "files",
        base_dir / "brugada_project" / "brugada-huca-dataset" / "files"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # If no files folder found, return None
    return None

# Find files folder
files_folder = find_files_folder(BASE_DIR)

# Find model
def find_model_path(base_dir):
    """Find the model file in various possible locations"""
    possible_paths = [
        base_dir / "saved_model" / "best_brugada_model.pkl",
        base_dir / "brugada_project" / "saved_model" / "best_brugada_model.pkl",
        base_dir / "best_brugada_model.pkl"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

MODEL_PATH = find_model_path(BASE_DIR)

# 2. Model Loading
@st.cache_resource
def load_model():
    if MODEL_PATH is None or not MODEL_PATH.exists():
        st.warning(f"⚠️ Model file not found. The app will run in demo mode.")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ Model loaded from: {MODEL_PATH.name}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("The app will run in demo mode with mock predictions.")
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
    
    if files_folder:
        st.info(f"📁 **Data Source:** Found ECG files")
    else:
        st.warning("📁 **Data Source:** No ECG files found. Using demo mode.")

# 3. Main UI Header
st.title("🫀 Brugada Syndrome AI Detector")
st.write("Early detection of cardiac abnormalities using ML-based ECG analysis.")

# 4. Input Section
patient_id = st.text_input("Enter Patient ID:", value="188981")

# Add target length option
target_length = st.slider("ECG Signal Length (samples):", 
                          min_value=1000, 
                          max_value=10000, 
                          value=5000, 
                          step=500,
                          help="Adjust the number of samples used for analysis")

if st.button("Run Diagnostic Analysis"):
    # Check if we have the files folder
    if not files_folder:
        st.error("❌ ECG files folder not found. Please check the deployment structure.")
        st.info("Make sure the 'files' folder containing ECG data is in the correct location.")
        st.code(f"Searched in: {BASE_DIR}")
        st.stop()
    
    try:
        # 5. Feature Extraction
        with st.spinner('Analyzing ECG signals...'):
            # Fix: Pass target_length parameter
            features = extract_ecg_features(patient_id, files_folder, target_length=target_length)
            input_df = pd.DataFrame([features])
            
            # Prediction Logic
            if model is not None:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
            else:
                # Demo mode: random prediction based on some features
                st.warning("⚠️ Running in DEMO MODE - Using mock predictions")
                # Simple mock logic based on V1 features if available
                if 'V1_std' in features:
                    probability = min(0.9, max(0.1, features['V1_std'] * 0.5))
                else:
                    probability = np.random.random()
                prediction = 1 if probability > 0.5 else 0

        # 6. Dashboard Layout
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
            
            # Risk Assessment Gauge
            st.write("#### Risk Probability Gauge")
            st.progress(float(probability))
            st.caption("Lower Risk 👈 [Probability Scale] 👉 Higher Risk")

        with col2:
            st.subheader("Key Signal Metrics (V1-V3)")
            # Check if V1, V2, V3 features exist
            std_cols = []
            for lead in ['V1', 'V2', 'V3']:
                col_name = f"{lead}_std"
                if col_name in features:
                    std_cols.append(col_name)
            
            if std_cols:
                chart_data = input_df[std_cols].T
                chart_data.columns = ['Standard Deviation']
                st.bar_chart(chart_data)
                st.caption("Standard Deviation across precordial leads V1, V2, and V3.")
            else:
                st.info("Precordial lead features not available in the extracted data")

        # 7. Technical Insights
        with st.expander("View Raw Extracted Features"):
            # Show only first 20 features to avoid clutter
            display_df = pd.DataFrame([{k: v for k, v in features.items() if not isinstance(v, (list, dict))}])
            st.dataframe(display_df)
        
        # 8. Additional ECG Metrics
        with st.expander("Detailed Signal Analysis"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Basic Signal Statistics**")
                stats = {
                    "Global Mean": features.get("global_mean", "N/A"),
                    "Global Std": features.get("global_std", "N/A"),
                    "Sampling Rate": features.get("fs", "N/A"),
                    "Signal Length": features.get("n_samples", "N/A")
                }
                for key, val in stats.items():
                    st.write(f"- {key}: {val:.2f}" if isinstance(val, (int, float)) else f"- {key}: {val}")
            
            with col_b:
                st.write("**Precordial Lead Metrics**")
                if 'precordial_max_peak' in features:
                    st.write(f"- Max Peak: {features['precordial_max_peak']:.3f}")
                if 'precordial_avg_std' in features:
                    st.write(f"- Avg Std: {features['precordial_avg_std']:.3f}")
                if 'v1_slope' in features:
                    st.write(f"- V1 Slope: {features['v1_slope']:.3f}")

        # 9. Downloadable Report
        report_lines = [
            f"Patient ID: {patient_id}",
            f"Result: {'High Risk' if prediction == 1 else 'Low Risk'}",
            f"Probability: {probability:.2%}",
            f"Target Length: {target_length} samples",
            "\nKey Metrics:",
            f"- Global Signal Std: {features.get('global_std', 'N/A')}",
            f"- Precordial Max Peak: {features.get('precordial_max_peak', 'N/A')}",
            f"- Sampling Rate: {features.get('fs', 'N/A')} Hz"
        ]
        report_text = "\n".join(str(line) for line in report_lines)
        
        st.download_button(
            "📥 Download Summary Report", 
            report_text, 
            file_name=f"Brugada_Report_{patient_id}.txt"
        )

    except FileNotFoundError as e:
        st.error(f"❌ ECG file not found for patient ID: {patient_id}")
        st.info(f"Please ensure the folder structure is correct. Looking in: {files_folder}")
        st.code(f"Error details: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Diagnostic Error: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Verify the Patient ID exists in the 'files' folder with proper .dat and .hea files")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: grey;">
    Developed for the International Data Science Challenge 2026 | Brugada Syndrome Detection System
</div>
""", unsafe_allow_html=True)