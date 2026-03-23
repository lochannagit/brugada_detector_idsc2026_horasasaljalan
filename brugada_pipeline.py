# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 2026

Improved Brugada Syndrome ECG Pipeline
Spyder / Anaconda version
"""

# ============================================================
# BRUGADA SYNDROME ECG PIPELINE - IMPROVED VERSION
# ============================================================
# Flow:
# 1. setup
# 2. load data
# 3. eda
# 4. visualization
# 5. preprocessing
# 6. feature engineering
# 7. ML model training
# 8. model evaluation
# 9. optional streamlit app code generation
# ============================================================

# =========================
# 1. SETUP
# =========================
import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import joblib

from scipy.stats import skew, kurtosis
from scipy.signal import detrend

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # ADDED: Missing import
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score
)

warnings.filterwarnings("ignore")

# =========================
# PATH SETTINGS (SINGLE DEFINITION)
# =========================
# Get the directory where THIS script is currently saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to script location
ZIP_PATH = os.path.join(BASE_DIR, "brugada-huca-dataset.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "brugada_project")
PLOT_DIR = os.path.join(EXTRACT_DIR, "plots")
MODEL_DIR = os.path.join(EXTRACT_DIR, "saved_model")

# Create directories if they don't exist
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Setup completed.")
print("Project folder:", os.path.abspath(EXTRACT_DIR))


# =========================
# ECG PROCESSING FUNCTIONS
# =========================
def load_ecg_record(patient_id, files_folder):
    """
    Read one ECG record using WFDB.
    """
    patient_id = str(int(patient_id))
    record_path = os.path.join(files_folder, patient_id, patient_id)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal
    leads = record.sig_name
    fs = record.fs
    return signal, leads, fs


def preprocess_ecg_signal(signal, target_length=5000):
    """
    Improved ECG preprocessing:
    1. Replace NaN/Inf
    2. Detrend each lead (removes baseline wander)
    3. Z-score normalize each lead (on actual signal only)
    4. Standardize length (edge padding to avoid artificial DC jumps)
    """
    signal = np.asarray(signal, dtype=float)

    # 1. Replace NaN / inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Detrend each lead
    processed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        processed[:, i] = detrend(signal[:, i])

    # 3. Z-score normalization per lead (calculated BEFORE padding)
    for i in range(processed.shape[1]):
        mean_val = np.mean(processed[:, i])
        std_val = np.std(processed[:, i])
        if std_val > 0:
            processed[:, i] = (processed[:, i] - mean_val) / std_val
        else:
            processed[:, i] = processed[:, i] - mean_val

    # 4. Standardize signal length
    current_len = processed.shape[0]

    if current_len > target_length:
        # Truncate
        processed = processed[:target_length, :]
    elif current_len < target_length:
        # Edge Padding: repeats the last value instead of adding zeros
        pad_len = target_length - current_len
        processed = np.pad(processed, ((0, pad_len), (0, 0)), mode='edge')

    return processed


def extract_ecg_features(patient_id, files_folder, target_length=5000):
    """
    Improved feature extraction from 12-lead ECG.
    Includes:
    - time-domain statistics
    - shape features
    - energy features
    - derivative features
    - simple frequency-domain features
    - Brugada-specific lead grouping (V1, V2, V3)
    """
    signal, leads, fs = load_ecg_record(patient_id, files_folder)
    signal = preprocess_ecg_signal(signal, target_length=target_length)

    features = {}
    features["fs"] = fs
    features["n_samples"] = signal.shape[0]
    features["n_leads"] = signal.shape[1]

    # Extract features for each lead
    for i, lead_name in enumerate(leads):
        x = signal[:, i]

        # Time-domain statistics
        features[f"{lead_name}_mean"] = np.mean(x)
        features[f"{lead_name}_std"] = np.std(x)
        features[f"{lead_name}_var"] = np.var(x)
        features[f"{lead_name}_min"] = np.min(x)
        features[f"{lead_name}_max"] = np.max(x)
        features[f"{lead_name}_range"] = np.ptp(x)
        features[f"{lead_name}_median"] = np.median(x)
        features[f"{lead_name}_q25"] = np.percentile(x, 25)
        features[f"{lead_name}_q75"] = np.percentile(x, 75)
        features[f"{lead_name}_iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
        features[f"{lead_name}_abs_mean"] = np.mean(np.abs(x))
        features[f"{lead_name}_rms"] = np.sqrt(np.mean(x**2))
        features[f"{lead_name}_skew"] = skew(x)
        features[f"{lead_name}_kurtosis"] = kurtosis(x)

        # Energy / complexity
        features[f"{lead_name}_energy"] = np.sum(x**2)
        features[f"{lead_name}_zero_crossings"] = np.sum(np.diff(np.signbit(x)).astype(int))

        # First derivative features
        dx = np.diff(x)
        features[f"{lead_name}_diff_mean"] = np.mean(dx) if len(dx) > 0 else 0
        features[f"{lead_name}_diff_std"] = np.std(dx) if len(dx) > 0 else 0
        features[f"{lead_name}_diff_energy"] = np.sum(dx**2) if len(dx) > 0 else 0

        # Frequency-domain features
        fft_vals = np.abs(np.fft.rfft(x))
        fft_freqs = np.fft.rfftfreq(len(x), d=1/fs)

        if len(fft_vals) > 1:
            # Skip the first frequency (0 Hz)
            dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
            dominant_freq = fft_freqs[dominant_freq_idx]
            spectral_power = np.sum(fft_vals**2)
        else:
            dominant_freq = 0.0
            spectral_power = 0.0

        features[f"{lead_name}_dominant_freq"] = dominant_freq
        features[f"{lead_name}_spectral_power"] = spectral_power

    # Global features
    features["global_mean"] = np.mean(signal)
    features["global_std"] = np.std(signal)
    features["global_var"] = np.var(signal)
    features["global_min"] = np.min(signal)
    features["global_max"] = np.max(signal)
    features["global_range"] = np.ptp(signal)
    features["global_energy"] = np.sum(signal**2)
    features["global_abs_mean"] = np.mean(np.abs(signal))

    # ============================================================
    # BRUGADA-SPECIFIC LEAD GROUPING
    # V1, V2, and V3 are the "Diagnostic Leads" for Brugada
    # ============================================================
    precordial_leads = [l for l in leads if l in ['V1', 'V2', 'V3']]
    
    if precordial_leads:
        # 1. Precordial Max Amplitude (Checking for ST-elevation height)
        precordial_data = [signal[:, leads.index(l)] for l in precordial_leads]
        features["precordial_max_peak"] = np.max(precordial_data)
        
        # 2. Precordial Variance (Brugada often has high-frequency "notching")
        features["precordial_avg_std"] = np.mean([np.std(d) for d in precordial_data])
        
        # 3. V1-V2 Slope (Type 1 Brugada has a characteristic "down-sloping" ST segment)
        v1_idx = leads.index('V1') if 'V1' in leads else None
        if v1_idx is not None:
            v1_signal = signal[:, v1_idx]
            # Simple approximation of signal "tilt"
            features["v1_slope"] = np.mean(np.diff(v1_signal))
    
    return features


# =========================
# DATA LOADING FUNCTIONS
# =========================
def unzip_dataset(zip_path, extract_dir):
    """
    Extract zip file if not already extracted.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    print("Dataset extracted successfully.")


def find_metadata_file(base_dir):
    """
    Find metadata.csv inside extracted dataset.
    """
    for root, dirs, files in os.walk(base_dir):
        if "metadata.csv" in files:
            return os.path.join(root, "metadata.csv")
    raise FileNotFoundError("metadata.csv not found after extraction.")


def find_files_folder(base_dir):
    """
    Find ECG files folder inside extracted dataset.
    """
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "files":
            return root
    raise FileNotFoundError("files folder not found after extraction.")


def check_record_exists(base_path):
    """Check if ECG record files exist"""
    return os.path.exists(base_path + ".hea") and os.path.exists(base_path + ".dat")


def plot_sample_ecg(patient_id, label, files_folder, save_name):
    """
    Plot first 3 leads of a sample ECG.
    """
    try:
        signal, leads, fs = load_ecg_record(patient_id, files_folder)
        time_axis = np.arange(signal.shape[0]) / fs

        plt.figure(figsize=(12, 8))
        n_leads_to_plot = min(3, signal.shape[1])

        for i in range(n_leads_to_plot):
            plt.subplot(n_leads_to_plot, 1, i + 1)
            plt.plot(time_axis, signal[:, i])
            plt.title(f"Patient {patient_id} | Lead: {leads[i]} | Label: {label}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()

        plt.savefig(os.path.join(PLOT_DIR, save_name))
        plt.show()

    except Exception as e:
        print(f"Could not plot ECG for patient {patient_id}: {e}")


# =========================
# MAIN EXECUTION BLOCK
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BRUGADA SYNDROME ECG PIPELINE")
    print("="*60)
    
    # Step 1: Extract and load data
    print("\n[1] Loading dataset...")
    try:
        unzip_dataset(ZIP_PATH, EXTRACT_DIR)
        metadata_path = find_metadata_file(EXTRACT_DIR)
        files_folder = find_files_folder(EXTRACT_DIR)
        
        print(f"Metadata found at: {metadata_path}")
        print(f"ECG files folder: {files_folder}")
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Step 2: EDA
    print("\n[2] Exploratory Data Analysis...")
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic info:")
    df.info()
    
    print("\nTarget distribution (brugada):")
    print(df["brugada"].value_counts(dropna=False))
    
    # Keep only binary classes: 0 = healthy, 1 = confirmed Brugada
    df_binary = df[df["brugada"].isin([0, 1])].copy()
    print(f"\nBinary dataset shape (only class 0 and 1): {df_binary.shape}")
    print(df_binary["brugada"].value_counts())
    
    # Step 3: Visualization
    print("\n[3] Creating visualizations...")
    # Bar chart of target class
    plt.figure(figsize=(6, 4))
    df_binary["brugada"].value_counts().sort_index().plot(kind="bar")
    plt.title("Brugada Class Distribution")
    plt.xlabel("Class (0=Healthy, 1=Brugada)")
    plt.ylabel("Number of Subjects")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
    plt.close()
    
    # Plot sample ECGs
    try:
        healthy_sample = df_binary[df_binary["brugada"] == 0]["patient_id"].iloc[0]
        brugada_sample = df_binary[df_binary["brugada"] == 1]["patient_id"].iloc[0]
        
        plot_sample_ecg(healthy_sample, 0, files_folder, "sample_healthy_ecg.png")
        plot_sample_ecg(brugada_sample, 1, files_folder, "sample_brugada_ecg.png")
    except Exception as e:
        print(f"Could not plot sample ECGs: {e}")
    
    # Step 4: Preprocessing - check valid records
    print("\n[4] Preprocessing - validating ECG records...")
    df_binary["patient_id_str"] = df_binary["patient_id"].astype(int).astype(str)
    df_binary["record_base_path"] = df_binary["patient_id_str"].apply(
        lambda x: os.path.join(files_folder, x, x)
    )
    
    df_binary["record_exists"] = df_binary["record_base_path"].apply(check_record_exists)
    print("\nRecord existence check:")
    print(df_binary["record_exists"].value_counts())
    
    # Keep only valid ECG records
    df_binary = df_binary[df_binary["record_exists"]].copy()
    df_binary.reset_index(drop=True, inplace=True)
    print(f"\nShape after keeping valid records: {df_binary.shape}")
    
    # Step 5: Feature Engineering
    print("\n[5] Feature Engineering...")
    feature_rows = []
    failed_patients = []
    
    for idx, row in df_binary.iterrows():
        pid = row["patient_id"]
        if idx % 50 == 0:
            print(f"Processing patient {idx+1}/{len(df_binary)}...")
        
        try:
            feats = extract_ecg_features(pid, files_folder, target_length=5000)
            feats["patient_id"] = pid
            feats["target"] = row["brugada"]
            feature_rows.append(feats)
        except Exception as e:
            failed_patients.append((pid, str(e)))
    
    features_df = pd.DataFrame(feature_rows)
    
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"Number of failed patients: {len(failed_patients)}")
    
    # Save engineered features
    features_df.to_csv(os.path.join(EXTRACT_DIR, "engineered_features.csv"), index=False)
    print("Engineered features saved.")
    
    # Step 6: ML Model Training
    print("\n[6] Training ML models...")
    
    # Prepare features and target
    drop_cols = ["patient_id", "target"]
    X = features_df.drop(columns=drop_cols, errors="ignore")
    y = features_df["target"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    # Cross-validation loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", model)
        ])
        
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
        )
        cv_results[name] = scores
        print(f"{name} Mean CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Identify best model
    best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
    print(f"\nBest model found: {best_model_name} (Mean F1: {cv_results[best_model_name].mean():.4f})")
    
    # Train final pipeline
    best_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", models[best_model_name])
    ])
    
    best_pipeline.fit(X_train, y_train)
    print("Best model training completed.")
    
    # Save model
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_brugada_model.pkl"))
    print(f"Best model saved to: {os.path.join(MODEL_DIR, 'best_brugada_model.pkl')}")
    
    # Step 7: Model Evaluation
    print("\n[7] Evaluating model on test set...")
    
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    try:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"ROC-AUC could not be computed: {e}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Healthy", "Brugada"])
    plt.yticks([0, 1], ["Healthy", "Brugada"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()
    
    # Feature importance for tree-based models
    model_obj = best_pipeline.named_steps["clf"]
    
    if hasattr(model_obj, "feature_importances_"):
        importances = pd.Series(
            model_obj.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        top_n = 20
        top_features = importances.head(top_n)
        
        plt.figure(figsize=(10, 7))
        top_features.sort_values().plot(kind="barh")
        plt.title(f"Top {top_n} Important Features")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "feature_importance_top20.png"))
        plt.close()
    
    # Save predictions
    results_df = X_test.copy()
    results_df["actual"] = y_test.values
    results_df["predicted"] = y_pred
    results_df["probability_brugada"] = y_prob
    results_df.to_csv(os.path.join(EXTRACT_DIR, "test_predictions.csv"), index=False)
    
    # Save CV summary
    cv_summary = pd.DataFrame({
        "Model": list(cv_results.keys()),
        "Mean_F1_CV": [cv_results[m].mean() for m in cv_results],
        "Std_F1_CV": [cv_results[m].std() for m in cv_results]
    }).sort_values(by="Mean_F1_CV", ascending=False)
    
    cv_summary.to_csv(os.path.join(EXTRACT_DIR, "cv_model_comparison.csv"), index=False)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nResults saved in: {EXTRACT_DIR}")
    print(f"- Engineered features: {os.path.join(EXTRACT_DIR, 'engineered_features.csv')}")
    print(f"- Test predictions: {os.path.join(EXTRACT_DIR, 'test_predictions.csv')}")
    print(f"- Model comparison: {os.path.join(EXTRACT_DIR, 'cv_model_comparison.csv')}")
    print(f"- Saved model: {os.path.join(MODEL_DIR, 'best_brugada_model.pkl')}")
    print(f"- Plots: {PLOT_DIR}")