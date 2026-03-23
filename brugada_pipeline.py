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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score
)

warnings.filterwarnings("ignore")

# -------------------------
# USER PATH SETTINGS (PORTABLE VERSION)
# -------------------------
import os

# 1. Get the directory where THIS script is currently saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define ZIP_PATH relative to your project folder
# This assumes the zip file is in the same folder as your script
ZIP_PATH = os.path.join(BASE_DIR, "brugada-huca-dataset.zip")

# 3. Define Output Folders inside the project directory
EXTRACT_DIR = os.path.join(BASE_DIR, "brugada_project")
PLOT_DIR = os.path.join(EXTRACT_DIR, "plots")
MODEL_DIR = os.path.join(EXTRACT_DIR, "saved_model")

# 4. Create directories if they don't exist
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Setup completed.")
print("Project folder (Absolute):", os.path.abspath(EXTRACT_DIR))

# =========================
# 2. LOAD DATA
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


# Extract dataset
unzip_dataset(ZIP_PATH, EXTRACT_DIR)

# Locate important paths
metadata_path = find_metadata_file(EXTRACT_DIR)
files_folder = find_files_folder(EXTRACT_DIR)

# Load metadata
df = pd.read_csv(metadata_path)

print("\nMetadata loaded successfully.")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())


# =========================
# 3. EDA
# =========================
print("\n====================")
print("EDA")
print("====================")

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic info:")
df.info()

print("\nTarget distribution (brugada):")
print(df["brugada"].value_counts(dropna=False))

if "basal_pattern" in df.columns:
    print("\nBasal pattern distribution:")
    print(df["basal_pattern"].value_counts(dropna=False))

if "sudden_death" in df.columns:
    print("\nSudden death distribution:")
    print(df["sudden_death"].value_counts(dropna=False))

print("\nDescriptive statistics:")
print(df.describe(include="all"))

# Keep only binary classes: 0 = healthy, 1 = confirmed Brugada
df_binary = df[df["brugada"].isin([0, 1])].copy()

print("\nBinary dataset shape (only class 0 and 1):", df_binary.shape)
print(df_binary["brugada"].value_counts())


# =========================
# 4. VISUALIZATION
# =========================
print("\n====================")
print("VISUALIZATION")
print("====================")

# 4A. Bar chart of target class
plt.figure(figsize=(6, 4))
df_binary["brugada"].value_counts().sort_index().plot(kind="bar")
plt.title("Brugada Class Distribution")
plt.xlabel("Class (0=Healthy, 1=Brugada)")
plt.ylabel("Number of Subjects")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
plt.show()


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


healthy_sample = df_binary[df_binary["brugada"] == 0]["patient_id"].iloc[0]
brugada_sample = df_binary[df_binary["brugada"] == 1]["patient_id"].iloc[0]

plot_sample_ecg(healthy_sample, 0, files_folder, "sample_healthy_ecg.png")
plot_sample_ecg(brugada_sample, 1, files_folder, "sample_brugada_ecg.png")


# =========================
# 5. PREPROCESSING
# =========================
print("\n====================")
print("PREPROCESSING")
print("====================")

df_binary["patient_id_str"] = df_binary["patient_id"].astype(int).astype(str)
df_binary["record_base_path"] = df_binary["patient_id_str"].apply(
    lambda x: os.path.join(files_folder, x, x)
)


def check_record_exists(base_path):
    return os.path.exists(base_path + ".hea") and os.path.exists(base_path + ".dat")


df_binary["record_exists"] = df_binary["record_base_path"].apply(check_record_exists)

print("\nRecord existence check:")
print(df_binary["record_exists"].value_counts())

# Keep only valid ECG records
df_binary = df_binary[df_binary["record_exists"]].copy()
df_binary.reset_index(drop=True, inplace=True)

print("\nShape after keeping valid records:", df_binary.shape)

# Inspect sample shapes
sample_lengths = []
for pid in df_binary["patient_id"].head(10):
    try:
        sig, leads, fs = load_ecg_record(pid, files_folder)
        sample_lengths.append(sig.shape)
    except Exception:
        sample_lengths.append(None)

print("\nExample ECG shapes from first 10 valid subjects:")
print(sample_lengths)


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
        # This prevents the FFT from seeing a massive "cliff" at the end of the signal
        pad_len = target_length - current_len
        processed = np.pad(processed, ((0, pad_len), (0, 0)), mode='edge')

    return processed


# =========================
# 6. FEATURE ENGINEERING
# =========================
print("\n====================")
print("FEATURE ENGINEERING")
print("====================")

def extract_ecg_features(patient_id, files_folder, target_length=5000):
    """
    Improved feature extraction from 12-lead ECG.
    Includes:
    - time-domain statistics
    - shape features
    - energy features
    - derivative features
    - simple frequency-domain features
    """
    signal, leads, fs = load_ecg_record(patient_id, files_folder)
    signal = preprocess_ecg_signal(signal, target_length=target_length)

    features = {}
    features["fs"] = fs
    features["n_samples"] = signal.shape[0]
    features["n_leads"] = signal.shape[1]

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
        features[f"{lead_name}_diff_mean"] = np.mean(dx)
        features[f"{lead_name}_diff_std"] = np.std(dx)
        features[f"{lead_name}_diff_energy"] = np.sum(dx**2)

        # Frequency-domain features
        fft_vals = np.abs(np.fft.rfft(x))
        fft_freqs = np.fft.rfftfreq(len(x), d=1/fs)

        if len(fft_vals) > 1:
            dominant_freq = fft_freqs[np.argmax(fft_vals[1:]) + 1]
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


feature_rows = []
failed_patients = []

for idx, row in df_binary.iterrows():
    pid = row["patient_id"]
    try:
        feats = extract_ecg_features(pid, files_folder, target_length=5000)
        feats["patient_id"] = pid
        feats["target"] = row["brugada"]
        feature_rows.append(feats)
    except Exception as e:
        failed_patients.append((pid, str(e)))

features_df = pd.DataFrame(feature_rows)

print("\nFeature matrix shape:", features_df.shape)
print("Number of failed patients during feature extraction:", len(failed_patients))

if len(failed_patients) > 0:
    print("First few failed patients:", failed_patients[:5])

print("\nSample engineered features:")
print(features_df.head())

print("\nFirst 20 feature names:")
print(features_df.columns[:20].tolist())

# Save engineered features
features_df.to_csv(os.path.join(EXTRACT_DIR, "engineered_features.csv"), index=False)
print("Engineered features saved.")


# =========================
# 7. ML MODEL TRAINING
# =========================
from sklearn.preprocessing import StandardScaler # Ensure this is imported

# ECG-only predictors
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
    "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

# 1. Cross-Validation Loop
for name, model in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()), 
        ("clf", model)
    ])

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    cv_results[name] = scores
    print(f"{name} Mean CV F1: {scores.mean():.4f}")

# 2. Identify the Best Model Name
# This line finds which key in the dictionary had the highest mean score
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest model found: {best_model_name}")

# 3. Build and Train the Final Pipeline using that best_model_name
best_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", models[best_model_name]) # Now best_model_name is definitely defined
])

best_pipeline.fit(X_train, y_train)
print("Best model training completed.")

joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_brugada_model.pkl"))
print("Best model saved successfully.")


# =========================
# 8. MODEL EVALUATION
# =========================
print("\n====================")
print("MODEL EVALUATION")
print("====================")

# Predictions
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:", round(acc, 4))
print("F1-score:", round(f1, 4))
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
try:
    auc_score = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(auc_score, 4))
except Exception as e:
    auc_score = None
    print("ROC-AUC could not be computed:", e)

# Confusion matrix plot
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
plt.show()

# ROC curve
if auc_score is not None:
    try:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))
        plt.show()
    except Exception as e:
        print("ROC curve could not be plotted:", e)

# Feature importance only for tree-based models
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
    plt.show()
else:
    print("Feature importance plot skipped because the best model is not tree-based.")

# Save predictions
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred
results_df["probability_brugada"] = y_prob
results_df.to_csv(os.path.join(EXTRACT_DIR, "test_predictions.csv"), index=False)

print("Predictions saved to test_predictions.csv")

# Save CV summary
cv_summary = pd.DataFrame({
    "Model": list(cv_results.keys()),
    "Mean_F1_CV": [cv_results[m].mean() for m in cv_results],
    "Std_F1_CV": [cv_results[m].std() for m in cv_results]
}).sort_values(by="Mean_F1_CV", ascending=False)

cv_summary.to_csv(os.path.join(EXTRACT_DIR, "cv_model_comparison.csv"), index=False)

print("\nCross-validation summary:")
print(cv_summary)
