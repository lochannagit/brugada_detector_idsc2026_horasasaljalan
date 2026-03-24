# 🫀 Brugada Syndrome ECG Detection System

AI-powered ECG analysis for early detection of Brugada Syndrome, a leading cause of sudden cardiac death in Southeast Asia.

## 🎯 About
This system uses machine learning to analyze 12-lead ECG signals and detect patterns characteristic of Brugada Syndrome Type 1, particularly in precordial leads V1-V3.

## 📊 Features
- Automated ECG preprocessing and feature extraction
- ML-based classification (Random Forest, Extra Trees, Logistic Regression)
- Interactive web interface for clinical screening
- Risk probability assessment with confidence scores

## 🚀 Live Demo
https://brugadadetectoridsc2026horasasaljalan-6visssyjgfxru9yfffzomt.streamlit.app/

## 📁 Available Patient IDs
The following patient IDs are available for testing:

| Patient ID |
|------------|
| 188981 |
| 251972 |
| 265715 |
| 267628 |
| 267630 |
| 286830 |
| 287355 |
| 292220 |
| 292666 |
| 304141 |

**Total: 10 patient records available**

## 📚 Citation

**If you use this dataset, please cite:**

**Brugada-HUCA Dataset:**
> Garcia-Isla, G., et al. (2022). Brugada-HUCA Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/xxxxx

**PhysioNet Platform:**
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

---

## ⚠️ Disclaimer

**This tool is for research and screening purposes only.** Not intended for clinical diagnosis. **Always consult a qualified cardiologist.**

## 🛠️ Installation

### Local Development
```bash
git clone https://github.com/YOUR_USERNAME/brugada-ecg-detector.git
cd brugada-ecg-detector
pip install -r requirements.txt
streamlit run app.py

