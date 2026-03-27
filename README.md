# Liver Cirrhosis Stage Detection Web App 🩺

**🌐 Live Demo:** [https://liver-cirrhosis-detection.streamlit.app/](https://liver-cirrhosis-detection.streamlit.app/)

An interactive, Machine Learning-powered web application that predicts the stage of liver cirrhosis (Stages 1 through 4) based on a patient's clinical and biochemical data. Built as part of the Unified Mentor Internship Project.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

## 🎯 Objective
Empower medical professionals and students with a predictive tool for the staging of liver cirrhosis. By inputting 18 clinical features (such as Bilirubin, Albumin, Prothrombin time, and presence of Ascites), the model classifies the disease progression into four stages: Inflammation, Fibrosis, Cirrhosis, and End-Stage. Early detection is critical for timely intervention.

## ✨ Features
- **Comprehensive Input Form:** Beautiful UI for entering patient demographics, clinical symptoms, and lab values.
- **Stage Probability Distribution:** An interactive Plotly chart that visualizes the model's certainty across all four stages.
- **Detailed Evaluation:** Embedded dashboard showing model accuracy (~88%), precision, and recall alongside feature importance analysis.
- **Educational Context:** Includes medical glossary terms for features like Hepatomegaly and Spiders, as well as baseline normal ranges for all lab metrics.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abhiguru25/liver-cirrhosis-stage.git
cd liver-cirrhosis-stage/liver_cirrhosis_stage
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Methodology & Preprocessing
- **Handling Missing Values:** Strategy implemented during training for incomplete clinical records.
- **Encoding:** Categorical variables (`Sex`, `Drug`, `Edema`, etc.) are processed using Label Encoding.
- **Scaling:** Continuous variables (`Alk_Phos`, `Prothrombin` etc.) are standardized using `StandardScaler` to handle extreme variances in medical ranges.
- **Algorithm:** Multi-class classifier yielding discrete stage outputs.

## 📁 Repository Structure
```
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Python dependencies
├── liver_cirrhosis_model.pkl   # Serialized ML Model
├── liver_scaler.pkl            # Serialized StandardScaler
├── stage_label_encoder.pkl     # Serialized LabelEncoder for the target variable
└── README.md                   # This documentation
```

---
*Developed by Abhivirani | Unified Mentor Internship Project*
