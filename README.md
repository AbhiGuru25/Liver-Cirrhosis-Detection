# Liver Cirrhosis Stage Detection 🩺

## Objective
Predict the level of patient liver damage (liver cirrhosis stage) given physical and biochemical test details (e.g., Bilirubin, Copper, Prothrombin time).

## Dataset
A tabular biomedical dataset describing various symptoms (Ascites, Spiders, Edema) alongside numerical health readings, utilized to predict the histological stage of cirrhosis (typically values from 1-4 depending on damage/scarring severity).

## Features
*   **Machine Learning Model:** Random Forest Classifier capable of interpreting both heavily skewed numeric biological distributions and binary encoded symptomatic attributes, balancing classes successfully to handle under-sampled late-stage instances.
*   **Web Interface:** A Streamlit clinical mock-up taking direct metric entries and giving a diagnostic stage prediction quickly.

## How to Run Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Files in this Repository
*   `Liver_Cirrhosis_Stage_Detection.ipynb`: Exploratory Data Analysis, missing value imputation, biological metric encoding, model pipeline.
*   `app.py`: Streamlit inference UI script.
*   `liver_cirrhosis_model.pkl`: Random Forest Model parameters.
*   `liver_scaler.pkl`: StandardScaler weights mapping.
*   `stage_label_encoder.pkl`: Class decoder logic to revert outputs to their nominal stages.
*   `Liver_Cirrhosis_Report.pdf`: Medical application summary describing class importance.
*   `requirements.txt`: Project requisite Python frameworks.
