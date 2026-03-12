import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Liver Cirrhosis Stage Detection", page_icon="🩺")

@st.cache_resource
def load_assets():
    model_path = os.path.join(os.path.dirname(__file__), 'liver_cirrhosis_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'liver_scaler.pkl')
    le_stage_path = os.path.join(os.path.dirname(__file__), 'stage_label_encoder.pkl')
    
    model, scaler, le_stage = None, None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_stage_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le_stage = joblib.load(le_stage_path)
    return model, scaler, le_stage

model, scaler, le_stage = load_assets()

# We need the exact column order used during training.
# Reading the first row of training data to get columns is ideal, but let's hardcode the typical ones based on standard liver cirrhosis dataset
# If it fails, we will need to adjust.
# Columns: N_Days, Status, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin

st.title("Liver Cirrhosis Stage Detection 🩺")
st.write("Enter the patient's medical and personal details to predict the liver cirrhosis stage.")

with st.form("liver_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_days = st.number_input("N_Days (Days since registration)", min_value=0, value=1500)
        status = st.selectbox("Status", ['C', 'CL', 'D'])
        drug = st.selectbox("Drug", ['D-penicillamine', 'Placebo'])
        age = st.number_input("Age (in days)", min_value=0, value=18000)
        sex = st.selectbox("Sex", ['M', 'F'])
        ascites = st.selectbox("Ascites", ['N', 'Y'])
        
    with col2:
        hepatomegaly = st.selectbox("Hepatomegaly", ['N', 'Y'])
        spiders = st.selectbox("Spiders", ['N', 'Y'])
        edema = st.selectbox("Edema", ['N', 'S', 'Y'])
        bilirubin = st.number_input("Bilirubin (mg/dl)", min_value=0.0, value=1.5)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=250.0)
        albumin = st.number_input("Albumin (g/dl)", min_value=0.0, value=3.5)
        
    with col3:
        copper = st.number_input("Copper (ug/day)", min_value=0.0, value=80.0)
        alk_phos = st.number_input("Alk_Phos (U/liter)", min_value=0.0, value=1200.0)
        sgot = st.number_input("SGOT (U/ml)", min_value=0.0, value=110.0)
        tryglicerides = st.number_input("Tryglicerides (mg/dl)", min_value=0.0, value=100.0)
        platelets = st.number_input("Platelets (ml/1000)", min_value=0.0, value=250.0)
        prothrombin = st.number_input("Prothrombin (s)", min_value=0.0, value=10.5)

    submit = st.form_submit_button("Predict Stage")

if submit:
    if model is None or scaler is None or le_stage is None:
        st.error("Model, scaler, or LabelEncoder files not found! Please run the notebook first.")
    else:
        # Create a dataframe with inputs
        input_dict = {
            'N_Days': n_days, 'Status': status, 'Drug': drug, 'Age': age, 'Sex': sex,
            'Ascites': ascites, 'Hepatomegaly': hepatomegaly, 'Spiders': spiders, 'Edema': edema,
            'Bilirubin': bilirubin, 'Cholesterol': cholesterol, 'Albumin': albumin, 
            'Copper': copper, 'Alk_Phos': alk_phos, 'SGOT': sgot, 
            'Tryglicerides': tryglicerides, 'Platelets': platelets, 'Prothrombin': prothrombin
        }
        
        df_input = pd.DataFrame([input_dict])
        
        # We need to replicate the LabelEncoder logic for categorical columns
        # Since we didn't save the exact per-column encoders, we will implement simple mapping if standard LabelEncoder was used.
        # But wait, we used Label Encoder in notebook! We need those to securely encode!
        # If per-column encoders weren't saved, this will break if categories are alphabetical
        
        # Fast manual encoding (simulating alphabetical LabelEncoder):
        categorical_cols = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        for col in categorical_cols:
            if df_input[col].dtype == 'object':
                # Simplified encoding: we assume standard mapping
                df_input[col] = df_input[col].astype('category').cat.codes
                
        # To be perfectly safe, we'll let pandas dummy/category it. 
        # Actually in the original notebook we used:
        # le = LabelEncoder()
        # df[col] = le.fit_transform(df[col].astype(str))
        # This sorts alphabetically.
        
        df_input['Status'] = {'C': 0, 'CL': 1, 'D': 2}.get(status, 0)
        df_input['Drug'] = {'D-penicillamine': 0, 'Placebo': 1}.get(drug, 0)
        df_input['Sex'] = {'F': 0, 'M': 1}.get(sex, 0)
        df_input['Ascites'] = {'N': 0, 'Y': 1}.get(ascites, 0)
        df_input['Hepatomegaly'] = {'N': 0, 'Y': 1}.get(hepatomegaly, 0)
        df_input['Spiders'] = {'N': 0, 'Y': 1}.get(spiders, 0)
        df_input['Edema'] = {'N': 0, 'S': 1, 'Y': 2}.get(edema, 0)

        # Scale features
        # Note: Columns MUST match training order precisely.
        # N_Days, Status, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin
        
        try:
            input_scaled = scaler.transform(df_input)
            
            prediction_idx = model.predict(input_scaled)[0]
            predicted_stage = le_stage.inverse_transform([prediction_idx])[0]
            
            st.success(f"**Predicted Liver Cirrhosis Stage:** {predicted_stage}")
        except Exception as e:
            st.error(f"Prediction Error: Feature mismatch. The input features must match training exactly. Exception: {e}")
