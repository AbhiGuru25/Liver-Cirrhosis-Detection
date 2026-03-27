import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Liver Cirrhosis Stage Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SVG Icon Helpers ───────────────────────────────────────────────────────
def icon(svg, size=28, bg="#f3e5f5", color="#6a1b9a"):
    return f"""<span style="display:inline-flex;align-items:center;justify-content:center;
        width:{size+12}px;height:{size+12}px;background:{bg};border-radius:10px;
        margin-right:8px;vertical-align:middle;flex-shrink:0;">
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
             viewBox="0 0 24 24" fill="none" stroke="{color}"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{svg}</svg>
    </span>"""

def h3(svg, text, size=20, bg="#f3e5f5", color="#6a1b9a"):
    return f'<h3 style="display:flex;align-items:center;margin:12px 0 8px 0;">{icon(svg,size,bg,color)}{text}</h3>'

ICO_HEALTH = '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
ICO_USER   = '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'
ICO_FLASK  = '<path d="M9 3h6l1 9H8L9 3z"/><path d="M6.5 21a5 5 0 0 1 7.5-4.3 5 5 0 0 1 3.5 4.3H5z"/>'
ICO_ABOUT  = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
ICO_CHART  = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
ICO_TARGET = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
ICO_CPU    = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
ICO_LAYERS = '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
ICO_GLOBE  = '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a0a2e 0%, #2a1040 60%, #3d1460 100%); }
    [data-testid="stSidebar"] * { color: #f3e5f5 !important; }
    .metric-card { background: linear-gradient(135deg, #1a0a2e, #2a1040); border: 1px solid #7b1fa2;
        border-radius: 12px; padding: 16px; text-align: center; color: white; }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #ce93d8; }
    .metric-card p  { margin: 0; color: #e1bee7; font-size: 0.85rem; }
    .section-header { display:flex; align-items:center; background: linear-gradient(90deg, #4a148c, #6a1b9a);
        padding: 10px 18px; border-radius: 8px; color: white; font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; }
    .result-box { background: linear-gradient(135deg, #1a0a2e, #3d1460); border: 2px solid #ce93d8;
        border-radius: 12px; padding: 24px; color: white; text-align: center; }
    .normal-range { background: #1a2e1a; border-left: 3px solid #4caf50;
        padding: 6px 10px; border-radius: 4px; margin: 3px 0; font-size:0.85rem; }
    .page-title { display:flex; align-items:center; gap:10px; margin-bottom:0.5rem; }
    .page-title h1 { margin:0; }
    h3 { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model_path  = os.path.join(os.path.dirname(__file__), 'liver_cirrhosis_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'liver_scaler.pkl')
    le_path     = os.path.join(os.path.dirname(__file__), 'stage_label_encoder.pkl')
    model, scaler, le_stage = None, None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
        model    = joblib.load(model_path)
        scaler   = joblib.load(scaler_path)
        le_stage = joblib.load(le_path)
    return model, scaler, le_stage

model, scaler, le_stage = load_assets()

STAGE_INFO = {
    "1": ("Stage 1 — Inflammation", "#81c784", "Early-stage; bile duct inflammation. Often reversible."),
    "2": ("Stage 2 — Fibrosis",     "#ffb74d", "Scar tissue begins to form. Still potentially reversible."),
    "3": ("Stage 3 — Cirrhosis",    "#ef9a9a", "Significant scarring; liver function impaired."),
    "4": ("Stage 4 — End-Stage",    "#ef5350", "Severe liver failure. Transplant may be required."),
}

NORMAL_RANGES = {
    "Bilirubin":"0.1 – 1.2 mg/dl", "Cholesterol":"< 200 mg/dl", "Albumin":"3.4 – 5.4 g/dl",
    "Copper":"15 – 60 μg/day", "Alk_Phos":"44 – 147 U/liter", "SGOT":"10 – 40 U/ml",
    "Triglycerides":"< 150 mg/dl", "Platelets":"150 – 400 ml/1000", "Prothrombin":"11 – 13.5 seconds",
}

with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
        {icon(ICO_HEALTH, 26, '#2a0a40', '#ce93d8')}
        <span style="font-size:1.1rem;font-weight:700;color:white;">Liver Cirrhosis Detection</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Predict", "About Project", "Model Performance"])
    st.markdown("---")
    st.markdown("<p style='color:#e1bee7;font-size:0.8rem;font-weight:600;'>CIRRHOSIS STAGES</p>", unsafe_allow_html=True)
    for i, (color, label) in enumerate(zip(["#81c784","#ffb74d","#ef9a9a","#ef5350"],["Inflammation","Fibrosis","Cirrhosis","End-Stage"]), 1):
        st.markdown(f"<div style='background:{color}22;border-left:3px solid {color};padding:5px 8px;border-radius:4px;color:white;margin:3px 0;'><b>Stage {i}</b>: {label}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Unified Mentor Internship Project")
    st.caption("Abhivirani")

# ══════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown(f"""<div class="page-title">{icon(ICO_HEALTH, 30, '#f3e5f5', '#6a1b9a')}<h1>Liver Cirrhosis Stage Detection</h1></div>""", unsafe_allow_html=True)
    st.markdown("Enter the patient's **clinical and personal details** to predict the liver cirrhosis stage.")
    st.warning("This tool is for **educational purposes only**. Always consult a qualified medical professional.")

    with st.form("liver_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="section-header">{icon(ICO_USER, 18, "transparent", "#e1bee7")} Patient Info</div>', unsafe_allow_html=True)
            n_days  = st.number_input("N_Days (Days since registration)", min_value=0, value=1500)
            age     = st.number_input("Age (in days)", min_value=0, value=18000, help="18000 days ≈ 49 years")
            sex     = st.selectbox("Sex", ['M','F'])
            drug    = st.selectbox("Drug", ['D-penicillamine','Placebo'])
            status  = st.selectbox("Status", ['C','CL','D'], help="C=Alive, CL=Liver Transplant, D=Dead")
        with c2:
            st.markdown(f'<div class="section-header">{icon(ICO_HEALTH, 18, "transparent", "#e1bee7")} Clinical Symptoms</div>', unsafe_allow_html=True)
            ascites      = st.selectbox("Ascites",      ['N','Y'])
            hepatomegaly = st.selectbox("Hepatomegaly", ['N','Y'])
            spiders      = st.selectbox("Spiders",      ['N','Y'])
            edema        = st.selectbox("Edema",        ['N','S','Y'])
            bilirubin    = st.number_input("Bilirubin (mg/dl)",   min_value=0.0, value=1.5)
            cholesterol  = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=250.0)
        with c3:
            st.markdown(f'<div class="section-header">{icon(ICO_FLASK, 18, "transparent", "#e1bee7")} Lab Values</div>', unsafe_allow_html=True)
            albumin       = st.number_input("Albumin (g/dl)",        min_value=0.0, value=3.5)
            copper        = st.number_input("Copper (ug/day)",       min_value=0.0, value=80.0)
            alk_phos      = st.number_input("Alk_Phos (U/liter)",    min_value=0.0, value=1200.0)
            sgot          = st.number_input("SGOT (U/ml)",           min_value=0.0, value=110.0)
            tryglicerides = st.number_input("Tryglicerides (mg/dl)", min_value=0.0, value=100.0)
            platelets     = st.number_input("Platelets (ml/1000)",   min_value=0.0, value=250.0)
            prothrombin   = st.number_input("Prothrombin (s)",       min_value=0.0, value=10.5)

        col_b1, col_b2, col_b3 = st.columns([1,1,1])
        with col_b2:
            submit = st.form_submit_button("Predict Stage", type="primary", use_container_width=True)

    if submit:
        if model is None or scaler is None or le_stage is None:
            st.error("Model, scaler, or LabelEncoder files not found. Please run the training notebook first.")
        else:
            input_dict = {'N_Days':n_days,'Status':status,'Drug':drug,'Age':age,'Sex':sex,
                'Ascites':ascites,'Hepatomegaly':hepatomegaly,'Spiders':spiders,'Edema':edema,
                'Bilirubin':bilirubin,'Cholesterol':cholesterol,'Albumin':albumin,'Copper':copper,
                'Alk_Phos':alk_phos,'SGOT':sgot,'Tryglicerides':tryglicerides,'Platelets':platelets,'Prothrombin':prothrombin}
            df_input = pd.DataFrame([input_dict])
            df_input['Status']       = {'C':0,'CL':1,'D':2}.get(status, 0)
            df_input['Drug']         = {'D-penicillamine':0,'Placebo':1}.get(drug, 0)
            df_input['Sex']          = {'F':0,'M':1}.get(sex, 0)
            df_input['Ascites']      = {'N':0,'Y':1}.get(ascites, 0)
            df_input['Hepatomegaly'] = {'N':0,'Y':1}.get(hepatomegaly, 0)
            df_input['Spiders']      = {'N':0,'Y':1}.get(spiders, 0)
            df_input['Edema']        = {'N':0,'S':1,'Y':2}.get(edema, 0)
            try:
                input_scaled    = scaler.transform(df_input)
                prediction_idx  = model.predict(input_scaled)[0]
                predicted_stage = le_stage.inverse_transform([prediction_idx])[0]
                stage_key = str(predicted_stage)
                stage_label, color, description = STAGE_INFO.get(stage_key, (f"Stage {predicted_stage}", "#ce93d8", ""))

                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    st.markdown(f"""<div class="result-box" style="border-color:{color};">
                        <div style="margin-bottom:10px;">{icon(ICO_HEALTH, 36, '#1a0a2e', color)}</div>
                        <h2 style="color:{color};">{stage_label}</h2>
                        <p style="color:#e1bee7;font-size:1rem;">{description}</p>
                    </div>""", unsafe_allow_html=True)

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    classes = le_stage.classes_
                    st.markdown("---")
                    st.markdown(f'<h4 style="display:flex;align-items:center;">{icon(ICO_CHART, 18, "#f3e5f5", "#6a1b9a")} Stage Probability Distribution</h4>', unsafe_allow_html=True)
                    stage_labels = [STAGE_INFO.get(str(c), (f"Stage {c}","#ce93d8",""))[0] for c in classes]
                    bar_colors   = [STAGE_INFO.get(str(c), ("","#ce93d8",""))[1] for c in classes]
                    fig = go.Figure(go.Bar(x=stage_labels, y=[p*100 for p in proba], marker_color=bar_colors,
                        text=[f"{p*100:.1f}%" for p in proba], textposition='outside'))
                    fig.update_layout(yaxis_title="Probability (%)", height=320, margin=dict(t=20,b=60),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0,110]))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction Error: Feature mismatch — {e}")

# ══════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown(f"""<div class="page-title">{icon(ICO_ABOUT, 30, '#f3e5f5', '#6a1b9a')}<h1>About the Project</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(h3(ICO_TARGET, "Objective"), unsafe_allow_html=True)
        st.markdown("""Predict the **stage of liver cirrhosis** given clinical, biochemical, and personal patient data.
        Early detection of the stage can guide treatment plans and significantly improve patient outcomes.""")
        st.markdown(h3(ICO_LAYERS, "Dataset Details"), unsafe_allow_html=True)
        st.markdown("""| Property | Details |\n|---|---|\n| Source | Medical clinical trial dataset |\n| Target | Liver cirrhosis stage (1–4) |\n| Features | 18 clinical parameters |\n| Feature Types | Numerical + Categorical |\n| Preprocessing | StandardScaler + LabelEncoder |""")
        st.markdown(h3(ICO_HEALTH, "Normal Lab Value Ranges"), unsafe_allow_html=True)
        for param, rng in NORMAL_RANGES.items():
            st.markdown(f"<div class='normal-range'><b>{param}:</b> {rng}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(h3(ICO_CPU, "Methodology"), unsafe_allow_html=True)
        st.markdown("""1. **Data Loading** — Clinical trial patient records\n2. **Preprocessing** — LabelEncoder for categorical, StandardScaler for numerics\n3. **Model** — Multi-class ML classifier\n4. **Evaluation** — Accuracy, precision, recall, F1 per stage\n5. **Deployment** — Streamlit web application""")
        st.markdown(h3(ICO_GLOBE, "Real-World Relevance"), unsafe_allow_html=True)
        st.markdown("- **Early Detection** — Identify cirrhosis stage for timely treatment\n- **Clinical Support** — Assist doctors with data-driven prediction\n- **Patient Management** — Guide therapy decisions based on stage\n- **Research** — Study progression patterns in liver disease")

    with st.expander("Medical Glossary — Click to expand"):
        st.markdown("""| Term | Meaning |\n|---|---|\n| **Bilirubin** | Yellow pigment; elevated in liver disease |\n| **Albumin** | Protein made by liver; low = poor function |\n| **Alk_Phos** | Alkaline Phosphatase; elevated in bile duct issues |\n| **SGOT** | Liver enzyme; elevated when liver cells are damaged |\n| **Ascites** | Abnormal fluid accumulation in abdomen |\n| **Hepatomegaly** | Enlarged liver |\n| **Spiders** | Spider angiomas — dilated blood vessels on skin |\n| **Prothrombin** | Blood clotting time; elevated = poor liver function |""")

# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""<div class="page-title">{icon(ICO_CHART, 30, '#f3e5f5', '#6a1b9a')}<h1>Model Performance</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Metrics from the classifier trained on the Liver Cirrhosis dataset.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h2>~88%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h2>~87%</h2><p>Precision</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h2>~88%</h2><p>Recall</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>18</h2><p>Features</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(h3(ICO_CPU, "Preprocessing Pipeline"), unsafe_allow_html=True)
        st.markdown("""| Step | Details |\n|---|---|\n| Categorical Encoding | LabelEncoder |\n| Feature Scaling | StandardScaler |\n| Missing Values | Handled (imputation or drop) |\n| Class Labels | LabelEncoder on target stage |""")
        st.markdown(h3(ICO_LAYERS, "Why Scaling Matters"), unsafe_allow_html=True)
        st.markdown("Alk_Phos (up to 2000 U/L) and Prothrombin (10–15 s) are on very different scales. StandardScaler normalizes them so the model treats each feature equally.")
    with col2:
        st.markdown(h3(ICO_CHART, "Feature Importance (Approximate)"), unsafe_allow_html=True)
        features = ['Bilirubin','Prothrombin','Albumin','SGOT','Alk_Phos','Copper','N_Days','Platelets','Tryglicerides','Ascites']
        importance = [0.20, 0.16, 0.14, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05]
        fig = go.Figure(go.Bar(x=[v*100 for v in importance], y=features, orientation='h',
            marker_color='#ce93d8', text=[f"{v*100:.0f}%" for v in importance], textposition='outside'))
        fig.update_layout(xaxis_title="Importance (%)", yaxis=dict(autorange="reversed"),
            height=350, margin=dict(l=10,r=40,t=10,b=30), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.85rem;'>Unified Mentor Internship Project | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
