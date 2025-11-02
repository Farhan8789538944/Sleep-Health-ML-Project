from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pickle
import os
import time

# ===========================
# Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="Stress Level Application",
    page_icon="💤",
    initial_sidebar_state="expanded"
)

st.title('Stress Level Prediction App 💤')
st.markdown('<span style="color:gray">This app predicts the stress level of a person based on the data provided.</span>', unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "Prediction"],
    icons=["window", "table", "cpu", "phone"],
    orientation="horizontal",
    default_index=0,
    styles={"nav-link-selected": {"background-color": "#176397"}}
)

# ===========================
# Dataset Load (safe)
# ===========================
DATA_PATHS = [
    './Datasets/cleaned-dataset.csv',
    '../Datasets/cleaned-dataset.csv',
    'Datasets/cleaned-dataset.csv'
]

df = None
for path in DATA_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        break

if df is None:
    st.error("❌ Dataset not found. Please ensure 'cleaned-dataset.csv' is in the Datasets folder.")
    st.stop()

# Drop columns safely
for col in ['Person ID', 'Sick']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Reorder columns if exist
expected_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
       'Physical Activity Level', 'BMI Category', 'Heart Rate',
       'Daily Steps', 'Sleep Disorder', 'BP High', 'BP Low', 'Stress Level']
df = df[[col for col in expected_cols if col in df.columns]]

# ===========================
# HOME TAB
# ===========================
if selected == "Home":
    with st.container():
        target_url = "https://public.tableau.com/app/profile/ramazan.erduran1816/viz/StressLevelHealth/Overview"
        image_url = "https://raw.githubusercontent.com/AshNumpy/Sleep-Health-ML-Project/main/Imgs/Homepage.png"

        st.markdown(
            f'<a href="{target_url}" target="_blank"><img src="{image_url}" alt="Homepage Image" width="100%"></a>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <h2 style="color:#176397">Project Overview</h2>
            <p style="color:#1D4665">
                This project analyzes sleep health and lifestyle data and predicts stress levels using machine learning.
            </p>
            """,
            unsafe_allow_html=True
        )

# ===========================
# DATASET TAB
# ===========================
elif selected == "Dataset":
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color='#176397', align='left', font=dict(color='white', size=12)),
        cells=dict(values=[df[col] for col in df.columns], fill_color='white', align='left', font=dict(color='#1D4665', size=12))
    )])

    fig.update_layout(height=800)
    st.subheader("Dataset Preview")
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# PREDICTION TAB
# ===========================
elif selected == "Prediction":

    st.subheader("Prediction Section")

    MODEL_PATHS = {
        "model": ["./Models/model.pkl", "./.streamlit/Models/model.pkl"],
        "gender": ["./Models/gender_encoder.pkl", "./.streamlit/Models/gender_encoder.pkl"],
        "bmi": ["./Models/bmi_category_encoder.pkl", "./.streamlit/Models/bmi_category_encoder.pkl"],
        "occupation": ["./Models/occupation_encoder.pkl", "./.streamlit/Models/occupation_encoder.pkl"],
        "sleep_disorder": ["./Models/sleep_disorder_encoder.pkl", "./.streamlit/Models/sleep_disorder_encoder.pkl"],
        "scaler": ["./Models/scaler.pkl", "./.streamlit/Models/scaler.pkl"]
    }

    def safe_load(paths, label):
        for path in paths:
            if os.path.exists(path):
                return joblib.load(path) if path.endswith('.pkl') else pickle.load(open(path, 'rb'))
        st.error(f"❌ Missing file: {label}. Please ensure it’s inside the Models folder.")
        st.stop()

    model = safe_load(MODEL_PATHS["model"], "model.pkl")
    gender_le = safe_load(MODEL_PATHS["gender"], "gender_encoder.pkl")
    bmiCategory_le = safe_load(MODEL_PATHS["bmi"], "bmi_category_encoder.pkl")
    occupation_le = safe_load(MODEL_PATHS["occupation"], "occupation_encoder.pkl")
    sleep_disorder_le = safe_load(MODEL_PATHS["sleep_disorder"], "sleep_disorder_encoder.pkl")
    scaler = safe_load(MODEL_PATHS["scaler"], "scaler.pkl")

    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ("Male", "Female"))
        Age = st.slider("Age", 18, 65, 30)
        Occupation = st.selectbox("Occupation", df['Occupation'].unique())
        SleepDuration = st.slider("Sleep Duration (Hours)", 0.0, 24.0, 7.0, 0.1)
        SleepQuality = st.slider("Sleep Quality (1–10)", 0, 10, 5, 1)
        PhysicalActivity = st.slider("Physical Activity (%)", 0.0, 100.0, 50.0, 1.0)
    with col2:
        BMI = st.selectbox("BMI Category", df['BMI Category'].unique())
        HeartRate = st.slider("Heart Rate", 60.0, 120.0, 70.0, 0.1)
        SleepDisorder = st.selectbox("Sleep Disorder", df['Sleep Disorder'].unique())
        DailySteps = st.slider("Daily Steps", 0, 10000, 5000, 1)
        BP_High = st.slider("BP High", 90.0, 180.0, 120.0, 0.1)
        BP_Low = st.slider("BP Low", 50.0, 120.0, 80.0, 0.1)

    if st.button("Predict Stress Level"):
        data = pd.DataFrame({
            'Gender': [Gender],
            'Age': [Age],
            'Occupation': [Occupation],
            'Sleep Duration': [SleepDuration],
            'Quality of Sleep': [SleepQuality],
            'Physical Activity Level': [PhysicalActivity],
            'BMI Category': [BMI],
            'Heart Rate': [HeartRate],
            'Daily Steps': [DailySteps],
            'Sleep Disorder': [SleepDisorder],
            'BP High': [BP_High],
            'BP Low': [BP_Low]
        })

        # Encoding
        data['Gender'] = gender_le.transform(data['Gender'])
        data['Occupation'] = occupation_le.transform(data['Occupation'])
        data['BMI Category'] = bmiCategory_le.transform(data['BMI Category'])
        data['Sleep Disorder'] = sleep_disorder_le.transform(data['Sleep Disorder'])

        numeric_cols = ['Age','Sleep Duration','Quality of Sleep','Physical Activity Level',
                        'Heart Rate','Daily Steps','BP High','BP Low']
        data[numeric_cols] = scaler.transform(data[numeric_cols])

        with st.spinner("Predicting..."):
            pred = model.predict(data)[0]
            time.sleep(1)
            st.success(f"✅ Predicted Stress Level: **{np.round(pred,2)} / 10**")
