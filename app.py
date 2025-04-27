# app.py
import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
model_path = 'models/heart_disease_model.pkl'

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please train the model first.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# App title
st.title("💓 Personal Health Dashboard - Heart Disease Prediction")

st.markdown("""
Welcome to your personal health dashboard!  
Fill in your details below to predict your risk of heart disease.  
""")

# User input form
st.header("📝 Enter Your Health Information")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
rest_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
num_major_vessels = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3, value=0)
thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode the inputs manually (must match training encoding!)
sex = 1 if sex == "Male" else 0
chest_pain_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
chest_pain = chest_pain_mapping[chest_pain]
fasting_bs = 1 if fasting_bs == "Yes" else 0
rest_ecg_mapping = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
rest_ecg = rest_ecg_mapping[rest_ecg]
exercise_angina = 1 if exercise_angina == "Yes" else 0
st_slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
st_slope = st_slope_mapping[st_slope]
thalassemia_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 0}
thalassemia = thalassemia_mapping[thalassemia]

# Create input DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [rest_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope],
    'NumMajorVessels': [num_major_vessels],
    'Thalassemia': [thalassemia]
})

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease. Please consult a doctor!")
        
        st.info("""
        **Health Tips:**
        - Exercise regularly (walk, yoga, cardio).
        - Maintain a healthy weight.
        - Eat a heart-healthy diet (fruits, vegetables, low-fat).
        - Avoid smoking and limit alcohol.
        - Manage stress (meditation, breathing exercises).
        """)
    else:
        st.success("✅ Low Risk of Heart Disease. Keep up the good lifestyle!")

# Footer
st.markdown("""
---

""")

