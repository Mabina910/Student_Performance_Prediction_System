import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


model = joblib.load("best_model.pkl")

st.title("student Exam Score Predictor")

study_hours = st.slider("Study hours per day", 0.0, 12.0,2.0)
attendance = st.slider("Attendance Percentage",0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)",1,10,5)
sleep_hours = st.slider("sleep Hours per night",0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part time Job",["No","Yes"])

ptj_encoded = 1 if part_time_job =="yes" else 0

if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours,attendance,mental_health,sleep_hours,ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max (0,min(100,prediction))

    st.success(f"Predict Exam Score : {prediction:.2f}")