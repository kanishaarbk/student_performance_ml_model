import streamlit as st
import numpy as np
import pickle

st.title("🎓 Student Performance Predictor")

# Model-ai load seiyidhu
with open("model (1).pkl", "rb") as f:
    model = pickle.load(f)

# User kitta irundhu input vaanguradhu[cite: 1]
hours_studied = st.number_input("Hours Studied", 1, 9, 5)
previous_scores = st.number_input("Previous Scores", 40, 99, 75)
extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
sleep_hours = st.number_input("Sleep Hours", 4, 9, 7)
sample_papers = st.number_input("Sample Papers Practiced", 0, 9, 2)

# Encoding (No -> 0, Yes -> 1)[cite: 1]
activity = 1 if extracurricular == "Yes" else 0

# Predict button matrum output
if st.button("Predict"):
    features = np.array([[hours_studied, previous_scores, activity, sleep_hours, sample_papers]])[cite: 1]
    prediction = model.predict(features)
    st.success(f"Estimated Performance Index: {prediction[0]:.2f}")
