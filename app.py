import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("🎓 Student Performance Predictor")

# Model-ai load seiyidhu
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Notebook-il ulla StandardScaler values (Mean matrum Scale)
mean = np.array([5.224, 69.445, 0.496, 6.485, 4.583])
scale = np.array([2.565, 17.343, 0.500, 1.715, 2.787])

# User input fields
hours_studied = st.number_input("Hours Studied", 1, 9, 5)
previous_scores = st.number_input("Previous Scores", 40, 99, 75)
extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
sleep_hours = st.number_input("Sleep Hours", 4, 9, 7)
sample_papers = st.number_input("Sample Papers Practiced", 0, 9, 2)

# Encoding (No -> 0, Yes -> 1)
activity = 1 if extracurricular == "Yes" else 0

if st.button("Predict"):
    # Raw inputs array
    raw_features = np.array([[hours_studied, previous_scores, activity, sleep_hours, sample_papers]])
    
    # Manual Standard Scaling
    scaled_features = (raw_features - mean) / scale
    
    # Prediction
    prediction = model.predict(scaled_features)
    st.success(f"Estimated Performance Index: {prediction[0]:.2f}")
