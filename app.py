import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("Student Average Score Predictor")


# Upload the model file
uploaded_model = st.file_uploader("student_score_predictor_xgb.pkl", type=["pkl"])

if uploaded_model is not None:
    # Load the model
    model = joblib.load(uploaded_model)

    st.success("✅ Model loaded successfully!")


# Input fields
    math = st.number_input("Math Score", 0, 100)
    history = st.number_input("History Score", 0, 100)
    physics = st.number_input("Physics Score", 0, 100)
    chemistry = st.number_input("Chemistry Score", 0, 100)
    biology = st.number_input("Biology Score", 0, 100)
    english = st.number_input("English Score", 0, 100)
    geography = st.number_input("Geography Score", 0, 100)

    if st.button("Predict Average"):
       X_new = np.array([[math, history, physics, chemistry, biology, english, geography]])
       prediction = model.predict(X_new)
       st.success(f"Predicted Average Score: {prediction[0]:.2f}")
 
       subjects = ["Math","History","Physics","Chemistry","Biology","English","Geography"]
       scores = [math, history, physics, chemistry, biology, english, geography]
       plt.bar(subjects, scores, color='skyblue')
       plt.ylabel("Score")
       plt.title("Student Subject Scores")
       st.pyplot(plt)

else:
    st.info("student_score_predictor_xgb.pkl")
