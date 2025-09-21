import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.title("Student Average Score Predictor")

# Upload the model file
uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

if uploaded_model is not None:
    model = joblib.load(student_score_predictor_xgb.pkl)

    # Input fields
    math = st.number_input("Math Score", 0, 100)
    history = st.number_input("History Score", 0, 100)
    science = st.number_input("Science Score", 0, 100)

    if st.button("Predict Average"):
        features = np.array([[math, history, science]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Average Score: {prediction:.2f}")

        # Optional: visualize input scores
        subjects = ["Math", "History", "Science"]
        scores = [math, history, science]

        fig, ax = plt.subplots()
        ax.bar(subjects, scores, color="skyblue")
        ax.set_ylim(0, 100)
        ax.set_title("Input Scores")
        st.pyplot(fig)
else:
    st.info("student_score_predictor_xgb.pkl")

