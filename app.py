import streamlit as st
import xgboost as xgb 
import joblib
import numpy as np
import matplotlib.pyplot as plt
  # make sure xgboost is in requirements.txt

st.title("Student Average Score Predictor")

# Upload the model file
uploaded_model = st.file_uploader("student_score_predictor_xgb.pkl", type=["pkl"])

if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
        st.write("Debug: Model type =", type(model))
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")


    # Input fields
    math = st.number_input("Math Score", 0, 100)
    history = st.number_input("History Score", 0, 100)
    science = st.number_input("Science Score", 0, 100)

    if st.button("Predict Average"):
        # Prepare input
        features = np.array([[math, history, science]])

        # Run prediction
        prediction = model.predict(features)[0]
        st.success(f"🎯 Predicted Average Score: {prediction:.2f}")

        # Visualize input
        subjects = ["Math", "History", "Science"]
        scores = [math, history, science]

        fig, ax = plt.subplots()
        ax.bar(subjects, scores, color="skyblue")
        ax.set_ylim(0, 100)
        ax.set_title("Input Scores")
        st.pyplot(fig)
else:
    st.info("student_score_predictor_xgb.pkl")

