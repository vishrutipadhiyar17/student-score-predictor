import streamlit as st
import joblib

# Load the model
model = joblib.load('score_predictor_model.pkl')

# Page config
st.set_page_config(page_title="Exam Score Predictor", page_icon="📘", layout="centered")

# App title and description
st.title("📘 Student Score Predictor")
st.markdown("Enter the number of hours you studied to predict your exam score using a machine learning model.")

# Input
hours = st.number_input("📚 Hours Studied", min_value=0.0, max_value=24.0, step=0.5)

# Predict button
if st.button("🎯 Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: **{prediction[0]:.2f}** out of 100")

# Footer
st.markdown("---")
st.markdown("💡 Built with Python, Scikit-learn, and Streamlit")
st.markdown("📍 Project by [Your Name]")

