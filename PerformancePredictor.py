import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Student Academic Predictor", page_icon="🎓")
st.title("🎓 Student Academic Performance Predictor")

st.markdown("Fill in the details below to predict your expected **exam score** and get **personalized suggestions** to improve it!")

# User inputs
age = st.number_input("Age", 10, 25, 18)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
study_hours = st.number_input("Study Hours per Day", 0, 24)
social_media = st.number_input("Social Media Hours", 0, 24)
netflix = st.number_input("Netflix Hours", 0, 24)
part_time = st.selectbox("Part-Time Job", ["Yes", "No"])
attendance = st.slider("Attendance (%)", 0, 100, 85)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
exercise = st.number_input("Exercise Frequency (per week)", 0, 14, 3)
parent_edu = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
mental_health = st.selectbox("Mental Health Rating", ["Poor", "Average", "Good"])
extracurricular = st.selectbox("Extracurricular Participation", ["Yes", "No"])

# Encoding input
input_data = pd.DataFrame([[  
    age,
    {"Male":0, "Female":1, "Other":2}[gender],
    study_hours,
    social_media,
    netflix,
    {"Yes":1, "No":0}[part_time],
    attendance,
    sleep,
    {"Poor":0, "Average":1, "Good":2}[diet],
    exercise,
    {"High School":0, "Bachelor":1, "Master":2, "PhD":3}[parent_edu],
    {"Poor":0, "Average":1, "Good":2}[internet],
    {"Poor":0, "Average":1, "Good":2}[mental_health],
    {"Yes":1, "No":0}[extracurricular]
]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict & show suggestions
if st.button("🔍 Predict Exam Score"):
    prediction = model.predict(scaled_input)
    st.success(f"📈 Your predicted exam performance is: **{prediction[0]:.2f}%**")

    st.markdown("### 📌 Personalized Suggestions to Improve Your Score")
    
    if study_hours < 2:
        st.write("📚 *Try to study at least 2–3 hours a day consistently to boost your preparation.*")
    if social_media > 3:
        st.write("📵 *Too much social media can be distracting. Consider reducing it to stay focused.*")
    if netflix > 2:
        st.write("🎞️ *Reducing Netflix binge sessions can help you gain more productive hours.*")
    if sleep < 6:
        st.write("😴 *Sleep at least 6–8 hours daily — it's crucial for memory and focus.*")
    if attendance < 75:
        st.write("🏫 *Attend more classes to grasp concepts better — it reflects in performance.*")
    if diet == "Poor":
        st.write("🥦 *Try improving your diet — better nutrition supports better brain function.*")
    if exercise < 2:
        st.write("🏃 *Even 2–3 sessions of light exercise weekly can boost energy and focus.*")
    if mental_health == "Poor":
        st.write("🧘 *Mental health matters. Consider mindfulness, journaling, or speaking with someone.*")
    if part_time == "Yes" and study_hours < 2:
        st.write("🕒 *If you're working part-time, create a balanced schedule for studies.*")

    st.markdown("---")
    st.info("💡 *These suggestions are based on patterns in student performance data. Everyone's journey is different — take what works for you!*")
