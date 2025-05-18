# 🎓 Student Academic Performance Predictor

This project is a Streamlit web application that predicts a student's **exam performance** based on various lifestyle, health, and study-related factors. It uses a **Random Forest Regression model** trained on the Kaggle dataset:  
👉 [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)

---

## 📌 Features

- Interactive web app built with **Streamlit**
- Trained using **Random Forest Regression**
- Takes inputs like study hours, sleep, diet, mental health, and more
- Outputs a predicted **exam score**
- Provides **personalized suggestions** to help students improve their performance

---

## 🗂️ Project Structure
├── rf_model.pkl # Trained Random Forest model
├── scaler.pkl # StandardScaler object
├── student_habits_performance.csv # Raw dataset from Kaggle
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
└── README.md # This file
