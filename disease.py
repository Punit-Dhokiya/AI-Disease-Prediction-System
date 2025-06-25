# modern_disease_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------ Custom Styling ------------------

st.set_page_config(page_title="🧠 AI Health Assistant", layout="wide")
st.markdown("""
    <style>
        .title { text-align: center; font-size: 48px; font-weight: bold; color: #004e89; }
        .subtitle { text-align: center; font-size: 20px; color: #318fb5; margin-bottom: 20px; }
        .section { font-size: 22px; color: #00587a; font-weight: bold; margin-top: 30px; }
        .summary-card {
            background: linear-gradient(145deg, #f8fbff, #e2f1ff);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-top: 40px;
        }
        .summary-card h2 {
            color: #05445e;
            font-size: 28px;
            margin-bottom: 20px;
        }
        .summary-item {
            margin-bottom: 12px;
            font-size: 18px;
            color: #1c1c1c;
        }
        .summary-label {
            font-weight: bold;
            color: #00334e;
        }
        .highlight {
            color: #ff6f61;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Model & Data ------------------


def train_model():
    df = pd.read_csv("Training.csv")
    label_column = 'prognosis' if 'prognosis' in df.columns else 'Disease'
    X = df.drop(columns=[label_column])
    y = df[label_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

    joblib.dump(model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(X.columns.tolist(), "symptoms.pkl")

    return accuracy


def predict_disease(selected_symptoms):
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    symptoms = joblib.load("symptoms.pkl")

    input_data = [1 if s in selected_symptoms else 0 for s in symptoms]
    input_array = np.array([input_data])

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array).max() * 100
    predicted_disease = le.inverse_transform([prediction])[0]

    return predicted_disease, probability


def ensure_model():
    if not (os.path.exists("model.pkl") and os.path.exists("label_encoder.pkl") and os.path.exists("symptoms.pkl")):
        st.info("🔁 Training model for the first time...")
        acc = train_model()
        st.success(f"✅ Model trained successfully with {acc:.2f}% accuracy")

# ------------------ Database Logging ------------------


def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
                 id INTEGER PRIMARY KEY,
                 name TEXT, age INTEGER, gender TEXT, 
                 symptoms TEXT, prediction TEXT, confidence REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn


def log_prediction(conn, name, age, gender, symptoms, prediction, confidence):
    c = conn.cursor()
    c.execute("INSERT INTO logs (name, age, gender, symptoms, prediction, confidence) VALUES (?, ?, ?, ?, ?, ?)",
              (name, age, gender, ', '.join(symptoms), prediction, confidence))
    conn.commit()


def load_logs():
    conn = sqlite3.connect("predictions.db")
    return pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)

# ------------------ App UI ------------------

ensure_model()
conn = init_db()
symptoms = joblib.load("symptoms.pkl")

# Add additional symptoms manually (if not in dataset)
additional_symptoms = [
    "fatigue", "skin_rash", "fever", "headache", "chills", "nausea", "vomiting", 
    "abdominal_pain", "diarrhoea", "joint_pain", "shortness_of_breath"
]
for s in additional_symptoms:
    if s not in symptoms:
        symptoms.append(s)

st.markdown("<div class='title'>🌐 AI Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Symptom Analyzer with Database & Smart UI</div>", unsafe_allow_html=True)

# User Info
st.markdown("<div class='section'>🧑 User Information</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
name = col1.text_input("👋 Your Name")
age = col2.number_input("🎂 Age", min_value=1, max_value=120, step=1)
gender = col3.selectbox("⚧ Gender", ["Male", "Female", "Other"])

# Symptom Selector
st.markdown("<div class='section'>🩺 Select Your Symptoms</div>", unsafe_allow_html=True)
selected_symptoms = st.multiselect("🔎 Choose symptoms (type to search):", options=sorted(symptoms))
custom_note = st.text_area("📝 Add any additional symptoms/notes")

if st.button("🚀 Diagnose Me"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        result, confidence = predict_disease(selected_symptoms)
        log_prediction(conn, name, age, gender, selected_symptoms, result, confidence)

        st.markdown(f"""
        <div class='summary-card'>
            <h2>📄 Summary Report</h2>
            <div class='summary-item'><span class='summary-label'>👤 Name:</span> {name or 'N/A'}</div>
            <div class='summary-item'><span class='summary-label'>🎂 Age:</span> {age}</div>
            <div class='summary-item'><span class='summary-label'>⚧ Gender:</span> {gender}</div>
            <div class='summary-item'><span class='summary-label'>🩹 Symptoms:</span> {', '.join(selected_symptoms)}</div>
            {'<div class="summary-item"><span class="summary-label">📝 Notes:</span> ' + custom_note + '</div>' if custom_note else ''}
            <div class='summary-item'><span class='summary-label'>🧞 Predicted Disease:</span> <span class='highlight'>{result}</span></div>
            <div class='summary-item'><span class='summary-label'>📊 Confidence:</span> <span class='highlight'>{confidence:.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section'>📊 Diagnostic Trends & Visual Insights</div>", unsafe_allow_html=True)
        logs_df = load_logs()

        # Bar chart for top diseases
        top_diseases = logs_df['prediction'].value_counts().head(5)
        fig1, ax1 = plt.subplots()
        top_diseases.plot(kind='barh', color="#318fb5", ax=ax1)
        ax1.set_title("Top 5 Predicted Diseases", color='#004e89')
        ax1.set_xlabel("Count", color='#004e89')
        ax1.tick_params(axis='x', colors='#004e89')
        ax1.tick_params(axis='y', colors='#004e89')
        st.pyplot(fig1)

        # Pie chart of gender distribution
        fig2, ax2 = plt.subplots()
        logs_df['gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                                  colors=['#00bcd4','#ffb347','#d3d3d3'], ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Gender Distribution", color='#004e89')
        st.pyplot(fig2)

        # Line chart of predictions over time
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        timeline_df = logs_df.groupby(logs_df['timestamp'].dt.date).size()
        fig3, ax3 = plt.subplots()
        timeline_df.plot(kind='line', marker='o', color="#00b894", ax=ax3)
        ax3.set_title("Prediction Trends Over Time", color='#004e89')
        ax3.set_ylabel("Number of Diagnoses", color='#004e89')
        ax3.set_xlabel("Date", color='#004e89')
        ax3.tick_params(axis='x', colors='#004e89')
        ax3.tick_params(axis='y', colors='#004e89')
        st.pyplot(fig3)