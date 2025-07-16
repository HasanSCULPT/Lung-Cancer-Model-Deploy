# 📁 Folder Structure:
# Lung-Cancer-Model-Deploy/
# ├─ lung_cancer_app.py
# ├─ lung_cancer_pipeline.pkl
# ├─ logo.png
# ├─ feathered_bg.png   ✅ ← background image 
# ├─ feathered_bg.png
# └─ requirements.txt


# =======================================
# File: lung_cancer_app.py
# =======================================

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as st
import base64
import smtplib
import io
import plotly.express as px
from email.message import EmailMessage
from fpdf import FPDF
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.inspection import permutation_importance

# =======================================
# Lung Cancer Diagnostics App
# =======================================

BACKGROUND_IMAGE = "background.png"
LOGO_IMAGE = "logo.png"
ENABLE_EMAIL = True  # Toggle email sending
DEFAULT_LANGUAGE = "en"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "your_email@example.com"
SENDER_PASSWORD = "your_password"  # Use env variables for security!

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# =====================================
# Background Setup
# =====================================
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

if BACKGROUND_IMAGE:
    set_png_as_page_bg("background.png")

# =====================================
# Load Model & Features
# =====================================
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# =====================================
# Language Translations
# =====================================
LANG_OPTIONS = {
    "en": "English",
    "fr": "Français",
    "ru": "Русский",
    "ar": "العربية",
    "uk": "Українська"
}

translations = {
    "en": {"title": "Lung Cancer Diagnostics Centre", "subtitle": "By HasanSCULPT | DSA 2025",
           "upload_csv": "Upload your CSV data", "prediction_results": "Prediction Results"},
    "fr": {"title": "Centre de Diagnostic du Cancer du Poumon", "subtitle": "Par HasanSCULPT | DSA 2025",
           "upload_csv": "Téléchargez votre fichier CSV", "prediction_results": "Résultats de la prédiction"},
    "ru": {"title": "Центр Диагностики Рака Легких", "subtitle": "ХасанСКАЛЬПТ | DSA 2025",
           "upload_csv": "Загрузите ваш CSV файл", "prediction_results": "Результаты прогноза"},
    "ar": {"title": "مركز تشخيص سرطان الرئة", "subtitle": "بواسطة حسنSculpt | DSA 2025",
           "upload_csv": "قم بتحميل ملف CSV الخاص بك", "prediction_results": "نتائج التنبؤ"},
    "uk": {"title": "Центр Діагностики Раку Легенів", "subtitle": "ХасанСКАЛЬПТ | DSA 2025",
           "upload_csv": "Завантажте свій CSV файл", "prediction_results": "Результати прогнозу"}
}

selected_lang = st.sidebar.selectbox("🌍 Language", options=list(LANG_OPTIONS.keys()),
                                     format_func=lambda x: LANG_OPTIONS[x], index=list(LANG_OPTIONS.keys()).index(DEFAULT_LANGUAGE))
tr = translations[selected_lang]

# =====================================
# Header
# =====================================
if LOGO_IMAGE:
    st.image(LOGO_IMAGE, width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

# Sidebar Threshold
st.sidebar.subheader("🛠 Adjust Classification Threshold")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

# =====================================
# File Upload
# =====================================
uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv")

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df_input.head())

    # ✅ Data Cleaning
    required_cols = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'ALCOHOL CONSUMING', 'PEER_PRESSURE',
                     'COUGHING', 'SHORTNESS OF BREATH']
    for col in required_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input.fillna(df_input.mode().iloc[0], inplace=True)

    # Align features
    df_input = pd.get_dummies(df_input, drop_first=True)
    for col in feature_names:
        if col not in df_input:
            df_input[col] = 0
    df_input = df_input[feature_names]

    # ✅ Predictions
    probs = pipeline.predict_proba(df_input)[:, 1]
    preds = (probs > threshold).astype(int)
    df_output = df_input.copy()
    df_output["Probability"] = probs
    df_output["Prediction"] = preds
    st.write(f"### {tr['prediction_results']}")
    st.dataframe(df_output[["Probability", "Prediction"]])
    st.download_button("📥 Download Results CSV", df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

    # ✅ Automatic Threshold Suggestions
    st.write("### 🔍 Automatic Threshold Suggestions")
    fpr, tpr, thresholds = roc_curve(preds, probs)
    youden_j = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_j)]
    st.info(f"ROC-Optimal Threshold: {optimal_threshold:.2f}")

    # ✅ Histogram
    fig, ax = plt.subplots()
    ax.hist(probs, bins=10, edgecolor='k')
    ax.axvline(threshold, color='red', linestyle='--')
    st.pyplot(fig)


    

else:
    st.info("⬅️ Upload a CSV file to start prediction")

# =====================================
# Individual Prediction
# =====================================
st.write("---")
st.write("### Individual Prediction")
age = st.number_input("Age", 0, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking", [0, 1])
anxiety = st.selectbox("Anxiety", [0, 1])
alcohol = st.selectbox("Alcohol Consuming", [0, 1])
peer_pressure = st.selectbox("Peer Pressure", [0, 1])
cough = st.selectbox("Coughing", [0, 1])
short_breath = st.selectbox("Shortness of Breath", [0, 1])
symptom_score = st.slider("SYMPTOM SCORE", 0, 10, 5)
lifestyle_score = st.slider("LIFESTYLE SCORE", 0, 5, 2)

if st.button("Predict Individual"):
    row = pd.DataFrame({'AGE': [age], 'GENDER': [1 if gender == "Male" else 0],
                        'SMOKING': [smoking], 'ANXIETY': [anxiety],
                        'ALCOHOL CONSUMING': [alcohol], 'PEER_PRESSURE': [peer_pressure],
                        'COUGHING': [cough], 'SHORTNESS OF BREATH': [short_breath],
                        'SYMPTOM_SCORE': [symptom_score], 'LIFESTYLE_SCORE': [lifestyle_score]})
    for col in feature_names:
        if col not in row:
            row[col] = 0
    row = row[feature_names]

    prob = pipeline.predict_proba(row)[0][1]
    pred = int(prob > threshold)
    st.success(f"{'🛑 LUNG CANCER' if pred else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

    # Confidence Bar Chart
    fig, ax = plt.subplots()
    ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
    st.pyplot(fig)

   
    # Export PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    st.download_button(label="📥 Download PDF", data=pdf_buffer, file_name="prediction_result.pdf", mime="application/pdf")

    # Email Result (Optional)
    if ENABLE_EMAIL:
        email = st.text_input("Enter your email to receive result:")
        if st.button("Send Email") and email:
            try:
                msg = EmailMessage()
                msg["Subject"] = "Lung Cancer Prediction Result"
                msg["From"] = SENDER_EMAIL
                msg["To"] = email
                msg.set_content(f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}\nProbability: {prob:.2f}")
                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
                    smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                    smtp.send_message(msg)
                st.success("✅ Email sent successfully!")
            except:
                st.error("❌ Failed to send email.")
