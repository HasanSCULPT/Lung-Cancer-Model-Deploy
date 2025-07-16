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

# ----------------------------
# ✅ Streamlit Configuration
# ----------------------------


st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ----------------------------
# ✅ Background Image
# ----------------------------
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg("background.png")

# ----------------------------
# ✅ Load Model & Features
# ----------------------------
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# ----------------------------
# ✅ Language Translations
# ----------------------------
def get_translation(language):
    translations = {
        "en": {
            "title": "Lung Cancer Diagnostics Centre",
            "subtitle": "By HasanSCULPT | DSA 2025",
            "upload_csv": "Upload your CSV data",
            "prediction_results": "Prediction Results",
            "download_csv": "Download Results CSV",
            "individual_entry": "Enter patient data below for prediction",
            "about_title": "📘 About Us",
            "about_desc": "This app is developed to assist in preliminary lung cancer risk prediction using machine learning.",
            "contact_title": "📧 Contact Us",
            "terms_title": "📜 Terms & Conditions",
            "terms_text": "This tool is for educational purposes only and does not replace professional medical advice."
        }
    }
    return translations.get(language, translations["en"])

# ----------------------------
# ✅ UI Setup
# ----------------------------
LANG_OPTIONS = {"en": "English"}
selected_lang = st.sidebar.selectbox("🌍 Select Language", options=list(LANG_OPTIONS.keys()), format_func=lambda x: LANG_OPTIONS[x])
tr = get_translation(selected_lang)

st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

page = st.sidebar.selectbox("Navigate", ["Prediction", "About", "Contact", "Terms"])

# ----------------------------
# ✅ Page Routing
# ----------------------------
if page == "About":
    st.title(tr['about_title'])
    st.write(tr['about_desc'])

elif page == "Contact":
    st.title(tr['contact_title'])
    st.write("Email: support@lungdiagnosis.ai")

elif page == "Terms":
    st.title(tr['terms_title'])
    st.write(tr['terms_text'])

# ----------------------------
# ✅ Prediction Page
# ----------------------------
if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv")

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_input.head())

        # ✅ Data Cleaning
        required_cols = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'ALCOHOL CONSUMING', 'PEER_PRESSURE', 'COUGHING', 'SHORTNESS OF BREATH']
        for col in required_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        if df_input.isnull().sum().sum() > 0:
            for col in df_input.columns:
                if df_input[col].dtype in ['int64', 'float64']:
                    df_input[col].fillna(df_input[col].mean(), inplace=True)
                else:
                    df_input[col].fillna(df_input[col].mode()[0], inplace=True)

        # ✅ Feature Alignment
        for col in feature_names:
            if col not in df_input:
                df_input[col] = 0
        df_input = df_input[feature_names]

        # ✅ Automatic Threshold Suggestions
        st.write("### 🔍 Automatic Threshold Suggestions")
        proba_temp = pipeline.predict_proba(df_input)[:, 1]
        suggested_recall_thresh = None
        suggested_roc_thresh = None

        # ✅ Option 1: Recall
        if "LUNG_CANCER" in df_input.columns:
            y_true = df_input["LUNG_CANCER"]
            best_recall, best_thresh = 0, threshold
            for t in np.arange(0.1, 0.9, 0.01):
                y_pred_temp = (proba_temp > t).astype(int)
                recall = recall_score(y_true, y_pred_temp)
                if recall > best_recall:
                    best_recall, best_thresh = recall, t
            st.success(f"✅ Suggested Threshold (Max Recall): {best_thresh:.2f} (Recall: {best_recall:.2f})")

        # ✅ Option 2: ROC
        fpr, tpr, thresholds = roc_curve((proba_temp > 0.5).astype(int), proba_temp)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        st.info(f"🔍 Suggested Threshold (ROC Optimal): {optimal_threshold:.2f}")

        # ✅ Predictions
        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)
        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction
        st.write(f"### {tr['prediction_results']}")
        st.dataframe(df_output[["Probability", "Prediction"]])
        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        # ✅ Histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        st.pyplot(fig)

    else:
        st.info("⬅️ Upload a CSV file to start prediction")

    # ✅ Individual Prediction
    st.write("---")
    st.write(f"### {tr['individual_entry']}")
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
        row = pd.DataFrame({
            'AGE': [age],
            'GENDER': [1 if gender == "Male" else 0],
            'SMOKING': [smoking],
            'ANXIETY': [anxiety],
            'ALCOHOL CONSUMING': [alcohol],
            'PEER_PRESSURE': [peer_pressure],
            'COUGHING': [cough],
            'SHORTNESS OF BREATH': [short_breath],
            'SYMPTOM_SCORE': [symptom_score],
            'LIFESTYLE_SCORE': [lifestyle_score]
        })

        for col in feature_names:
            if col not in row:
                row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold)
        st.success(f"{'🛑 LUNG CANCER' if pred == 1 else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

        # ✅ Confidence Bar
        fig, ax = plt.subplots()
        bars = ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center')
        st.pyplot(fig)

        # ✅ SHAP Explanation
        explainer = shap.KernelExplainer(pipeline.predict_proba, np.zeros((1, len(feature_names))))
        shap_values = explainer.shap_values(row)
        st.write("### SHAP Explanation")
        shap.force_plot(explainer.expected_value[1], shap_values[1], row, matplotlib=True)
        st.pyplot()

        # ✅ Export CSV + PDF
        result_df = pd.DataFrame({"Prediction": ["Lung Cancer" if pred else "No Lung Cancer"], "Probability": [prob]})
        st.download_button(label="📥 Download CSV", data=result_df.to_csv(index=False), file_name="prediction_result.csv", mime="text/csv")

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
