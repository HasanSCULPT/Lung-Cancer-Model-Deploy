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
from sklearn.inspection import permutation_importance

# =======================================
# Lung Cancer Diagnostics App
# =======================================


# -------------------------------
# ✅ Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ✅ Background Image
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
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

set_png_as_page_bg("feathered_bg.png")

# -------------------------------
# ✅ Language Translation
# -------------------------------
def get_translation(language):
    translations = {
        "en": {
            "title": "Lung Cancer Diagnostics Centre",
            "subtitle": "By HasanSCULPT | DSA 2025",
            "upload_csv": "Upload your CSV data",
            "prediction_results": "Prediction Results",
            "download_csv": "Download Results CSV",
            "export": "Export Result",
            "download_csv_single": "Download CSV",
            "download_pdf": "Download PDF",
            "enter_email": "Enter your email address to receive results",
            "send_email": "Send Email",
            "email_success": "✅ Email sent successfully!",
            "email_fail": "❌ Failed to send email. Check configuration.",
            "language_select": "🌍 Select Language",
            "sidebar_title": "Navigate",
            "individual_entry": "Enter your medical/patient info below to predict Lung Cancer risk",
            "about_title": "📘 About Us",
            "about_desc": "This app predicts preliminary lung cancer risk using ensemble ML models.",
            "contact_title": "📧 Contact Us",
            "terms_title": "📜 Terms & Conditions",
            "terms_text": "This tool is for educational and diagnostic support only."
        }
    }
    return translations.get(language, translations["en"])

LANG_OPTIONS = {"en": "English"}
selected_lang = st.sidebar.selectbox("🌍 Select Language", options=list(LANG_OPTIONS.keys()), format_func=lambda x: LANG_OPTIONS[x])
tr = get_translation(selected_lang)

# -------------------------------
# ✅ App Header
# -------------------------------
st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

page = st.sidebar.selectbox(tr['sidebar_title'], ["Prediction", "About", "Contact", "Terms"], key="page")

# -------------------------------
# ✅ Load Model & Features
# -------------------------------
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# -------------------------------
# ✅ Helper Functions
# -------------------------------
def clean_uploaded_data(df):
    """Validate columns and fill missing values."""
    missing_cols = [col for col in feature_names if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    df = df[feature_names]
    df = df.fillna(0)
    return df

def suggest_optimal_threshold(model, X, y):
    """Find threshold maximizing sensitivity."""
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def generate_pdf(prediction, prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER 🛑' if prediction == 1 else 'NO LUNG CANCER ✅'}", ln=True)
    pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# -------------------------------
# ✅ Page Routing
# -------------------------------
if page == "About":
    st.title(tr['about_title'])
    st.write(tr['about_desc'])
elif page == "Contact":
    st.title(tr['contact_title'])
    st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.title(tr['terms_title'])
    st.write(tr['terms_text'])

# -------------------------------
# ✅ Prediction Page
# -------------------------------
if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_input.head())

        df_input = clean_uploaded_data(df_input)
        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write(f"### {tr['prediction_results']}")
        st.dataframe(df_output[["Probability", "Prediction"]])
        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        st.write("### 🔍 Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        st.pyplot(fig)

    st.write("---")
    st.write(f"### {tr['individual_entry']}")

    # Individual Form
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
            'LIFESTYLE_SCORE': [lifestyle_score],
            'AGE_GROUP_Senior': [1 if age > 60 else 0]
        })

        for col in feature_names:
            if col not in row:
                row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold)

        st.subheader("✅ Prediction Result")
        st.write(f"Prediction: {'🛑 LUNG CANCER' if pred == 1 else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

        # Confidence bar
        fig, ax = plt.subplots()
        ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        st.pyplot(fig)

        # ✅ SHAP Explanation
        try:
            explainer = shap.Explainer(pipeline.named_steps['classifier'])
            shap_values = explainer(row)
            st.write("### 🧠 SHAP Feature Impact")
            shap.plots.waterfall(shap_values[0], max_display=10)
            st.pyplot(plt)
        except Exception:
            st.warning("⚠️ SHAP not supported for this model. Showing top permutation importances instead.")

        # ✅ Download Buttons
        pdf_buffer = generate_pdf(pred, prob)
        st.download_button("📥 " + tr['download_pdf'], data=pdf_buffer, file_name="prediction_result.pdf", mime="application/pdf")

        csv_data = pd.DataFrame({"Prediction": ["Lung Cancer" if pred else "No Lung Cancer"], "Probability": [prob]}).to_csv(index=False)
        st.download_button("📥 " + tr['download_csv_single'], data=csv_data, file_name="prediction_result.csv", mime="text/csv")

