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

# ----------------------------
# ✅ Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ✅ Background Image Setup
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

set_png_as_page_bg("background.png")

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
            "export": "Export Result",
            "download_csv_single": "Download CSV",
            "download_pdf": "Download PDF",
            "enter_email": "Enter your email address to receive results",
            "send_email": "Send Email",
            "email_success": "✅ Email sent successfully!",
            "email_fail": "❌ Failed to send email. Check configuration.",
            "language_select": "🌍 Select Language",
            "sidebar_title": "Navigate",
            "individual_entry": "Enter your medical/patient information below:",
            "about_title": "📘 About Us",
            "about_desc": "This app assists in preliminary lung cancer risk prediction using machine learning.",
            "contact_title": "📧 Contact Us",
            "terms_title": "📜 Terms & Conditions",
            "terms_text": "This tool is for educational and diagnostic support only. Not a substitute for medical advice."
        }
    }
    return translations.get(language, translations["en"])

LANG_OPTIONS = {"en": "English"}
selected_lang = st.sidebar.selectbox("🌍 Select Language", options=list(LANG_OPTIONS.keys()), format_func=lambda x: LANG_OPTIONS[x])
tr = get_translation(selected_lang)

# ----------------------------
# ✅ Header Section
# ----------------------------
st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

# ----------------------------
# ✅ Sidebar Navigation
# ----------------------------
page = st.sidebar.selectbox(tr['sidebar_title'], ["Prediction", "About", "Contact", "Terms"], key="page")

# ----------------------------
# ✅ Email Setup
# ----------------------------
def send_email(recipient_email, subject, body, attachment_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = "your_email@example.com"
        msg["To"] = recipient_email
        msg.set_content(body)

        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = f.name

        msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("your_email@example.com", "your_password")
            smtp.send_message(msg)
        return True
    except Exception:
        return False

email = st.text_input(tr['enter_email'], key="email")
if email and st.button(tr['send_email'], key="email_btn"):
    success = send_email(email, tr['title'], "See attached result.", "prediction_result.pdf")
    if success: st.success(tr['email_success'])
    else: st.error(tr['email_fail'])

# ----------------------------
# ✅ Page Routing
# ----------------------------
if page == "About":
    st.title(tr['about_title']); st.write(tr['about_desc'])
elif page == "Contact":
    st.title(tr['contact_title']); st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.title(tr['terms_title']); st.write(tr['terms_text'])

# ----------------------------
# ✅ Prediction Page
# ----------------------------
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv", key="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data"); st.dataframe(df_input.head())

        # ✅ Data Cleaning & Validation
        required_cols = ['AGE', 'GENDER', 'SMOKING', 'ANXIETY', 'ALCOHOL CONSUMING', 'PEER_PRESSURE', 'COUGHING', 'SHORTNESS OF BREATH']
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        if missing_cols:
            st.warning(f"⚠ Missing columns: {missing_cols}. Filling defaults.")
            for col in missing_cols: df_input[col] = 0

        if df_input.isnull().sum().sum() > 0:
            st.warning("⚠ Missing values detected. Imputing...")
            for col in df_input.columns:
                if df_input[col].dtype in ['int64','float64']: df_input[col].fillna(df_input[col].mean(), inplace=True)
                else: df_input[col].fillna(df_input[col].mode()[0], inplace=True)

        # Align with model features
        df_input = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_input: df_input[col] = 0
        df_input = df_input[feature_names]

        # Predictions
        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)
        df_output = df_input.copy(); df_output["Probability"] = proba; df_output["Prediction"] = prediction
        st.write(f"### {tr['prediction_results']}"); st.dataframe(df_output[["Probability", "Prediction"]])

        # ✅ Optimal Threshold Suggestion
        fpr, tpr, thresholds = roc_curve((proba > 0.5).astype(int), proba)
        youden_j = tpr - fpr; best_idx = np.argmax(youden_j); optimal_threshold = thresholds[best_idx]
        st.info(f"🔍 Suggested Threshold: **{optimal_threshold:.2f}**")
        if st.button("Apply Suggested Threshold"): threshold = float(optimal_threshold); st.success(f"✅ Threshold updated to {threshold:.2f}")

        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        # Probability Distribution Plot
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k'); ax.axvline(threshold, color='red', linestyle='--')
        ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # ✅ Individual Prediction Form
    st.write("---"); st.write(f"### {tr['individual_entry']}")
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
            'AGE': [age], 'GENDER': [1 if gender == "Male" else 0], 'SMOKING': [smoking],
            'ANXIETY': [anxiety], 'ALCOHOL CONSUMING': [alcohol], 'PEER_PRESSURE': [peer_pressure],
            'COUGHING': [cough], 'SHORTNESS OF BREATH': [short_breath],
            'SYMPTOM_SCORE': [symptom_score], 'LIFESTYLE_SCORE': [lifestyle_score],
            'AGE_GROUP_Senior': [1 if age > 60 else 0]
        })
        for col in feature_names:
            if col not in row: row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]; pred = int(prob > threshold)
        st.success(f"{'🛑 LUNG CANCER' if pred == 1 else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

        # ✅ Confidence Bar Chart
        fig, ax = plt.subplots()
        bars = ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1); ax.set_ylabel("Probability"); ax.set_title("Prediction Confidence")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center')
        st.pyplot(fig)

        # ✅ SHAP Explanation
        st.write("### 🧠 Feature Contribution (SHAP)")
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(row)
        shap_df = pd.DataFrame({"Feature": row.columns, "SHAP Value": shap_values[1][0]}).sort_values("SHAP Value", key=abs, ascending=False)
        fig4 = px.bar(shap_df.head(10), x="SHAP Value", y="Feature", orientation='h', title="Top 10 Contributing Features")
        st.plotly_chart(fig4)
