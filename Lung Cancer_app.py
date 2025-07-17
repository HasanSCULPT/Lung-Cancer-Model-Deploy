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

# =========================================================
# ✅ Lung Cancer Diagnostic App (Streamlit)
# By HasanSCULPT | DSA 2025
# =========================================================
# 🔹 Deployment: Streamlit Cloud or Local
# 🔹 Features:
#    ✅ Multilingual UI (EN, FR, AR, RU, UK)
#    ✅ Upload CSV for batch prediction (with cleaning)
#    ✅ Individual prediction form
#    ✅ Threshold tuning (Max Recall & ROC)
#    ✅ SHAP (KernelExplainer) OR Permutation toggle
#    ✅ Confidence bar chart
#    ✅ Download results as CSV & PDF
#    ✅ Email sending (placeholders included)
#    ✅ Background image & logo supported
# =========================================================


# ----------------------------
# ✅ Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ✅ Background Image & Logo
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

# ✅ Load Model & Features
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")



# ✅ Language Translations
def get_translation(language):
    translations = {
        "en": {"title": "Lung Cancer Diagnostics Centre", "subtitle": "By HasanSCULPT | DSA 2025",
               "upload_csv": "Upload your CSV data", "prediction_results": "Prediction Results",
               "download_csv": "Download Results CSV", "export": "Export Result",
               "download_csv_single": "Download CSV", "download_pdf": "Download PDF",
               "enter_email": "Enter your email address to receive results", "send_email": "Send Email",
               "email_success": "✅ Email sent successfully!", "email_fail": "❌ Failed to send email.",
               "language_select": "🌍 Select Language", "sidebar_title": "Navigate",
               "individual_entry": "Enter patient information for individual prediction",
               "about_title": "📘 About Us", "about_desc": """This app is developed by HasanSCULPT to assist in preliminary lung cancer risk prediction using ensemble machine learning based on symptomatic analytics and lifestyle.
    This Diagnostic application allows for Individual Prediction + Batch CSV upload with validation & cleaning, Confidence chart for individual predictions, Toggle for SHAP or 
    Permutation Importance for individual prediction. 
    It should be noted also that in order to accurately execute raw batch predictions, datasets must be properly cleaned, features correctly encoded, because the model utilizes this 
    numeric idetifiers 1 and 0, meaning 1 equalsLung cancer while 0 equals No Lung cancer. Gender should also utilize numeric identifiers instead of MALE or FEMALE identifiers, these 
    measures if taken would further enhance a more accurate predictions.""",
               "contact_title": "📧 Contact Us", "terms_title": "📜 Terms & Conditions",
               "terms_text": "Disclaimer: This tool is for educational and diagnostic support only. Not an absolute substitute for professional medical advice."},
        "fr": {"title": "Centre de Diagnostic du Cancer du Poumon", "subtitle": "Par HasanSCULPT | DSA 2025",
               "upload_csv": "Téléchargez votre fichier CSV", "prediction_results": "Résultats de la prédiction",
               "download_csv": "Télécharger CSV", "export": "Exporter le résultat",
               "download_csv_single": "Télécharger CSV", "download_pdf": "Télécharger PDF",
               "enter_email": "Entrez votre email", "send_email": "Envoyer l'email",
               "email_success": "✅ Email envoyé avec succès!", "email_fail": "❌ Échec de l'envoi de l'email.",
               "language_select": "🌍 Sélectionnez la langue", "sidebar_title": "Navigation",
               "individual_entry": "Entrez les informations du patient",
               "about_title": "📘 À propos", "about_desc": "Cette application prédit le risque de cancer du poumon.",
               "contact_title": "📧 Contactez-nous", "terms_title": "📜 Conditions générales",
               "terms_text": "Cet outil est à des fins éducatives uniquement."}
    }
    return translations.get(language, translations["en"])

# 🌐 Language Selector
LANG_OPTIONS = {
    "en": "English",
    "fr": "Français",
    "ru": "Русский",
    "ar": "العربية",
    "uk": "Українська"
}
selected_lang = st.sidebar.selectbox(
    "🌍 Select Language",
    options=list(LANG_OPTIONS.keys()),
    format_func=lambda x: LANG_OPTIONS[x],
    key="lang"
)

# 🌐 Retrieve selected translation
tr = get_translation(selected_lang)
# ----------------------------
# ✅ Header Section
# ----------------------------
st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

# Sidebar Navigation
page = st.sidebar.selectbox(tr['sidebar_title'], ["Prediction", "About", "Contact", "Terms"], key="page")

# Email input
# Email sender (placeholder)
email = st.text_input(tr['enter_email'], key="email")
if email and st.button(tr['send_email'], key="email_btn"):
    success = send_email(email, tr['title'], "See attached result.", "prediction_result.pdf")
    if success:
        st.success(tr['email_success'])
    else:
        st.error(tr['email_fail'])
# ----------------------------
# ✅ Email Setup (Placeholder)
# ----------------------------
def send_email(recipient_email, subject, body, attachment_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = "your_email@example.com"
        msg["To"] = recipient_email
        msg.set_content(body)
        with open(attachment_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="prediction_result.pdf")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("your_email@example.com", "your_password")
            smtp.send_message(msg)
        return True
    except:
        return False

# ----------------------------
# ✅ Page Routing
# ----------------------------
if page == "About":
    st.title(tr['about_title']); st.write(tr['about_desc'])
elif page == "Contact":
    st.title(tr['contact_title']); st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.title(tr['terms_title']); st.write(tr['terms_text'])
elif page == "Prediction":
    # Sidebar Threshold
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv", key="csv")
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data"); st.dataframe(df_input.head())

        # ✅ Data Cleaning
        required_cols = ['AGE','GENDER','SMOKING','ANXIETY','ALCOHOL CONSUMING','PEER_PRESSURE','COUGHING','SHORTNESS OF BREATH']
        for col in required_cols:
            if col not in df_input.columns: df_input[col] = 0
        for col in df_input.columns:
            if df_input[col].isnull().sum() > 0:
                if df_input[col].dtype in ['int64','float64']:
                    df_input[col].fillna(df_input[col].mean(), inplace=True)
                else:
                    df_input[col].fillna(df_input[col].mode()[0], inplace=True)   
        # Align features
        df_input = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_input: df_input[col] = 0
        df_input = df_input[feature_names]
        

        #Automatic Threshold Suggestion
        # 
        #✅ Prediction
        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write(f"### {tr['prediction_results']}")
        st.dataframe(df_output[["Probability", "Prediction"]])
        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        st.pyplot(fig)

    # Histogram
        
        

     # Probability Distribution Plot
        
  

    # ✅ Individual Prediction
    st.write("---"); st.write(f"### {tr['individual_entry']}")
    age = st.number_input("Age",0,100,50); gender = st.selectbox("Gender",["Male","Female"])
    smoking = st.selectbox("Smoking",[0,1]); anxiety = st.selectbox("Anxiety",[0,1])
    alcohol = st.selectbox("Alcohol Consuming",[0,1]); peer_pressure = st.selectbox("Peer Pressure",[0,1])
    cough = st.selectbox("Coughing",[0,1]); short_breath = st.selectbox("Shortness of Breath",[0,1])
    symptom_score = st.slider("SYMPTOM SCORE",0,10,5); lifestyle_score = st.slider("LIFESTYLE SCORE",0,5,2)

    if st.button("Predict Individual"):
        row = pd.DataFrame({'AGE':[age],'GENDER':[1 if gender=="Male" else 0],'SMOKING':[smoking],
                            'ANXIETY':[anxiety],'ALCOHOL CONSUMING':[alcohol],'PEER_PRESSURE':[peer_pressure],
                            'COUGHING':[cough],'SHORTNESS OF BREATH':[short_breath],
                            'SYMPTOM_SCORE':[symptom_score],'LIFESTYLE_SCORE':[lifestyle_score]})
        for col in feature_names:
            if col not in row: row[col]=0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]; pred = int(prob>threshold)
        st.success(f"{'🛑 LUNG CANCER' if pred==1 else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")

        # ✅ Confidence Chart
        fig, ax = plt.subplots(); bars = ax.bar(["No Lung Cancer","Lung Cancer"],[1-prob,prob],color=["green","red"])
        ax.set_ylim(0,1); ax.set_ylabel("Probability"); ax.set_title("Prediction Confidence")
        for bar in bars: yval=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2.0,yval+0.02,f"{yval:.2f}",ha='center')
        st.pyplot(fig)

        # ✅ Toggle SHAP or Permutation
        
        perm = permutation_importance(pipeline,row,[pred],n_repeats=5,random_state=42)
        st.write("### Permutation Importance")
        fig3, ax3 = plt.subplots(); ax3.barh(feature_names, perm.importances_mean,color='teal')
        ax3.set_title("Permutation Importance"); st.pyplot(fig3)

        # ✅ Export PDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
        pdf.cell(200,10,txt="Lung Cancer Prediction Result",ln=True,align='C')
        pdf.cell(200,10,txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}",ln=True)
        pdf.cell(200,10,txt=f"Probability: {prob:.2f}",ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name="prediction_result.pdf", mime="application/pdf")

