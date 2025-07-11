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
from email.message import EmailMessage
from fpdf import FPDF
from sklearn.inspection import permutation_importance

# Streamlit setup first
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ⬛ Feathered Background Setup
def add_body_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    page_bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

add_body_background("feathered_bg.png")

# 🌍 Language translations

def get_translation(language):
    translations = {...}  # Full dictionary remains unchanged for brevity
    return translations.get(language, translations["en"])

# Email sender (set your SMTP credentials securely!)
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
    except Exception as e:
        return False

# Language selector in sidebar
LANG_OPTIONS = {
    "en": "English",
    "fr": "Français",
    "ru": "Русский",
    "ar": "العربية",
    "uk": "Українська"
}

selected_lang = st.sidebar.selectbox(
    "🌍 Language",
    options=list(LANG_OPTIONS.keys()),
    format_func=lambda x: LANG_OPTIONS[x],
    key="lang"
)
tr = get_translation(selected_lang) # Then, retrieve translation with
def get_translation(language):
    translations = {
        "en": { ... },
        "fr": { ... },
        ...
    }
    return translations.get(language, translations["en"])



# App Title and Subtitle
st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

# Sidebar Navigation
page = st.sidebar.selectbox(tr['sidebar_title'], ["Prediction", "About", "Contact", "Terms"], key="page")

# Email input
email = st.text_input(tr['enter_email'], key="email")
if email and st.button(tr['send_email'], key="email_btn"):
    success = send_email(email, tr['title'], "See attached result.", "prediction_result.pdf")
    if success:
        st.success(tr['email_success'])
    else:
        st.error(tr['email_fail'])

# Page Routing
if page == "About":
    st.title(tr['about_title'])
    st.write(tr['about_desc'])
elif page == "Contact":
    st.title(tr['contact_title'])
    st.write("Phone: +234-000-0000")
    st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.title(tr['terms_title'])
    st.write(tr['terms_text'])

# Load pipeline & feature names
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv", key="csv")
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_input.head())

        df_input = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_input:
                df_input[col] = 0
        df_input = df_input[feature_names]

        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write(f"### {tr['prediction_results']}")
        st.dataframe(df_output[["Probability", "Prediction"]])

        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        st.write("### 🔍 Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("### 📊 Precomputed Permutation Importance")
        importance_data = {...}  # unchanged
        importance_df = pd.DataFrame(importance_data).sort_values(by="Importance", ascending=True)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.barh(importance_df["Feature"], importance_df["Importance"], color='teal')
        ax3.set_xlabel("Mean Importance Score")
        ax3.set_title("Permutation Importance (Precomputed)")
        st.pyplot(fig3)

        st.write("### 🧠 SHAP Explanation (Random Forest)")
        rf_model = pipeline.named_steps["model"].estimators[0]
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(df_input)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values[1], df_input, plot_type="bar")
        st.pyplot()
    else:
        st.info("⬅️ Upload a CSV file to start prediction")

    # Individual Prediction Form
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
    age_group_senior = 1 if age > 60 else 0

    if st.button("Predict Individual", key="ind_pred"):
        row = pd.DataFrame({
            'AGE': [age], 'GENDER': [1 if gender == "Male" else 0],
            'SMOKING': [smoking], 'ANXIETY': [anxiety], 'ALCOHOL CONSUMING': [alcohol],
            'PEER_PRESSURE': [peer_pressure], 'COUGHING': [cough],
            'SHORTNESS OF BREATH': [short_breath],
            'SYMPTOM_SCORE': [symptom_score], 'LIFESTYLE_SCORE': [lifestyle_score],
            'AGE_GROUP_Senior': [age_group_senior]
        })

        for col in feature_names:
            if col not in row:
                row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold)

        if pred == 1:
            st.success(f"🛑 Predicted: LUNG CANCER (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Predicted: NO LUNG CANCER (Probability: {prob:.2f})")

        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots()
        bars = ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
        st.pyplot(fig)

        if st.button(tr['export'], key="exp_btn"):
            result_df = pd.DataFrame({"Prediction": ["Lung Cancer" if pred == 1 else "No Lung Cancer"], "Probability": [prob]})
            st.download_button("📥 " + tr['download_csv_single'], result_df.to_csv(index=False), "prediction_result.csv", "text/csv")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER 🛑' if pred == 1 else 'NO LUNG CANCER ✅'}", ln=True)
            pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
            pdf.output("prediction_result.pdf")

            with open("prediction_result.pdf", "rb") as f:
                st.download_button("📥 " + tr['download_pdf'], f, file_name="prediction_result.pdf")
