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

st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# ✅ Background Image
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

# ✅ Load Model & Features
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# ✅ Hardcoded SHAP background sample for better accuracy
background_sample = pd.DataFrame({
    'AGE': [45, 60, 35, 55, 70],
    'GENDER': [1, 0, 1, 1, 0],
    'SMOKING': [1, 0, 1, 1, 1],
    'ANXIETY': [0, 1, 0, 1, 0],
    'ALCOHOL CONSUMING': [0, 1, 0, 0, 1],
    'PEER_PRESSURE': [1, 1, 0, 1, 0],
    'COUGHING': [1, 1, 0, 1, 1],
    'SHORTNESS OF BREATH': [0, 1, 0, 1, 1],
    'SYMPTOM_SCORE': [5, 8, 3, 7, 9],
    'LIFESTYLE_SCORE': [2, 1, 3, 1, 0]
})
for col in feature_names:
    if col not in background_sample:
        background_sample[col] = 0
background_sample = background_sample[feature_names]

# ✅ UI Header
st.image("logo.png", width=100)
st.title("🔬 Lung Cancer Diagnostics Centre")
st.write("## By HasanSCULPT | DSA 2025")

# ✅ Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Prediction", "About", "Contact", "Terms"])

# ----------------------------
# ✅ About & Contact Pages
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
# -----------------------------
# Pages
# -----------------------------
if page == "About":
    st.header("📘 About Us")
    st.write("""This app is developed by HasanSCULPT to assist in preliminary lung cancer risk prediction using ensemble machine learning based on symptomatic analytics and lifestyle.
    This Diagnostic application allows for Individual Prediction + Batch CSV upload with validation & cleaning, Confidence chart for individual predictions, Toggle for SHAP or 
    Permutation Importance for individual prediction. 
    It should be noted also that in order to accurately execute raw batch predictions, datasets must be properly cleaned, features correctly encoded, because the model utilizes this 
    numeric idetifiers 1 and 0, meaning 1 equalsLung cancer while 0 equals No Lung cancer. Gender should also utilize numeric identifiers instead of MALE or FEMALE identifiers, these 
    measures if taken would further enhance a more accurate predictions.""")
    
elif page == "Contact":
    st.header("📧 Contact")
    st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.header("📜 Terms")
    st.write("Disclaimer: This tool is for educational and diagnostic support only. Not an absolute substitute for professional medical advice.")
else:
    # Prediction Page
    st.header("Welcom To The Lung Cancer Diagnostics Centre")="Font size 12")

# ----------------------------
# ✅ Prediction Page
# ----------------------------
if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df_input.head())

        required_cols = ['AGE','GENDER','SMOKING','ANXIETY','ALCOHOL CONSUMING','PEER_PRESSURE','COUGHING','SHORTNESS OF BREATH']
        for col in required_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        for col in df_input.columns:
            if df_input[col].isnull().sum() > 0:
                if df_input[col].dtype in ['int64','float64']:
                    df_input[col].fillna(df_input[col].mean(), inplace=True)
                else:
                    df_input[col].fillna(df_input[col].mode()[0], inplace=True)
# Align features
        df_input = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_input:
                df_input[col] = 0
        df_input = df_input[feature_names]

        st.write("### 🔍 Automatic Threshold Suggestions")
        proba_temp = pipeline.predict_proba(df_input)[:,1]

        if "LUNG_CANCER" in df_input.columns:
            y_true = df_input["LUNG_CANCER"]
            best_recall, best_thresh = 0, threshold
            for t in np.arange(0.1, 0.9, 0.01):
                y_pred_temp = (proba_temp > t).astype(int)
                recall = recall_score(y_true, y_pred_temp)
                if recall > best_recall:
                    best_recall, best_thresh = recall, t
            st.success(f"Max Recall Threshold: {best_thresh:.2f}")
            if st.button("Apply Max Recall"):
                threshold = best_thresh

        fpr, tpr, thresholds_roc = roc_curve((proba_temp > 0.5).astype(int), proba_temp)
        optimal_threshold = thresholds_roc[np.argmax(tpr - fpr)]
        st.info(f"ROC Optimal Threshold: {optimal_threshold:.2f}")
        if st.button("Apply ROC Optimal"):
            threshold = optimal_threshold

        proba = pipeline.predict_proba(df_input)[:,1]
        prediction = (proba > threshold).astype(int)
        df_output = pd.DataFrame({"Probability": proba,"Prediction": prediction})
        st.write("### Prediction Results")
        st.dataframe(df_output)
        st.download_button("📥 Download CSV", df_output.to_csv(index=False), "batch_predictions.csv","text/csv")
        # Histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        st.pyplot(fig)
        

# =====================================
# Individual Prediction
# =====================================
    st.write("---")
    st.write("### Individual Prediction")
    age = st.number_input("Age",0,100,50)
    gender = st.selectbox("Gender",["Male","Female"])
    smoking = st.selectbox("Smoking",[0,1])
    anxiety = st.selectbox("Anxiety",[0,1])
    alcohol = st.selectbox("Alcohol Consuming",[0,1])
    peer_pressure = st.selectbox("Peer Pressure",[0,1])
    cough = st.selectbox("Coughing",[0,1])
    short_breath = st.selectbox("Shortness of Breath",[0,1])
    symptom_score = st.slider("SYMPTOM SCORE",0,10,5)
    lifestyle_score = st.slider("LIFESTYLE SCORE",0,5,2)

    if st.button("Predict Individual"):
        row = pd.DataFrame({
            'AGE':[age],
            'GENDER':[1 if gender=="Male" else 0],
            'SMOKING':[smoking],
            'ANXIETY':[anxiety],
            'ALCOHOL CONSUMING':[alcohol],
            'PEER_PRESSURE':[peer_pressure],
            'COUGHING':[cough],
            'SHORTNESS OF BREATH':[short_breath],
            'SYMPTOM_SCORE':[symptom_score],
            'LIFESTYLE_SCORE':[lifestyle_score]
        })
        for col in feature_names:
            if col not in row:
                row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold)
        st.success(f"{'🛑 LUNG CANCER' if pred else '✅ NO LUNG CANCER'} (Probability: {prob:.2f})")
       # Confidence Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["No Lung Cancer","Lung Cancer"],[1-prob, prob],color=["green","red"])
        ax.set_ylim(0,1)
        st.pyplot(fig)

        
        st.write("### Permutation Importance")
        result = permutation_importance(pipeline,row,[pred],n_repeats=5,random_state=42)
        importance_df = pd.DataFrame({"Feature":feature_names,"Importance":result.importances_mean})
        importance_df = importance_df.sort_values(by="Importance",ascending=False)
        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"][:10],importance_df["Importance"][:10],color='teal')
        st.pyplot(fig)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial",size=12)
        pdf.cell(200,10,txt="Lung Cancer Prediction Result",ln=True,align='C')
        pdf.cell(200,10,txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}",ln=True)
        pdf.cell(200,10,txt=f"Probability: {prob:.2f}",ln=True)
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(label="📥 Download PDF",data=pdf_buffer,file_name="prediction_result.pdf",mime="application/pdf")

