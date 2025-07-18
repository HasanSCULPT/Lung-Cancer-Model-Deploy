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

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# -----------------------------
# Background Image
# -----------------------------
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_png_as_page_bg(png_file):
    try:
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
    except FileNotFoundError:
        st.warning("⚠ Background image not found. Skipping...")
        
set_png_as_page_bg("background.png")  # Update file name if needed

# -----------------------------
# Load Model & Features
# -----------------------------
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🔍 Settings")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
enable_shap = st.sidebar.checkbox("Enable SHAP for Individual Prediction", value=False)

# -----------------------------
# App Title
# -----------------------------
st.image("logo.png", width=100)
st.title("🔬 Lung Cancer Diagnostics Centre")
st.write("### Powered by Ensemble ML | DSA 2025")

page = st.sidebar.radio("Navigate", ["Prediction", "About", "Contact", "Terms"])

# -----------------------------
# Pages
# -----------------------------
if page == "About":
    st.header("📘 About Us")
    st.write("This app helps predict lung cancer risk using machine learning. Educational use only.")
elif page == "Contact":
    st.header("📧 Contact")
    st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.header("📜 Terms")
    st.write("This tool is for educational purposes only and does not replace medical advice.")
else:
    # Prediction Page
    st.header("Lung Cancer Prediction")

    # -------------------------
    # Upload CSV
    # -------------------------
    uploaded_file = st.file_uploader("📂 Upload CSV for Batch Prediction", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)

        # ✅ Validate Columns
        missing_cols = [col for col in feature_names if col not in df_input.columns]
        for col in missing_cols:
            df_input[col] = 0
        extra_cols = [col for col in df_input.columns if col not in feature_names]
        if extra_cols:
            st.warning(f"Removing extra columns: {extra_cols}")
            df_input = df_input.drop(columns=extra_cols)
        df_input = df_input[feature_names]

        # ✅ Predictions
        proba = pipeline.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write("### ✅ Prediction Results")
        st.dataframe(df_output[["Probability", "Prediction"]])

        st.download_button("📥 Download Predictions", df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        # ✅ Histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, color="teal", edgecolor="black")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

        # ✅ Threshold Suggestions
        st.subheader("🔍 Optimal Threshold Suggestions")

        # Option 1: Recall Maximization
        best_thresh, best_recall = threshold, 0
        if "LUNG_CANCER" in df_input.columns:  # If actual labels are in data
            y_true = df_input["LUNG_CANCER"]
            for t in np.arange(0.1, 0.9, 0.01):
                y_pred_temp = (proba > t).astype(int)
                recall = recall_score(y_true, y_pred_temp)
                if recall > best_recall:
                    best_recall, best_thresh = recall, t
            st.success(f"✅ Recall-based Suggested Threshold: {best_thresh:.2f} (Recall: {best_recall:.2f})")

        # Option 2: ROC Curve Youden J
        fpr, tpr, thresholds = roc_curve((proba > 0.5).astype(int), proba)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        st.info(f"📈 ROC-based Suggested Threshold: {optimal_threshold:.2f}")
        if st.button("Apply ROC Threshold"):
            threshold = float(optimal_threshold)
            st.success(f"✅ Threshold updated to {threshold:.2f}")

    else:
        st.info("Upload a CSV to start batch prediction.")

    st.write("---")
    # -------------------------
    # Individual Prediction
    # -------------------------
    st.subheader("🧍 Individual Prediction")
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
            "AGE": [age],
            "GENDER": [1 if gender == "Male" else 0],
            "SMOKING": [smoking],
            "ANXIETY": [anxiety],
            "ALCOHOL CONSUMING": [alcohol],
            "PEER_PRESSURE": [peer_pressure],
            "COUGHING": [cough],
            "SHORTNESS OF BREATH": [short_breath],
            "SYMPTOM_SCORE": [symptom_score],
            "LIFESTYLE_SCORE": [lifestyle_score]
        })

        # Ensure all features exist
        for col in feature_names:
            if col not in row:
                row[col] = 0
        row = row[feature_names]

        prob = pipeline.predict_proba(row)[0][1]
        pred = int(prob > threshold)

        st.write(f"**Prediction:** {'LUNG CANCER' if pred == 1 else 'NO LUNG CANCER'}")
        st.write(f"**Probability:** {prob:.2f}")

        # Confidence bar chart
        fig, ax = plt.subplots()
        ax.bar(["No Lung Cancer", "Lung Cancer"], [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # ✅ SHAP Explanation (optional)
        
        # ✅ Download Buttons
        result_df = pd.DataFrame({
            "Prediction": ["LUNG CANCER" if pred else "NO LUNG CANCER"],
            "Probability": [prob]
        })
        st.download_button("📥 Download Result (CSV)", result_df.to_csv(index=False), "prediction_result.csv", "text/csv")

        # ✅ Export PDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
        pdf.cell(200,10,txt="Lung Cancer Prediction Result",ln=True,align='C')
        pdf.cell(200,10,txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}",ln=True)
        pdf.cell(200,10,txt=f"Probability: {prob:.2f}",ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name="prediction_result.pdf", mime="application/pdf")

