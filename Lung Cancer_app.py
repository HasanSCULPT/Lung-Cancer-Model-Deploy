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
from fpdf import FPDF
from sklearn.inspection import permutation_importance

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

# Load trained pipeline and feature names
pipeline = joblib.load("lung_cancer_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit setup
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")
add_body_background("feathered_bg.png")
st.image("logo.png", width=100)
st.title("🔬 Lung Cancer Diagnostics Centre")
st.write("## By HasanSCULPT | DSA 2025")

page = st.sidebar.selectbox("Navigate", ["Prediction", "About", "Contact", "Terms"])

def validate_inputs(age, symptom_score, lifestyle_score):
    if not (0 <= age <= 100):
        st.warning("⚠️ Age should be between 0 and 100.")
        return False
    if not (0 <= symptom_score <= 10):
        st.warning("⚠️ Symptom Score should be between 0 and 10.")
        return False
    if not (0 <= lifestyle_score <= 5):
        st.warning("⚠️ Lifestyle Score should be between 0 and 5.")
        return False
    return True

if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type="csv")
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

        st.write("### Prediction Results")
        st.dataframe(df_output[["Probability", "Prediction"]])

        st.download_button("📥 Download Results CSV", df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")

        st.write("### 🔍 Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("### 📊 Precomputed Permutation Importance")
        importance_data = {
            "Feature": ["SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY",
                        "ALCOHOL CONSUMING", "ANXIETY", "COUGHING", "WHEEZING", "SMOKING", "GENDER",
                        "AGE_GROUP_Senior", "AGE", "YELLOW_FINGERS", "PEER_PRESSURE", "CHEST PAIN",
                        "LIFESTYLE_RISK", "ALLERGY", "FATIGUE", "AGE_GROUP_Middle-aged", "CHRONIC DISEASE"],
            "Importance": [0.0629, 0.0371, 0.0274, 0.0258, 0.0242, 0.0242, 0.0210, 0.0194, 0.0194, 0.0113,
                           0.0097, 0.0097, 0.0081, 0.0081, 0.0048, 0.0016, 0.0, 0.0, 0.0, -2.2e-17]
        }
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

    # --- Individual Form ---
    st.write("---")
    st.write("### Or Enter Individual medical  Patient Information below to predict whether you're likely to have Lung Cancer or not.")

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

    if st.button("Predict Individual"):
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

        if st.button("Export Result"):
            result_df = pd.DataFrame({"Prediction": ["Lung Cancer" if pred == 1 else "No Lung Cancer"], "Probability": [prob]})
            st.download_button("📥 Download CSV", result_df.to_csv(index=False), "prediction_result.csv", "text/csv")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER 🛑' if pred == 1 else 'NO LUNG CANCER ✅'}", ln=True)
            pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
            pdf.output("prediction_result.pdf")

            with open("prediction_result.pdf", "rb") as f:
                st.download_button("📥 Download PDF", f, file_name="prediction_result.pdf")

elif page == "About":
    st.title("📘 About Us")
    st.write("This app is developed by HasanSCULPT to assist in preliminary lung cancer risk prediction using ensemble machine learning based on symptoms and lifestyle.")

elif page == "Contact":
    st.title("📧 Contact Us")
    st.write("Phone: +234-000-0000")
    st.write("Email: support@lungdiagnosis.ai")

elif page == "Terms":
    st.title("📜 Terms & Conditions")
    st.write("This tool is for educational and diagnostic support only. Not a substitute for professional medical advice.")
