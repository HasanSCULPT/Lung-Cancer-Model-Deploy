import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.inspection import permutation_importance

# Load model and scaler
voting_clf = joblib.load("voting_clf.pkl")

st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")
st.image("logo.png", width=100)
st.title("🔬 Lung Cancer Prediction using Ensemble Model")
st.write("## By HasanSCULPT | DSA 2025")

# Sidebar: Navigation
page = st.sidebar.selectbox("Navigate", ["Prediction", "About", "Contact", "Terms"])

if page == "Prediction":
    st.sidebar.subheader("🔧 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    # Sidebar: Upload data
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type="csv")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_input.head())

        df_input = pd.get_dummies(df_input, drop_first=True)
        model_features = voting_clf.named_estimators_["rf"].feature_names_in_
        for col in model_features:
            if col not in df_input:
                df_input[col] = 0
        df_input = df_input[model_features]

        proba = voting_clf.predict_proba(df_input)[:, 1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write("### Prediction Results")
        st.dataframe(df_output[["Probability", "Prediction"]])

        st.write("### 🔍 Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("### 📊 Precomputed Permutation Importance (Top Predictors)")
        importance_data = {
            "Feature": [
                "SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH",
                "SWALLOWING DIFFICULTY", "ALCOHOL CONSUMING", "ANXIETY",
                "COUGHING", "WHEEZING", "SMOKING", "GENDER", "AGE_GROUP_Senior",
                "AGE", "YELLOW_FINGERS", "PEER_PRESSURE", "CHEST PAIN",
                "LIFESTYLE_RISK", "ALLERGY", "FATIGUE", "AGE_GROUP_Middle-aged", "CHRONIC DISEASE"
            ],
            "Importance": [
                6.290323e-02, 3.709677e-02, 2.741935e-02, 2.580645e-02, 2.419355e-02,
                2.419355e-02, 2.096774e-02, 1.935484e-02, 1.935484e-02, 1.129032e-02,
                9.677419e-03, 9.677419e-03, 8.064516e-03, 8.064516e-03, 4.838710e-03,
                1.612903e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, -2.220446e-17
            ]
        }
        importance_df = pd.DataFrame(importance_data).sort_values(by="Importance", ascending=True)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.barh(importance_df["Feature"], importance_df["Importance"], color='teal')
        ax3.set_xlabel("Mean Importance Score")
        ax3.set_title("Permutation Importance (Precomputed)")
        st.pyplot(fig3)

        st.write("### 🧠 SHAP Explanation (Random Forest)")
        explainer = shap.TreeExplainer(voting_clf.named_estimators_["rf"])
        shap_values = explainer.shap_values(df_input)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values[1], df_input, plot_type="bar")
        st.pyplot()

    else:
        st.info("⬅️ Upload a CSV file to start prediction")

    st.write("---")
    st.write("### Or Enter Individual Patient Information")
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
        for col in voting_clf.named_estimators_["rf"].feature_names_in_:
            if col not in row:
                row[col] = 0
        row = row[voting_clf.named_estimators_["rf"].feature_names_in_]
        prob = voting_clf.predict_proba(row)[0][1]
        pred = int(prob > threshold)
        st.success(f"Predicted: {'Lung Cancer' if pred==1 else 'No Lung Cancer'} (Probability: {prob:.2f})")

elif page == "About":
    st.title("📘 About Us")
    st.write("This lung cancer diagnostic app is developed By HasanSCULPT to assist in preliminary lung cancer risk prediction using an ensemble ensemble of Random Forest, Logistic Regression and optionally SVCmachine learning model based on patient lifestyle and symptom data It is built for educational and clinical support purposes..")

elif page == "Contact":
    st.title("📧 Contact Us")
    st.write("Phone: +234-456-7890")
    st.write("For questions or feedback, email us at: support@lungdiagnosis.ai")

elif page == "Terms":
    st.title("📜 Terms & Conditions")
    st.write("This tool is intended for educational or preliminary diagnostic use only and not as a substitute for professional medical advice. Always consult a certified medical professional for clinical decisions.")


