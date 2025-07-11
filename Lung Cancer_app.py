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
# Add background image
add_background("feathered_bg.png")


import base64

def add_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
        encoded = base64.b64encode(img_data).decode()
        css = f"""
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
        st.markdown(css, unsafe_allow_html=True)

    st.markdown(page_bg_css, unsafe_allow_html=True)

# 🌍 Language translations

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
            "individual_entry": "Or Enter Individual Patient Information",
            "about_title": "📘 About Us",
            "about_desc": "This app is developed by HasanSCULPT to assist in preliminary lung cancer risk prediction using ensemble machine learning based on symptoms and lifestyle.",
            "contact_title": "📧 Contact Us",
            "terms_title": "📜 Terms & Conditions",
            "terms_text": "This tool is for educational and diagnostic support only. Not a substitute for professional medical advice."
        },
        "fr": {
            "title": "Centre de Diagnostic du Cancer du Poumon",
            "subtitle": "Par HasanSCULPT | DSA 2025",
            "upload_csv": "Téléchargez votre fichier CSV",
            "prediction_results": "Résultats de la prédiction",
            "download_csv": "Télécharger les résultats CSV",
            "export": "Exporter le résultat",
            "download_csv_single": "Télécharger CSV",
            "download_pdf": "Télécharger PDF",
            "enter_email": "Entrez votre adresse e-mail pour recevoir les résultats",
            "send_email": "Envoyer l'e-mail",
            "email_success": "✅ Email envoyé avec succès !",
            "email_fail": "❌ Échec de l'envoi de l'e-mail.",
            "language_select": "🌍 Sélectionnez la langue",
            "sidebar_title": "Navigation",
            "individual_entry": "Ou entrez les informations individuelles du patient",
            "about_title": "📘 À propos de nous",
            "about_desc": "Cette application a été développée par HasanSCULPT pour aider à la prédiction préliminaire du risque de cancer du poumon.",
            "contact_title": "📧 Contactez-nous",
            "terms_title": "📜 Conditions générales",
            "terms_text": "Cet outil est à des fins éducatives uniquement et ne remplace pas un avis médical professionnel."
        },
        "ru": {
            "title": "Центр Диагностики Рака Легких",
            "subtitle": "ХасанСКАЛЬПТ | DSA 2025",
            "upload_csv": "Загрузите ваш CSV файл",
            "prediction_results": "Результаты прогноза",
            "download_csv": "Скачать CSV с результатами",
            "export": "Экспортировать результат",
            "download_csv_single": "Скачать CSV",
            "download_pdf": "Скачать PDF",
            "enter_email": "Введите ваш email для получения результата",
            "send_email": "Отправить email",
            "email_success": "✅ Email успешно отправлен!",
            "email_fail": "❌ Не удалось отправить Email.",
            "language_select": "🌍 Выберите язык",
            "sidebar_title": "Навигация",
            "individual_entry": "Или введите информацию о пациенте",
            "about_title": "📘 О нас",
            "about_desc": "Это приложение разработано HasanSCULPT для помощи в предварительном прогнозировании риска рака легких.",
            "contact_title": "📧 Связаться с нами",
            "terms_title": "📜 Условия использования",
            "terms_text": "Этот инструмент предназначен только для образовательных целей и не заменяет профессиональную медицинскую консультацию."
        },
        "ar": {
            "title": "مركز تشخيص سرطان الرئة",
            "subtitle": "بواسطة حسنSculpt | DSA 2025",
            "upload_csv": "قم بتحميل ملف CSV الخاص بك",
            "prediction_results": "نتائج التنبؤ",
            "download_csv": "تحميل نتائج CSV",
            "export": "تصدير النتيجة",
            "download_csv_single": "تحميل CSV",
            "download_pdf": "تحميل PDF",
            "enter_email": "أدخل بريدك الإلكتروني لتلقي النتائج",
            "send_email": "إرسال بريد إلكتروني",
            "email_success": "✅ تم إرسال البريد الإلكتروني بنجاح!",
            "email_fail": "❌ فشل في إرسال البريد الإلكتروني.",
            "language_select": "🌍 اختر اللغة",
            "sidebar_title": "القائمة الجانبية",
            "individual_entry": "أدخل معلومات المريض الفردية",
            "about_title": "📘 معلومات عنا",
            "about_desc": "تم تطوير هذا التطبيق بواسطة حسنSculpt للمساعدة في التنبؤ الأولي بمخاطر سرطان الرئة.",
            "contact_title": "📧 تواصل معنا",
            "terms_title": "📜 الشروط والأحكام",
            "terms_text": "هذه الأداة لأغراض تعليمية فقط ولا تعتبر بديلاً عن الاستشارة الطبية المهنية."
        },
        "uk": {
            "title": "Центр Діагностики Раку Легенів",
            "subtitle": "ХасанСКАЛЬПТ | DSA 2025",
            "upload_csv": "Завантажте свій CSV файл",
            "prediction_results": "Результати прогнозу",
            "download_csv": "Завантажити результати CSV",
            "export": "Експортувати результат",
            "download_csv_single": "Завантажити CSV",
            "download_pdf": "Завантажити PDF",
            "enter_email": "Введіть свою електронну пошту для отримання результатів",
            "send_email": "Надіслати Email",
            "email_success": "✅ Email успішно надіслано!",
            "email_fail": "❌ Не вдалося надіслати Email.",
            "language_select": "🌍 Виберіть мову",
            "sidebar_title": "Навігація",
            "individual_entry": "Або введіть інформацію про пацієнта",
            "about_title": "📘 Про нас",
            "about_desc": "Цей додаток розроблений HasanSCULPT для допомоги у попередньому прогнозуванні ризику раку легенів.",
            "contact_title": "📧 Зв'язатися з нами",
            "terms_title": "📜 Умови використання",
            "terms_text": "Цей інструмент призначено лише для освітніх цілей і не замінює професійну медичну консультацію."
        }
    }
    return translations.get(language, translations["en"])


# 🌐 Language Selector Setup
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
        importance_data = {
    "Feature": [
        "SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY",
        "ALCOHOL CONSUMING", "ANXIETY", "COUGHING", "WHEEZING", "SMOKING", "GENDER",
        "AGE_GROUP_Senior", "AGE", "YELLOW_FINGERS", "PEER_PRESSURE", "CHEST PAIN",
        "LIFESTYLE_RISK", "ALLERGY", "FATIGUE", "AGE_GROUP_Middle-aged", "CHRONIC DISEASE"
    ],
    "Importance": [
        0.0629, 0.0371, 0.0274, 0.0258, 0.0242, 0.0242, 0.0210, 0.0194, 0.0194, 0.0113,
        0.0097, 0.0097, 0.0081, 0.0081, 0.0048, 0.0016, 0.0, 0.0, 0.0, -2.2e-17
    ]
}

        importance_df = pd.DataFrame(importance_data).sort_values(by="Importance", ascending=True)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.barh(importance_df["Feature"], importance_df["Importance"], color='teal')
        ax3.set_xlabel("Mean Importance Score")
        ax3.set_title("Permutation Importance (Precomputed)")
        st.pyplot(fig3)

try:
    model = pipeline.named_steps["model"]  # This is your VotingClassifier

    if isinstance(model, VotingClassifier):
        # Attempt to use RandomForest or fallback to LogisticRegression
        base_models = dict(model.named_estimators_)

        if "rf" in base_models:
            base_model = base_models["rf"]
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(df_input)
        elif "lr" in base_models:
            base_model = base_models["lr"]
            explainer = shap.LinearExplainer(base_model, df_input)
            shap_values = explainer.shap_values(df_input)
        else:
            raise ValueError("No SHAP-compatible estimator found in VotingClassifier.")

    elif hasattr(model, "estimators_"):  # e.g., RandomForest directly
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)

    else:
        raise ValueError("Unsupported model for SHAP explanation.")

    st.write("### 🧠 SHAP Explanation")
    shap.summary_plot(shap_values[1], df_input, plot_type="bar")  # For classification
    st.pyplot()

except Exception as e:
    st.warning(f"⚠️ SHAP explanation could not be generated.\n\nError: {e}")

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
