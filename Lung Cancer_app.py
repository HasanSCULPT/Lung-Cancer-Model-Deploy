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
# Sample background data (hardcoded or load from CSV in your repo)
# ✅ Background sample for SHAP & Permutation
background_data = {
    "SYMPTOM_SCORE": [3, 5, 2, 4],
    "LIFESTYLE_SCORE": [1, 2, 1, 3],
    "SHORTNESS OF BREATH": [1, 0, 1, 1],
    "SWALLOWING DIFFICULTY": [0, 1, 0, 1],
    "ALCOHOL CONSUMING": [1, 0, 1, 0],
    "ANXIETY": [0, 1, 0, 1],
    "COUGHING": [1, 1, 0, 1],
    "WHEEZING": [1, 1, 1, 0],
    "SMOKING": [1, 0, 1, 0],
    "GENDER": [1, 0, 1, 1],
    "AGE_GROUP_Senior": [0, 1, 0, 1],
    "AGE": [55, 65, 45, 60],
    "YELLOW_FINGERS": [1, 0, 1, 0],
    "PEER_PRESSURE": [0, 1, 0, 1],
    "CHEST PAIN": [1, 0, 1, 0],
    "LIFESTYLE_RISK": [2, 3, 1, 4],
    "ALLERGY": [0, 1, 0, 1],
    "FATIGUE": [1, 0, 1, 0],
    "AGE_GROUP_Middle-aged": [1, 0, 1, 0],
    "CHRONIC DISEASE": [0, 1, 0, 1]
}

# ✅ Expected Features
expected_features = [
    "AGE", "GENDER", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
    "SYMPTOM_SCORE", "LIFESTYLE_SCORE", "LIFESTYLE_RISK", "AGE_GROUP_Senior", "AGE_GROUP_Middle-aged"
]

# ✅ Static Precomputed Permutation Importance
importance_data = {
    "Feature": [
        "SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY", "ALCOHOL CONSUMING", "ANXIETY",
        "COUGHING", "WHEEZING", "SMOKING", "GENDER", "AGE_GROUP_Senior",
        "AGE", "YELLOW_FINGERS", "PEER_PRESSURE", "CHEST PAIN",
        "LIFESTYLE_RISK", "ALLERGY", "FATIGUE", "AGE_GROUP_Middle-aged", "CHRONIC DISEASE"
    ],
    "Importance": [
        6.29e-02, 3.70e-02, 2.74e-02, 2.58e-02, 2.41e-02,
        2.41e-02, 2.09e-02, 1.93e-02, 1.93e-02, 1.12e-02,
        9.67e-03, 9.67e-03, 8.06e-03, 8.06e-03, 4.83e-03,
        1.61e-03, 0.0, 0.0, 0.0, 0.0
    ]
}




# ✅ Language Translations
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
            "individual_entry": "Please enter your medical/patient information below to predict whether you're likely to have Lung Cancer or not.",
            "about_title": "📘 About Us",
            "about_desc": """This app is developed by HasanSCULPT to assist in preliminary lung cancer risk prediction using ensemble machine learning based on symptomatic analytics and lifestyle.
            
This Diagnostic application allows for:
- Individual Prediction + Batch CSV upload with validation & cleaning
- Confidence chart for individual predictions
- Toggle for SHAP or Permutation Importance

Important: For accurate batch predictions, datasets must be cleaned and features properly encoded (1 = Lung Cancer, 0 = No Lung Cancer). Gender should also use numeric identifiers instead of text."""
            ,
            "contact_title": "📧 Contact Us",
            "terms_title": "📜 Terms & Conditions",
            "terms_text": "Disclaimer: This tool is for educational and diagnostic support only. Not a substitute for professional medical advice."
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
            "individual_entry": "Veuillez entrer les informations médicales du patient ci-dessous pour prédire la probabilité d'un cancer du poumon.",
            "about_title": "📘 À propos de nous",
            "about_desc": """Cette application, développée par HasanSCULPT, facilite la prédiction préliminaire du risque de cancer du poumon grâce à l'apprentissage automatique d'ensemble, basé sur l'analyse des symptômes et le mode de vie."""
            ,
            "contact_title": "📧 Contactez-nous",
            "terms_title": "📜 Conditions générales",
            "terms_text": "Avertissement : Cet outil est uniquement destiné à des fins éducatives et diagnostiques. Il ne remplace pas un avis médical professionnel certifié."
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
            "individual_entry": "Введите информацию о пациенте ниже для прогноза риска рака легких.",
            "about_title": "📘 О нас",
            "about_desc": "Это приложение разработано компанией HasanSCULPT для прогнозирования риска рака легких.",
            "contact_title": "📧 Связаться с нами",
            "terms_title": "📜 Условия использования",
            "terms_text": "Отказ от ответственности: данный инструмент предназначен исключительно для образовательных и диагностических целей."
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
            "individual_entry": "يرجى إدخال بياناتك الطبية أدناه للتنبؤ بخطر الإصابة بسرطان الرئة.",
            "about_title": "📘 معلومات عنا",
            "about_desc": "تم تطوير هذا التطبيق للمساعدة في التنبؤ بمخاطر الإصابة بسرطان الرئة باستخدام التعلم الآلي.",
            "contact_title": "📧 تواصل معنا",
            "terms_title": "📜 الشروط والأحكام",
            "terms_text": "إخلاء مسؤولية: هذه الأداة مخصصة للأغراض التعليمية والدعم التشخيصي فقط."
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
            "individual_entry": "Введіть дані пацієнта для прогнозу ризику раку легень.",
            "about_title": "📘 Про нас",
            "about_desc": "Цей додаток допомагає прогнозувати ризик раку легень за допомогою ансамблевого машинного навчання.",
            "contact_title": "📧 Зв'язатися з нами",
            "terms_title": "📜 Умови використання",
            "terms_text": "Цей інструмент призначений лише для освітньої та діагностичної підтримки."
        }
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
        

        # ✅ Automatic Threshold Suggestion
       
         # Plot
        sorted_idx = result.importances_mean.argsort()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(X_background.columns[sorted_idx], result.importances_mean[sorted_idx])
        ax.set_title("Permutation Importance (Based on Sample)")
        plt.tight_layout()
        st.pyplot(fig)    
         
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

        # ✅ Permutation Importance Toggle
        if st.checkbox("Show Permutation Importance"):
            try:
                result = permutation_importance(pipeline, df_input, [pred], n_repeats=5, random_state=42)
                sorted_idx = result.importances_mean.argsort()[::-1]
                fig2, ax2 = plt.subplots()
                ax2.barh(np.array(expected_features)[sorted_idx], result.importances_mean[sorted_idx])
                st.pyplot(fig2)
            except:
                st.warning("Live calculation failed. Showing static importance chart.")
                fig3, ax3 = plt.subplots()
                ax3.barh(importance_data["Feature"], importance_data["Importance"])
        
        

        # ✅ Export PDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
        pdf.cell(200,10,txt="Lung Cancer Prediction Result",ln=True,align='C')
        pdf.cell(200,10,txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}",ln=True)
        pdf.cell(200,10,txt=f"Probability: {prob:.2f}",ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name="prediction_result.pdf", mime="application/pdf")

