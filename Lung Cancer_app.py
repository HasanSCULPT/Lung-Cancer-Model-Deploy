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

# ----------------------------
# ✅ Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="Lung Cancer Diagnostics App", layout="centered")

# Background Image
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
            "individual_entry": "Welcome To the Diagnostics Centre, please enter your medical/patient information below to predict whether patient is likely to have Lung Cancer or not",
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

# ----------------------------
# ✅ Header Section
# ----------------------------
# ✅ Title
st.image("logo.png", width=100)
st.title(f"🔬 {tr['title']}")
st.write(f"## {tr['subtitle']}")

# ----------------------------
# ✅ Sidebar Navigation
# ----------------------------

# Sidebar Navigation
page = st.sidebar.selectbox(tr['sidebar_title'], ["Prediction", "About", "Contact", "Terms"], key="page")

# ----------------------------
# ✅ Email Setup
# ----------------------------
# Email Function
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
# Pages
if page == "About":
    st.title(tr['about_title'])
    st.write(tr['about_desc'])
elif page == "Contact":
    st.title(tr['contact_title'])
    st.write("Email: support@lungdiagnosis.ai")
elif page == "Terms":
    st.title(tr['terms_title'])
    st.write(tr['terms_text'])

# ----------------------------
# ✅ Prediction Page
# ----------------------------    

# ✅ Prediction Page
if page == "Prediction":
    st.sidebar.subheader("🛠 Adjust Classification Threshold")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded_file = st.sidebar.file_uploader(tr['upload_csv'], type="csv", key="csv")
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_input.head())
        st.dataframe(df_input.tail())

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


        # ✅ Data Cleaning & Feature Alignment
        df_input = pd.get_dummies(df_input, drop_first=True)
        for col in feature_names:
            if col not in df_input:
                df_input[col] = 0
        df_input = df_input[feature_names]

        # ✅ Automatic Threshold Suggestion
        st.write("🔍 Calculating optimal threshold for high recall...")
        y_true = None  # If labels exist in data
        proba_temp = pipeline.predict_proba(df_input)[:, 1]
        best_thresh = threshold
        best_recall = 0
        if "LUNG_CANCER" in df_input.columns:  # Optional
            y_true = df_input["LUNG_CANCER"]
            for t in np.arange(0.1, 0.9, 0.01):
                y_pred_temp = (proba_temp > t).astype(int)
                recall = recall_score(y_true, y_pred_temp)
                if recall > best_recall:
                    best_recall = recall
                    best_thresh = t
            st.success(f"✅ Suggested Threshold: {best_thresh:.2f} (Recall: {best_recall:.2f})")


       
        
        # ✅ Prediction
        proba = pipeline.predict_proba(df_input)[0][1]
        prediction = (proba > threshold).astype(int)

        df_output = df_input.copy()
        df_output["Probability"] = proba
        df_output["Prediction"] = prediction

        st.write(f"### {tr['prediction_results']}")
        st.dataframe(df_output[["Probability", "Prediction"]])
        st.download_button("📥 " + tr['download_csv'], df_output.to_csv(index=False), "batch_predictions.csv", "text/csv")


        # ROC Curve
        st.write("### 🔍 ROC Curve")
        fpr, tpr, roc_thresholds = roc_curve(prediction, proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Precision-Recall Curve
        st.write("### 📈 Precision-Recall Curve")
        precision, recall, pr_thresholds = precision_recall_curve(prediction, proba)
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision, color="green")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve")
        st.pyplot(fig_pr)

        # Optimal Threshold Suggestion
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = roc_thresholds[optimal_idx]
        st.sidebar.success(f"Recommended Threshold: {optimal_threshold:.2f}")

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(proba, bins=10, edgecolor='k')
        ax.axvline(threshold, color='red', linestyle='--')
        st.pyplot(fig)

        # ✅ Precomputed Permutation Importance
        importance_data = {
            "Feature": ["SYMPTOM_SCORE", "LIFESTYLE_SCORE", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "YELLOW_FINGERS"],
            "Importance": [0.0629, 0.0371, 0.0274, 0.0258, 0.0081]
        }
        importance_df = pd.DataFrame(importance_data)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.barh(importance_df["Feature"], importance_df["Importance"], color='teal')
        ax3.set_title("Permutation Importance (Precomputed)")
        st.pyplot(fig3)
    else:
        st.info("⬅️ Upload a CSV file to start prediction")

    # ✅ Individual Prediction Form
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
            'LIFESTYLE_SCORE': [lifestyle_score]
        })

        for col in feature_names:
            if col not in row:
                row[col] = 0
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

        # SHAP Explanation
        explainer = shap.KernelExplainer(pipeline.predict_proba, np.zeros((1, len(feature_names))))
        shap_values = explainer.shap_values(row)
        st.write("### SHAP Explanation")
        shap.force_plot(explainer.expected_value[1], shap_values[1], row, matplotlib=True)
        st.pyplot()


        # ✅ Export CSV + PDF
        result_df = pd.DataFrame({"Prediction": ["Lung Cancer" if pred else "No Lung Cancer"], "Probability": [prob]})
        st.download_button(label="📥 Download CSV", data=result_df.to_csv(index=False), file_name="prediction_result.csv", mime="text/csv")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Lung Cancer Prediction Result", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Prediction: {'LUNG CANCER' if pred else 'NO LUNG CANCER'}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(label="📥 Download PDF", data=pdf_buffer, file_name="prediction_result.pdf", mime="application/pdf")



