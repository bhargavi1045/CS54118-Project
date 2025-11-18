import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_manual import ManualNN
from model_manual_nb import ManualNB
from model_manual_svm import ManualSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import re

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="SPAM or HAM", layout="wide")

# -----------------------------------------------------
# Minimal CSS for beauty
# -----------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: white; /* changed from blue */
        margin-top: -10px;
        margin-bottom: 10px; /* reduced to remove white block */
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Cleaning
# -----------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(path):
    df = pd.read_csv(path, sep="\t", names=["label","message"], encoding="utf-8").reset_index(drop=False).rename(columns={"index":"orig_index"})
    df['clean'] = df['message'].apply(clean_text)
    df['label_num'] = df['label'].map({'ham':0,'spam':1})
    return df

# -----------------------------------------------------
# Ensure folders
# -----------------------------------------------------
os.makedirs("models/nn", exist_ok=True)
os.makedirs("models/nb", exist_ok=True)
os.makedirs("models/svm", exist_ok=True)

# -----------------------------------------------------
# Load Models
# -----------------------------------------------------
nb = joblib.load("models/nb/manual_nb_model.joblib")

vect = joblib.load("models/nn/tfidf_vectorizer.joblib")
nn = ManualNN(input_dim=vect.transform(["test"]).shape[1])
nn.load_weights("models/nn/nn_weights.npz")

svm_vect_path = "models/svm/tfidf_vectorizer.joblib"
svm_model_path = "models/svm/manual_svm_model.joblib"

svm_available = os.path.exists(svm_model_path) and os.path.exists(svm_vect_path)
if svm_available:
    vect_svm = joblib.load(svm_vect_path)
    svm = ManualSVM()
    svm.load_model(svm_model_path)
else:
    vect_svm = None
    svm = None

# -----------------------------------------------------
# NAVIGATION (fixed)
# -----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Predict SMS", "Compare Models"])

# -----------------------------------------------------
# PAGE 1 — PREDICT
# -----------------------------------------------------
if page == "Predict SMS":

    # Updated title
    st.markdown('<div class="main-title">SMS Classifier - Spam/Ham</div>', unsafe_allow_html=True)

    model_options = ["Naive Bayes", "Neural Network"]
    if svm_available:
        model_options.append("SVM")

    model_choice = st.radio("Select model", model_options)
    sms_input = st.text_area("Enter SMS here:")

    if st.button("Predict"):
        if not sms_input.strip():
            st.warning("Please enter a message.")
        else:
            cleaned = clean_text(sms_input)

            if model_choice == "Naive Bayes":
                pred = nb.predict(cleaned)
                prob = nb.predict_proba(cleaned)
            elif model_choice == "Neural Network":
                x = vect.transform([cleaned]).toarray().astype(np.float32)
                pred = nn.predict(x)[0]
                prob = nn.predict_proba(x)[0]
            else:
                x = vect_svm.transform([cleaned])
                pred = svm.predict(x)[0]
                prob = svm.predict_proba(x)[0]

            label = "Spam" if pred == 1 else "Ham"
            color = "#FF4B4B" if label == "Spam" else "#4CAF50"

            # ---------- Compact prediction block ----------
            st.markdown(
                f"""
                <div style='background:{color}; padding:5px 10px; border-radius:12px; text-align:center; color:white; display:inline-block;'>
                    <span style='font-size:20px; font-weight:bold;'>{label}</span>
                    <span style='margin-left:10px; font-size:14px;'>(Spam Probability: {prob:.2f})</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------------------------------
# PAGE 2 — COMPARE MODELS
# -----------------------------------------------------
elif page == "Compare Models":

    st.title("Model Comparison")

    df = load_data("data/SMSSpamCollection")
    y_true = df['label_num'].values

    y_pred_nb = np.array([nb.predict(m) for m in df['clean']])
    y_prob_nb = np.array([nb.predict_proba(m) for m in df['clean']])

    X_nn = vect.transform(df['clean']).toarray().astype(np.float32)
    y_pred_nn = nn.predict(X_nn)
    y_prob_nn = nn.predict_proba(X_nn)

    if svm_available:
        X_svm = vect_svm.transform(df['clean'])
        y_pred_svm = svm.predict(X_svm)
        y_prob_svm = svm.predict_proba(X_svm)

    def metrics(y, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1-score": f1_score(y, y_pred),
            "ROC-AUC": roc_auc_score(y, y_prob)
        }

    metrics_nb = metrics(y_true, y_pred_nb, y_prob_nb)
    metrics_nn = metrics(y_true, y_pred_nn, y_prob_nn)

    if svm_available:
        metrics_svm = metrics(y_true, y_pred_svm, y_prob_svm)
        df_metrics = pd.DataFrame([metrics_nb, metrics_nn, metrics_svm], index=["Naive Bayes", "Neural Network", "SVM"])
    else:
        df_metrics = pd.DataFrame([metrics_nb, metrics_nn], index=["Naive Bayes", "Neural Network"])

    st.subheader("Metrics Table")
    st.dataframe(df_metrics)

    st.subheader("Confusion Matrices")

    # Define labels
    labels = ["Ham", "Spam"]

    cm_nb = confusion_matrix(y_true, y_pred_nb)
    cm_nn = confusion_matrix(y_true, y_pred_nn)

    if svm_available:
        cm_svm = confusion_matrix(y_true, y_pred_svm)
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax[0])
        ax[0].set_title("Naive Bayes", fontsize=25, pad=20)

        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=ax[1])
        ax[1].set_title("Neural Network", fontsize=25, pad=20)

        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels, ax=ax[2])
        ax[2].set_title("SVM", fontsize=25, pad=20)

    else:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax[0])
        ax[0].set_title("Naive Bayes")

        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels, yticklabels=labels, ax=ax[1])
        ax[1].set_title("Neural Network")

    st.pyplot(fig)

    # ---------- Misclassified examples ----------
    st.subheader("Misclassified Examples")
    def show_misclassified(y_true, y_pred, model_name):
        mask = y_true != y_pred
        mis_df = df[mask][['message', 'label']].copy()
        mis_df['Predicted'] = y_pred[mask]
        # Map 0/1 back to Ham/Spam
        mis_df['Predicted'] = mis_df['Predicted'].map({0:'Ham', 1:'Spam'})
        st.markdown(f"**{model_name}**")
        st.dataframe(mis_df)

    show_misclassified(y_true, y_pred_nb, "Naive Bayes")
    show_misclassified(y_true, y_pred_nn, "Neural Network")
    if svm_available:
        show_misclassified(y_true, y_pred_svm, "SVM")
