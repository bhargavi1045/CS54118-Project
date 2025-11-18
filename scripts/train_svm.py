import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from model_manual_svm import ManualSVM

# Ensure folder exists
os.makedirs("models/svm", exist_ok=True)

# 1️⃣ Load dataset
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
df['label_num'] = df['label'].map({'ham':0,'spam':1})

# 2️⃣ Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])
    return text

df['clean'] = df['message'].apply(clean_text)

# 3️⃣ Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean'], df['label_num'], test_size=0.2, stratify=df['label_num'], random_state=42
)

# 4️⃣ TF-IDF
vect = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
X_train = vect.fit_transform(X_train_text)
X_test  = vect.transform(X_test_text)

# 5️⃣ Train ManualSVM
svm = ManualSVM(C=1.0, max_iter=10000)
svm.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

# 7️⃣ Save model and vectorizer
svm.save_model("models/svm/manual_svm_model.joblib")
joblib.dump(vect, "models/svm/tfidf_vectorizer.joblib")
