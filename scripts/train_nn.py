import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model_manual import ManualNN

# -----------------------------
# Ensure folder exists
os.makedirs("models/nn", exist_ok=True)

# -----------------------------
# Cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+"," ",text)
    text = re.sub(r"[^a-z\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# -----------------------------
# Load dataset
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label","message"])
df['clean'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham':0,'spam':1})

# Balance classes
df_ham = df[df.label=='ham']
df_spam = df[df.label=='spam']
df_spam_up = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
df_bal = pd.concat([df_ham, df_spam_up]).sample(frac=1, random_state=42)

X_text = df_bal['clean'].values
y = df_bal['label_num'].values

# -----------------------------
# TF-IDF vectorization
vect = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
X = vect.fit_transform(X_text).toarray().astype(np.float32)

# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Compute class weights
n_samples = len(y_train)
class_weights = {0: n_samples/(2*(n_samples-sum(y_train))), 1: n_samples/(2*sum(y_train))}

# -----------------------------
# Create Neural Network object
nn = ManualNN(input_dim=X.shape[1], hidden_dim=128, lr=0.01)

# Train
nn.fit(X_train, y_train, epochs=100, batch_size=32, X_val=X_val, y_val=y_val, class_weights=class_weights)

# Save model + vectorizer
nn.save_weights("models/nn/nn_weights.npz")
joblib.dump(vect, "models/nn/tfidf_vectorizer.joblib")

# -----------------------------
# Evaluate on test
y_pred = nn.predict(X_test)
y_prob = nn.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))
print("F1-score:", f1_score(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test,y_prob))
