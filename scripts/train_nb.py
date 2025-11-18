import os

# Ensure folder exists
os.makedirs("models/nb", exist_ok=True)

import pandas as pd, joblib, re
from sklearn.utils import resample
from model_manual_nb import ManualNB

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label","message"])
df['clean'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham':0,'spam':1})

df_ham = df[df.label=='ham']
df_spam = df[df.label=='spam']
df_spam_up = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
df_bal = pd.concat([df_ham, df_spam_up]).sample(frac=1, random_state=42)

nb = ManualNB(alpha=1)
nb.build_vocab(df_bal['clean'], df_bal['label_num'])
joblib.dump(nb, "models/nb/manual_nb_model.joblib")
print("Naive Bayes model saved!")
