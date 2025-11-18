import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def load_data(path="data/SMSSpamCollection", oversample=True):
    df = pd.read_csv(path, sep="\t", names=["label","message"])
    df['clean'] = df['message'].apply(clean_text)
    df['label_num'] = df['label'].map({'ham':0,'spam':1})

    if oversample:
        df_ham = df[df.label=='ham']
        df_spam = df[df.label=='spam']
        df_spam_upsampled = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
        df = pd.concat([df_ham, df_spam_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_text = df['clean'].values
    y = df['label_num'].values

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
    X = vect.fit_transform(X_text).toarray().astype('float32')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, vect
