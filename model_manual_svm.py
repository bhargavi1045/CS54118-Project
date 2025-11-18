import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

class ManualSVM:
    def __init__(self, C=1.0, max_iter=10000, seed=42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.clf = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.seed)
        self.scaler = StandardScaler(with_mean=False)  # sparse data

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X):
        """
        Approximate probability using decision function + sigmoid
        """
        X_scaled = self.scaler.transform(X)
        df = self.clf.decision_function(X_scaled)
        probs = 1 / (1 + np.exp(-df))  # sigmoid
        return probs

    def save_model(self, path):
        joblib.dump({"clf": self.clf, "scaler": self.scaler}, path)

    def load_model(self, path):
        obj = joblib.load(path)
        self.clf = obj["clf"]
        self.scaler = obj["scaler"]
