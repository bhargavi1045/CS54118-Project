import numpy as np

class ManualNN:
    """
    Feed-forward neural network (1 hidden layer) for binary classification.
    Supports optional class weighting for imbalanced datasets.
    """
    
    def __init__(self, input_dim, hidden_dim=128, lr=0.01, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Initialize weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, 1))

    # --- Activations ---
    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # --- Forward pass ---
    def forward(self, X):
        self.X = X
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    # --- Loss (binary cross-entropy with optional class weights) ---
    def compute_loss(self, y_true, class_weights=None):
        eps = 1e-9
        y_true = y_true.reshape(-1, 1)
        p = np.clip(self.a2, eps, 1 - eps)
        if class_weights is None:
            loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        else:
            weights = np.where(y_true == 1, class_weights[1], class_weights[0])
            loss = -np.mean(weights * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        return loss

    # --- Backprop + gradient descent ---
    def backward_and_update(self, y_true, class_weights=None):
        m = self.X.shape[0]
        y = y_true.reshape(-1, 1)
        a2 = self.a2

        # Apply class weights if given
        if class_weights is not None:
            w = np.where(y == 1, class_weights[1], class_weights[0]).reshape(-1, 1)
        else:
            w = 1.0

        dz2 = w * (a2 - y) / m
        dW2 = self.a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * self.relu_grad(self.z1)
        dW1 = self.X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # --- Train loop ---
    def fit(self, X_train, y_train, epochs=80, batch_size=32,
            X_val=None, y_val=None, class_weights=None, verbose=True):
        n = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for i in range(0, n, batch_size):
                X_batch = X_shuf[i:i + batch_size]
                y_batch = y_shuf[i:i + batch_size]
                self.forward(X_batch)
                self.backward_and_update(y_batch, class_weights=class_weights)

            # Epoch metrics
            y_pred_train = self.predict(X_train)
            train_acc = np.mean(y_pred_train == y_train)
            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                val_acc = np.mean(y_pred_val == y_val)
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - train_acc: {train_acc:.4f}")

    # --- Predict ---
    def predict_proba(self, X):
        return self.forward(X).reshape(-1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # --- Save/load weights ---
    def save_weights(self, path):
        np.savez_compressed(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )

    def load_weights(self, path):
        npz = np.load(path, allow_pickle=True)
        self.W1 = npz['W1']
        self.b1 = npz['b1']
        self.W2 = npz['W2']
        self.b2 = npz['b2']
        self.input_dim = int(npz['input_dim'])
        self.hidden_dim = int(npz['hidden_dim'])
