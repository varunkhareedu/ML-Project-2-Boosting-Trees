import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_loss = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_val = y[left_mask].mean() if np.any(left_mask) else 0
                right_val = y[right_mask].mean() if np.any(right_mask) else 0
                prediction = np.where(left_mask, left_val, right_val)
                loss = np.mean((y - prediction) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_val
                    self.right_value = right_val

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature <= self.threshold, self.left_value, self.right_value)


class GradientBoostingClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_pred = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        y = y.reshape(-1)
        pred = np.full(y.shape, self.initial_pred, dtype=np.float64)

        for _ in range(self.n_estimators):
            prob = self.sigmoid(pred)
            gradient = y - prob  # Gradient of log-loss
            stump = DecisionStump()
            stump.fit(X, gradient)
            update = stump.predict(X)
            pred += self.learning_rate * update
            self.models.append(stump)

    def predict_proba(self, X):
        pred = np.full((X.shape[0],), self.initial_pred, dtype=np.float64)
        for model in self.models:
            pred += self.learning_rate * model.predict(X)
        return self.sigmoid(pred)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)