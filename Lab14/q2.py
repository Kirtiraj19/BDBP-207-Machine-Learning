# Implement Adaboost classifier without using scikit-learn. Use the Iris dataset.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to binary classification
# class 0 = -1, class 1 = +1
X = X[y != 2]
y = y[y != 2]
y = np.where(y == 0, -1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)

        if self.polarity == 1:
            pred[X[:, self.feature] < self.threshold] = -1
        else:
            pred[X[:, self.feature] > self.threshold] = -1

        return pred


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])

                for threshold in thresholds:
                    p = 1
                    pred = np.ones(n_samples)
                    pred[X[:, feature] < threshold] = -1

                    error = sum(w[y != pred])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        stump.polarity = p
                        stump.threshold = threshold
                        stump.feature = feature
                        min_error = error

            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

            predictions = stump.predict(X)

            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)

            self.models.append(stump)

    def predict(self, X):
        clf_preds = [model.alpha * model.predict(X) for model in self.models]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)


# Train model
model = AdaBoost(n_estimators=10)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))