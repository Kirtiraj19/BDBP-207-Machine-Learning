#Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset.
#Fetch the dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Tree Node
# ---------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# ---------------------------
# Decision Tree Regressor
# ---------------------------
class DecisionTreeRegressor:

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape

        # Stop conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feature, best_thresh = self.best_split(X, y)

        if best_feature is None:
            return Node(value=np.mean(y))

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def variance_reduction(self, y, left_y, right_y):

        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        parent_mse = self.mse(y)

        child_mse = (n_left/n) * self.mse(left_y) + \
                    (n_right/n) * self.mse(right_y)

        return parent_mse - child_mse

    def best_split(self, X, y):

        best_gain = -1
        split_feature = None
        split_thresh = None

        n_samples, n_features = X.shape

        for feature in range(n_features):

            thresholds = np.unique(X[:, feature])

            for thresh in thresholds:

                left_idx = X[:, feature] <= thresh
                right_idx = X[:, feature] > thresh

                gain = self.variance_reduction(
                    y,
                    y[left_idx],
                    y[right_idx]
                )

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_thresh = thresh

        return split_feature, split_thresh

    def predict_one(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

# ---------------------------
# Train model
# ---------------------------
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)

# Prediction
pred = tree.predict(X_test)

# Evaluation
mse = np.mean((y_test - pred) ** 2)

print("Predicted Values:", pred[:10])
print("Actual Values   :", y_test[:10])
print("Mean Squared Error:", mse)




#-------------------------------------
#Using Scikit learn
#-------------------------------------

# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error
#
# # Load dataset
# data = load_diabetes()
# X = data.data
# y = data.target
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Model
# model = DecisionTreeRegressor(random_state=42)
#
# # Train
# model.fit(X_train, y_train)
#
# # Predict
# y_pred = model.predict(X_test)
#
# # MSE
# mse = mean_squared_error(y_test, y_pred)
#
# print("Predicted Values:", y_pred[:10])
# print("Actual Values   :", y_test[:10])
# print("Mean Squared Error:", mse)
