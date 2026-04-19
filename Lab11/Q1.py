#Implement decision tree classifier without using scikit-learn using the iris dataset.
#Fetch the iris dataset from scikit-learn library.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# --------------------------
# Node class
# --------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
# --------------------------
# Decision Tree Classifier
# --------------------------
class DecisionTree:

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):

        num_samples, num_features = X.shape

        # Stop conditions
        if len(set(y)) == 1:
            return Node(value=y[0])

        if num_samples < 2:
            return Node(value=self.majority_vote(y))

        # Find best split
        best_feature, best_thresh = self.best_split(X, y)

        if best_feature is None:
            return Node(value=self.majority_vote(y))

        # Split
        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)

        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def information_gain(self, y, left_y, right_y):

        parent_entropy = self.entropy(y)

        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left/n) * self.entropy(left_y) + \
                        (n_right/n) * self.entropy(right_y)

        return parent_entropy - child_entropy

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

                gain = self.information_gain(
                    y,
                    y[left_idx],
                    y[right_idx]
                )

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_thresh = thresh

        return split_feature, split_thresh

    def majority_vote(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict_one(self, x, node):

        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])
# --------------------------
# Train model
# --------------------------
tree = DecisionTree()
tree.fit(X_train, y_train)

# Prediction
pred = tree.predict(X_test)
# Accuracy
accuracy = np.sum(pred == y_test) / len(y_test)
print("Predictions:", pred)
print("Actual:", y_test)
print("Accuracy:", accuracy)



#-------------------------------------
#Using Scikit learn
#-------------------------------------
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
#
# # Load dataset
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Model
# model = DecisionTreeClassifier(random_state=42)
#
# # Train
# model.fit(X_train, y_train)
#
# # Predict
# y_pred = model.predict(X_test)
#
# # Accuracy
# acc = accuracy_score(y_test, y_pred)
#
# print("Predicted:", y_pred)
# print("Actual   :", y_test)
# print("Accuracy :", acc)