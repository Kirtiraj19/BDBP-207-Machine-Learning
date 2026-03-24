#Data normalization - scale the values between 0 and 1. Implement code from scratch.

from sklearn.datasets import load_breast_cancer

#load dataset
data = load_breast_cancer()
X = data.data
y = data.target
split = int(0.8 * len(X))
X_train , X_test = X[:split], X[split:]
y_train , y_test = y[:split], y[split:]

# Normalisation (fit on train only)
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

X_train_norm = (X_train - X_min) / (X_max - X_min)
X_test_norm  = (X_test  - X_min) / (X_max - X_min)

print("Min:", X_train_norm.min())
print("Max:", X_train_norm.max())
