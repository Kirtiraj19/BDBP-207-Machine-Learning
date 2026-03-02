# Use your implementation and train ML models for both californiahousing and simulated datasets
# and compare your results with the scikit-learn models.

#read csv file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

def load_data():
    [X, y] = fetch_california_housing(return_X_y=True)
    return X, y
X, y = load_data()
# X and y
X_train = X[:16512]     # use numpy slicing
y_train = y[:16512]

X_test = X[16513:]
y_test = y[16513:]

# reshape y
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# ================= SCALING =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= ADD BIAS =================
m_train = X_train.shape[0]
m_test = X_test.shape[0]

X_train = np.concatenate((np.ones((m_train, 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((m_test, 1)), X_test), axis=1)

# ================= INITIALIZE THETA =================
theta = np.zeros((X_train.shape[1], 1))


# ================= HYPOTHESIS =================
def hypothesis(X, theta):
    return np.dot(X, theta)


# ================= COST FUNCTION =================
def cost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((h - y) ** 2)


# ================= GRADIENT DESCENT =================
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)

    for i in range(iterations):
        h = hypothesis(X, theta)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta = theta - alpha * gradient

        if i % 100 == 0:
            print("Iteration", i, "Cost =", cost(X, y, theta))

    return theta


# ================= TRAIN CUSTOM MODEL =================
alpha = 0.01
iterations = 3000

theta_final = gradient_descent(X_train, y_train, theta, alpha, iterations)

# ================= TEST CUSTOM MODEL =================
y_pred_custom = hypothesis(X_test, theta_final)
r2_custom = r2_score(y_test, y_pred_custom)

print("\nCustom Model R2 Score =", r2_custom)

# ================= SCIKIT-LEARN MODEL =================
model = LinearRegression()
model.fit(X_train[:, 1:], y_train)  # remove bias column

y_pred_sklearn = model.predict(X_test[:, 1:])
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("Scikit-Learn R2 Score =", r2_sklearn)