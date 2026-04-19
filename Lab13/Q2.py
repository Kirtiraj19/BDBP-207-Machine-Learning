# Implement bagging regressor without using scikit-learn

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Load data
data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = []

# Create 10 bootstrap models
for i in range(10):
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample = X_train[idx]
    y_sample = y_train[idx]

    model = DecisionTreeRegressor()
    model.fit(X_sample, y_sample)
    models.append(model)

# Predict average
predictions = []

for model in models:
    predictions.append(model.predict(X_test))

final_pred = np.mean(predictions, axis=0)

mse = np.mean((y_test - final_pred) ** 2)

print("MSE:", mse)