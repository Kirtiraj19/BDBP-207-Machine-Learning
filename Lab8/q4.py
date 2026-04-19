# Step 1: Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
# Step 2: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
split = int(0.8 * len(X))
X_train , X_test = X[:split], X[split:]
y_train , y_test = y[:split], y[split:]

# Step 4: Encode target (Label Encoding)
le = LabelEncoder()
y = le.fit_transform(y)

# Step 5: Apply Ordinal Encoding (for understanding)
ordinal = OrdinalEncoder()
X_ordinal = ordinal.fit_transform(X)

# Step 6: Apply One-Hot Encoding (best for model)
onehot = OneHotEncoder(sparse_output=False)
X_onehot = onehot.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_onehot, y, test_size=0.2, random_state=42
)

# Step 8: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Prediction
y_pred = model.predict(X_test)

# Step 10: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)