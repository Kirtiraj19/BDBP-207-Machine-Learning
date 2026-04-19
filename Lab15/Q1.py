# Implement Gradient Boost Regression and Classification using scikit-learn. Use the Boston housing dataset from the ISLP package for the regression problem and weekly dataset from the ISLP package and use Direction as the target variable for the classification.
#-------------------------------------------------------------
#Gradient Boost Regression using Boston Housing Dataset (ISLP)
#-------------------------------------------------------------

import pandas as pd
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston dataset
Boston = load_data("Boston")

X = Boston.drop("medv", axis=1)
y = Boston["medv"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))




#---------------------------------------------------------
#Gradient Boost Classification using Weekly Dataset (ISLP)
#---------------------------------------------------------

from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Weekly dataset
Weekly = load_data("Weekly")

# Remove Today column to avoid data leakage
X = Weekly.drop(["Direction", "Today"], axis=1)

# Target
y = Weekly["Direction"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Result
print("\nClassification Results")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))