#Compute SONAR classification results with and without data pre-processing (data normalization).
#Perform data pre-processing with your implementation and with scikit-learn methods and compare the results.

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv("sonar data.csv", header=None)

# Features and target
X = data.iloc[:, :-1].values   # convert to numpy for manual scaling
y = data.iloc[:, -1]

# Encode labels (R=0, M=1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Define 10-fold CV
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

model1 = LogisticRegression(max_iter=1000)

scores1 = cross_val_score(model1, X, y, cv=kfold, scoring='accuracy')

print("WITHOUT NORMALIZATION")
print("Mean Accuracy:", scores1.mean())
print("Std Dev:", scores1.std())


# Standardization formula:
# X_new = (X - mean) / std

# Calculate mean and std
mean = X.mean(axis=0)
std = X.std(axis=0)

# Apply normalization
X_manual = (X - mean) / std

model2 = LogisticRegression(max_iter=1000)

scores2 = cross_val_score(model2, X_manual, y, cv=kfold, scoring='accuracy')

print("\nMANUAL NORMALIZATION")
print("Mean Accuracy:", scores2.mean())
print("Std Dev:", scores2.std())

# Using pipeline to avoid data leakage

model3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

scores3 = cross_val_score(model3, X, y, cv=kfold, scoring='accuracy')

print("\nSCIKIT-LEARN NORMALIZATION")
print("Mean Accuracy:", scores3.mean())
print("Std Dev:", scores3.std())