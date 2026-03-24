#Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load dataset
data = load_breast_cancer()

# features and labels
X = data.data
y = data.target

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: compute mean and standard deviation from TRAINING data
mu = np.mean(X_train, axis=0)        # mean of each feature
sigma = np.std(X_train, axis=0)      # std deviation of each feature

# avoid division by zero
sigma[sigma == 0] = 1

# Step 2: apply standardization formula
# z = (x - mean) / std
X_train_std = (X_train - mu) / sigma
X_test_std = (X_test - mu) / sigma   # use same mu and sigma for test

# print first 5 rows
print(X_train_std[:5])
print(X_test_std[:5])