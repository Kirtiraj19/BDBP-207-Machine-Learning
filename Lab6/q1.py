# K-fold cross validation. Implement for K = 10. Implement from scratch, then,
# use scikit-learn methods.

##______Implementation of k-fold from scratch_________________

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# load dataset
data = load_breast_cancer()

# X contains input features
X = data.data

# y contains target labels (0 or 1)
y = data.target

k = 10                              # number of folds

n = len(X)                          # total number of samples

fold_size = n // k                  # size of each fold

accuracies = []                     # list to store accuracy of each fold

# loop over all folds
for i in range(k):

    start = i * fold_size           # starting index of test fold
    end = start + fold_size         # ending index of test fold

    # test data for this fold
    X_test = X[start:end]
    y_test = y[start:end]

    # training data = all data except test fold
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    # create logistic regression model
    model = LogisticRegression(max_iter=10000)

    # train the model on training data
    model.fit(X_train, y_train)

    # calculate accuracy on test data
    accuracy = model.score(X_test, y_test)

    # store accuracy
    accuracies.append(accuracy)

# print accuracy of each fold
print("Accuracy of each fold:", accuracies)

# compute average accuracy
print("Average accuracy:", np.mean(accuracies))



##_____Using scikit learn_______________________________

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load dataset
data = load_breast_cancer()

# features
X = data.data
# labels
y = data.target

# create logistic regression model
model = LogisticRegression(max_iter=10000)

# perform 10-fold cross validation
scores = cross_val_score(model, X, y, cv=10)

# print accuracy of each fold
print("Accuracy for each fold:", scores)

# print average accuracy
print("Average accuracy:", scores.mean())
