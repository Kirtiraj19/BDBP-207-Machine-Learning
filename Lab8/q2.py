# Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_data():
    data = load_breast_cancer()     # load sklearn dataset

    X = data.data                   # features
    y = data.target                 # target (already numeric: 0,1)

    return X, y

def split_data(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    return X_train, X_test, y_train, y_test

def lasso_logistic_regression(X_train, y_train):

    lasso_model = LogisticRegression(solver='saga',l1_ratio=1,C=1,max_iter=500)

    lasso_model.fit(X_train, y_train)

    return lasso_model

def ridge_logistic_regression(X_train, y_train):

    ridge_model = LogisticRegression( solver='saga',l1_ratio=0,C=1,max_iter=5000)

    ridge_model.fit(X_train, y_train)

    return ridge_model

def prediction(lasso_model,ridge_model,X_test, y_test):

    prediction_lasso = lasso_model.predict(X_test)
    prediction_ridge = ridge_model.predict(X_test)

    accuracy_lasso = accuracy_score(y_test, prediction_lasso)
    accuracy_ridge = accuracy_score(y_test, prediction_ridge)

    print("Lasso Accuracy:", accuracy_lasso)
    print("Ridge Accuracy:", accuracy_ridge)

    print("Lasso Non-zero Coeff:", np.sum(lasso_model.coef_ != 0))
    print("Ridge Non-zero Coeff:", np.sum(ridge_model.coef_ != 0))

def main():

    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    lasso_model = lasso_logistic_regression(X_train, y_train)
    ridge_model = ridge_logistic_regression(X_train, y_train)
    prediction(lasso_model,ridge_model, X_test, y_test)



if __name__ == '__main__':
    main()