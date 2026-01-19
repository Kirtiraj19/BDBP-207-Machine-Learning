import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def load_data():
    [X, y] = fetch_california_housing(return_X_y=True)
    return X, y
def main():

    X,y = load_data()
    print("X.shape is :", X.shape)
    print("y.shape is :", y.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=999)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scale=scaler.transform(X_train)
    X_test_scale=scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scale, y_train)
    y_pred = model.predict(X_test_scale)
    r2 = r2_score(y_test, y_pred)
    print("r2 is :",r2)

    print("Done!")

if __name__ == "__main__":
    main()




