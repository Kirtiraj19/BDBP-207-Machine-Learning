# Plot the data points and the obtained regression line from all three approaches
# and compare the outcome.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1 : Create sample data

X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

# Step 2 : Gradient Descent

w = 0      # weight (slope)
b = 0      # bias (intercept)

learning_rate = 0.01
iterations = 1000

for i in range(iterations):

    # prediction using model y = wx + b
    y_pred = w*X + b

    # error between predicted and actual values
    error = y_pred - y

    # compute gradients
    dw = np.mean(error * X)   # derivative w.r.t weight
    db = np.mean(error)       # derivative w.r.t bias

    # update parameters using gradient descent rule
    w = w - learning_rate * dw
    b = b - learning_rate * db


# predictions using gradient descent model
y_pred_gd = w*X + b

# Step 3 : Sklearn Linear Regression

# sklearn expects 2D input
X_sklearn = X.reshape(-1,1)

model = LinearRegression()

# train model
model.fit(X_sklearn, y)

# predictions using sklearn
y_pred_sklearn = model.predict(X_sklearn)


# Step 4 : Plot comparison

plt.scatter(X,y,label="Original Data")

plt.plot(X,y_pred_gd,label="Gradient Descent")

plt.plot(X,y_pred_sklearn,label="Scikit-learn")

plt.legend()

plt.show()