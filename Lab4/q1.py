#Implement gradient descent algorithm from scratch using Python
import numpy as np

def gradient_descent(X_with_bias_train,y_np_train,h,alpha,theta_matrix):
    thet=[]
    for i in range(len(X_with_bias_train[0])):
        s=0
        for j in range(len(y_np_train)):
            s+=(h[j][0]-y_np_train[j][0])*X_with_bias_train[j][i]    # summation

        thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta

    theta=np.array(thet)
    return theta.reshape(-1,1)















# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# # import warnings
# # warnings.filterwarnings('ignore')
# class GradientDescentRegressor:
#     def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
#         self.learning_rate = learning_rate
#         self.n_iterations = n_iterations
#         self.verbose = verbose
#         self.weights = None  # Will store feature weights
#         self.bias = None  # Will store the intercept
#         self.cost_history = []  # Track how cost decreases over time
#
#     def _compute_cost(self, X, y, weights, bias):
#         m = len(y)  # number of samples
#     # Make predictions: y_pred = X·w + b
#         predictions = X.dot(weights) + bias
#
#     # Calculate squared errors and average them
#         cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
#
#         return cost
#
#     def _compute_gradients(self, X, y, weights, bias):
#         m = len(y)
#
#         # Calculate predictions
#         predictions = X.dot(weights) + bias
#
#         # Calculate error
#         error = predictions - y
#
#         # Gradient for weights (vectorized)
#         # X.T means transpose - flips rows and columns
#         dw = (1 / m) * X.T.dot(error)
#
#         # Gradient for bias (just average of errors)
#         db = (1 / m) * np.sum(error)
#
#         return dw, db
#
#     def fit(self, X, y):
#         # Convert to numpy arrays
#         X = np.array(X)
#         y = np.array(y)
#
#         # Get dimensions
#         n_samples, n_features = X.shape
#
#         # Initialize parameters to zero
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         print(f"\nStarting training with {n_samples} samples and {n_features} features")
#         print(f"Learning rate: {self.learning_rate}")
#         print(f"Iterations: {self.n_iterations}\n")
#
#         # GRADIENT DESCENT LOOP
#         for i in range(self.n_iterations):
#             # Step 1: Compute gradients
#             dw, db = self._compute_gradients(X, y, self.weights, self.bias)
#
#             # Step 2: Update weights using gradient descent formula
#             # New weight = Old weight - learning_rate × gradient
#             self.weights = self.weights - self.learning_rate * dw
#             self.bias = self.bias - self.learning_rate * db
#
#             # Step 3: Calculate and store cost to track progress
#             cost = self._compute_cost(X, y, self.weights, self.bias)
#             self.cost_history.append(cost)
#
#             # Print progress every 100 iterations
#             if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
#                 print(f"Iteration {i:4d}: Cost = {cost:.6f}")
#
#         print(f"\n✓ Training completed!")
#         print(f"  Final cost: {self.cost_history[-1]:.6f}")
#         print(f"  Learned {n_features} weights + 1 bias term")
#
#         return self