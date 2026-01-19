#For a design or feature matrix,
#Compute the covariance matrix using matrix multiplications. Verify your results by using numpy library operations


import numpy as np

# Given matrix X
X = np.array([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]])

#Compute mean of each feature (column-wise)
mean = np.mean(X, axis=0)

#Mean-center the data
X_c = X - mean

#Compute covariance matrix using formula
n = X.shape[0]  # number of samples
cov_matrix = (X_c.T @ X_c) / (n - 1)

print("Covariance Matrix:")
print(cov_matrix)


#OUTPUT
# Covariance Matrix:
# [[ 0.7  -0.25 -0.25]
#  [-0.25  0.5  -0.25]
#  [-0.25 -0.25  0.5 ]]
