# Here is a linear model. y = 2x1 + 3x2 + 3x3 + 4
# The coefficients, represented as theta, is a vector given
# There are 5 samples represented as a matrix, X, given. Compute
# X theta

# Feature matrix X (5 samples, 3 features)
X = [
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
]

# Parameter vector theta (coefficients)
theta = [2, 3, 3]

# List to store output values
y = []

# Loop through each row (sample) in X
for row in X:

    # Initialize value for this sample
    value = 0

    # Loop through each feature
    for i in range(len(theta)):

        # Multiply feature value with corresponding theta
        value = value + row[i] * theta[i]

    # Store computed output
    y.append(value)

# Print final result
print("Xθ =", y)

#Output
# Xθ = [8, 6, 7, 8, 9]