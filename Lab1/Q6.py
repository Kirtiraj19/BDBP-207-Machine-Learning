#Implement y = 2x1 + 3x2 + 3x3 + 4, where x1, x2 and x3 are three independent
#variables. Compute the gradient of y at a few points and print the values.

# Gradient is a vector of partial derivatives
# Partial derivative of y w.r.t x1 = 2
# Partial derivative of y w.r.t x2 = 3
# Partial derivative of y w.r.t x3 = 3
gradient = [2, 3, 3]

# Some sample points (x1, x2, x3)
points = [(0, 0, 0), (1, 2, 3), (4, 5, 6)]

# Loop through each point
for p in points:

    # Print the gradient at each point
    # For linear functions, gradient is SAME everywhere
    print("At point", p, "gradient =", gradient)


#Output
# At point (0, 0, 0) gradient = [2, 3, 3]
# At point (1, 2, 3) gradient = [2, 3, 3]
# At point (4, 5, 6) gradient = [2, 3, 3]