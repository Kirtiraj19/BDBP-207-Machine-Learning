#Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
#Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
#What is the value of x1 at which the function value (y) is zero. What do you infer from this?

# List of x values where derivative will be checked
points = [-5, -3, 0, 3, 5]

# Loop through each x value
for x in points:

    # Derivative of y = x^2 is dy/dx = 2x
    derivative = 2 * x

    # Print x and its derivative
    print("x =", x, "Derivative =", derivative)

# When derivative = 0, function has minimum
# Here derivative is 0 at x = 0

#Output
# x = -3 Derivative = -6
# x = 0 Derivative = 0
# x = 3 Derivative = 6
# x = 5 Derivative = 10