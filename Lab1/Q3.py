#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
import matplotlib

# Create an empty list to store x values
x_values = []

# Create an empty list to store y values
y_values = []

# Start x from -10
x = -10

# Step size: how much x increases in each iteration
step = 0.2

# Loop continues as long as x is less than or equal to 10
while x <= 10:

    # Calculate y using the quadratic equation:
    # y = 2x^2 + 3x + 4
    # x ** 2 means x squared
    y = 2 * (x ** 2) + 3 * x + 4

    # Store the current x value in the list
    x_values.append(x)

    # Store the corresponding y value in the list
    y_values.append(y)

    # Increase x by step size to move to next point
    x += step

# Print first 5 (x, y) pairs to verify the result
for i in range(5):
    print("x =", x_values[i], "y =", y_values[i])


#Output
# x = -9.8 y = 166.68000000000004
# x = -9.600000000000001 y = 159.52000000000004
# x = -9.400000000000002 y = 152.52000000000007
# x = -9.200000000000003 y = 145.68000000000012