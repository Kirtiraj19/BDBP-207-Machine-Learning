# Implement Gaussian PDF - mean = 0, sigma = 15 in the
# range[start=-100, stop=100, num=100]
# Import math library for sqrt() and exp()
import math

# Lists to store x and Gaussian y values
x_values = []
y_values = []

# Start x from -100
x = -100

# Total range = 100 - (-100) = 200
# Step size to generate 100 values
step = 200 / 100

# Loop 100 times
for i in range(100):

    # Store current x value
    x_values.append(x)

    # Calculate exponent part of Gaussian formula
    # exponent = -(x^2) / 2
    exponent = -(x * x) / 2

    # Calculate Gaussian function value
    # 1 / sqrt(2π) normalizes the curve
    # exp(exponent) computes e^exponent
    y = (1 / math.sqrt(2 * math.pi)) * math.exp(exponent)

    # Store Gaussian value
    y_values.append(y)

    # Move to next x value
    x = x + step

# Print first few Gaussian values
print("First 5 Gaussian values:", y_values[:5])


#Output
# First 5 Gaussian values: [0.0, 0.0, 0.0, 0.0, 0.0]