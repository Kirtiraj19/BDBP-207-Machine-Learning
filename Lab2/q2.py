# Compute the dot product of two vectors, x and y given below x = [2  1  2]T and y = [1  2  2]T .
# What is the meaning of the dot product of two vectors? Illustrate that with your own example.

# Defining vectors as Python lists
x = [2, 1, 2]
y = [1, 2, 2]

# Initialize dot product result
dot_product = 0

# Loop through each element
for i in range(len(x)):
    dot_product = dot_product + (x[i] * y[i])

# Print result
print("Dot Product:", dot_product)
