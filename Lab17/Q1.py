import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Load CSV File
# -----------------------------
df = pd.read_csv("Kernel.csv")

# -----------------------------
# Transform Function
# ϕ(x1,x2) = (x1², √2*x1*x2, x2²)
# -----------------------------
def Transform(x1, x2):
    return np.array([
        x1**2,
        np.sqrt(2) * x1 * x2,
        x2**2
    ])

# -----------------------------
# Plot Original 2D Data
# -----------------------------
plt.figure(figsize=(8,6))

for i in range(len(df)):
    x1 = df.loc[i, 'x1']
    x2 = df.loc[i, 'x2']
    label = df.loc[i, 'label']

    if label == "Blue":
        plt.scatter(x1, x2, color='blue')
    else:
        plt.scatter(x1, x2, color='red')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Original 2D Data")
plt.grid(True)
plt.show()
# -----------------------------
# Plot Transformed 3D Data
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(df)):
    x1 = df.loc[i, 'x1']
    x2 = df.loc[i, 'x2']
    label = df.loc[i, 'label']

    point = Transform(x1, x2)

    if label == "Blue":
        ax.scatter(point[0], point[1], point[2], color='blue')
    else:
        ax.scatter(point[0], point[1], point[2], color='red')

ax.set_xlabel("x1²")
ax.set_ylabel("√2 x1x2")
ax.set_zlabel("x2²")
ax.set_title("Transformed 3D Data")
plt.show()

# -----------------------------
# Dot Product in Higher Dimension
# x1 = [3,6], x2 = [10,10]
# -----------------------------
a = np.array([3,6])
b = np.array([10,10])

phi_a = Transform(a[0], a[1])
phi_b = Transform(b[0], b[1])

dot_product = np.dot(phi_a, phi_b)

print("Dot Product in Higher Dimension =", dot_product)

# -----------------------------
# Polynomial Kernel
# K(a,b) = (a.b)^2
# -----------------------------
def kernel(x, y):
    return (np.dot(x, y))**2

kernel_value = kernel(a, b)

print("Polynomial Kernel Value =", kernel_value)

# -----------------------------
# Verification
# -----------------------------
if dot_product == kernel_value:
    print("Both values are equal.")
else:
    print("Values are different.")




