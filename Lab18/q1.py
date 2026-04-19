import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Load CSV
df = pd.read_csv("Data.csv")

# Features
X = df[['x1', 'x2']].values

# Labels
y = np.array([0 if i == "Blue" else 1 for i in df['label']])

# Plot original data
for label, name in [(0, "Blue"), (1, "Red")]:
    pts = X[y == label]
    plt.scatter(pts[:, 0], pts[:, 1], label=name)

plt.legend()
plt.title("Original Data")
plt.show()

# RBF Kernel Function
def rbf_kernel(x, z, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x-z)**2)

print("Sample Kernel Value:", rbf_kernel(X[0], X[1]))

# Kernel Matrix
def kernel_matrix(X, gamma=0.1):
    n = len(X)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i,j] = rbf_kernel(X[i], X[j], gamma)

    return K

K = kernel_matrix(X)
print(K)

# Train Models
model_rbf = SVC(kernel='rbf', gamma=0.1)
model_rbf.fit(X, y)

model_poly = SVC(kernel='poly', degree=2)
model_poly.fit(X, y)

# Plot boundary
def plot_boundary(model, X, y, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    for label in [0,1]:
        pts = X[y == label]
        plt.scatter(pts[:,0], pts[:,1])

    plt.title(title)
    plt.show()

plot_boundary(model_rbf, X, y, "RBF Kernel")
plot_boundary(model_poly, X, y, "Polynomial Kernel")