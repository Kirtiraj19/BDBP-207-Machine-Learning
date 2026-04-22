# Implement K-Means algorithm ground-up using Python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Sample Dataset
# -----------------------------
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
])

# -----------------------------
# Number of Clusters
# -----------------------------
k = 3

# Random Initial Centroids
np.random.seed(42)
centroids = X[np.random.choice(len(X), k, replace=False)]

# -----------------------------
# KMeans Iterations
# -----------------------------
for iteration in range(10):

    clusters = []

    # Assign points to nearest centroid
    for point in X:
        distances = [np.linalg.norm(point - c) for c in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)

    clusters = np.array(clusters)

    # Update centroids
    new_centroids = []

    for i in range(k):
        cluster_points = X[clusters == i]
        new_centroids.append(cluster_points.mean(axis=0))

    new_centroids = np.array(new_centroids)

    # Stop if no change
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

# -----------------------------
# Final Output
# -----------------------------
print("Final Centroids:\n", centroids)

# -----------------------------
# Plot Clusters
# -----------------------------
colors = ['red', 'blue', 'green']

for i in range(k):
    pts = X[clusters == i]
    plt.scatter(pts[:,0], pts[:,1], color=colors[i])

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("K-Means From Scratch")
plt.legend()
plt.grid(True)
plt.show()



# # -----------------------------
# # using scikit learn
# # -----------------------------
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# # -----------------------------
# # Sample Dataset
# # -----------------------------
# X = np.array([
#     [1, 2],
#     [1.5, 1.8],
#     [5, 8],
#     [8, 8],
#     [1, 0.6],
#     [9, 11],
#     [8, 2],
#     [10, 2],
#     [9, 3]
# ])
#
# # -----------------------------
# # Apply KMeans
# # -----------------------------
# model = KMeans(
#     n_clusters=3,
#     random_state=42,
#     n_init=10
# )
#
# model.fit(X)
#
# labels = model.labels_
# centroids = model.cluster_centers_
#
# # -----------------------------
# # Output
# # -----------------------------
# print("Cluster Labels:")
# print(labels)
#
# print("\nCentroids:")
# print(centroids)
#
# # -----------------------------
# # Plot Clusters
# # -----------------------------
# colors = ['red', 'blue', 'green']
#
# for i in range(3):
#     pts = X[labels == i]
#     plt.scatter(pts[:,0], pts[:,1], color=colors[i], label="Cluster "+str(i+1))
#
# plt.scatter(
#     centroids[:,0],
#     centroids[:,1],
#     color='black',
#     marker='X',
#     s=200,
#     label='Centroids'
# )
#
# plt.title("K-Means using scikit-learn")
# plt.legend()
# plt.grid(True)
# plt.show()
