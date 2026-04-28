import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)

# K-Means from scratch
class KMeans:
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the closest centroid
            labels = np.argmin(np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)

            # Update centroids as the mean of all data points assigned to each centroid
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.K)])

            # Check convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        return np.argmin(np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)

# Create a KMeans instance and fit it to the data
kmeans = KMeans(K=3)
kmeans.fit(X)

# Predict cluster labels
labels = kmeans.predict(X)

# Print cluster labels
print("Cluster Labels:", labels)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.5)
plt.show()