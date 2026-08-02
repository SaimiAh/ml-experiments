# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data using make_blobs
def generate_data():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
    return X

# K-Means Clustering from scratch
class KMeansScratch:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Assign each data point to the closest centroid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)

# Using scikit-learn's KMeans for comparison
def kmeans_sklearn(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_

if __name__ == "__main__":
    # Generate data
    X = generate_data()

    # K-Means from scratch
    kmeans_scratch = KMeansScratch(n_clusters=3)
    kmeans_scratch.fit(X)
    labels_scratch = kmeans_scratch.predict(X)

    # K-Means using scikit-learn
    labels_sklearn = kmeans_sklearn(X, n_clusters=3)

    print("K-Means from scratch labels:")
    print(labels_scratch)
    print("\nK-Means using scikit-learn labels:")
    print(labels_sklearn)

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=labels_scratch)
    plt.show()