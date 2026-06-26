# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

# Generate synthetic data
# We're using make_moons to create a graph-like structure
X, y = make_moons(n_samples=200, noise=0.05)

# Introduce graph structure by creating a distance matrix
# This distance matrix represents the edges between nodes in the graph
distance_matrix = np.sqrt(((X[:, np.newaxis] - X)**2).sum(axis=2))

# Apply threshold to create a binary adjacency matrix
# This represents the connections between nodes in the graph
threshold = 0.5
adjacency_matrix = np.where(distance_matrix < threshold, 1, 0)

# Perform KMeans clustering on the graph
# This is a simple example of a graph neural network application
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Print cluster labels
print("Cluster labels:", kmeans.labels_)

# Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.show()

if __name__ == "__main__":
    print("Graph Neural Network Demo")
    print("Number of nodes:", len(X))
    print("Number of edges:", np.sum(adjacency_matrix))