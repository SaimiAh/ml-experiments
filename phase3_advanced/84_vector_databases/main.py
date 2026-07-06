# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Generate synthetic data using make_blobs
# This creates a dataset of 1000 samples, each with 2 features, in 5 clusters
if __name__ == "__main__":
    # Create synthetic data
    X, y = make_blobs(n_samples=1000, n_features=2, centers=5, random_state=0)

    # Calculate pairwise distances between all points
    distances = pairwise_distances(X)

    # Create a simple vector database by storing the distances
    vector_db = pd.DataFrame(distances)

    # Print the first few rows of the vector database
    print("Vector Database (first 5 rows):")
    print(vector_db.head())

    # Use the vector database to find the nearest neighbors to a query point
    query_point = X[0]  # Use the first point as the query point
    distances_to_query = np.linalg.norm(X - query_point, axis=1)
    nearest_neighbors = np.argsort(distances_to_query)[1:6]  # Get the indices of the 5 nearest neighbors

    # Print the nearest neighbors
    print("\nNearest Neighbors to Query Point:")
    print(nearest_neighbors)

    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(query_point[0], query_point[1], c='red', marker='x', s=100)  # Mark the query point
    plt.show()