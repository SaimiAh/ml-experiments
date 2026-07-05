# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data using make_blobs
# WHAT: Create a dataset with 3 clusters and 100 samples
# WHY: To demonstrate a simple RAG-like process
def generate_data():
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
    return X, y

# Simple Retrieval Augmented Generation (RAG) function
# WHAT: Find the nearest neighbors for a given point
# WHY: To demonstrate a basic concept of RAG
def rag_nearest_neighbors(X, point, n_neighbors=5):
    distances = np.linalg.norm(X - point, axis=1)
    indices = np.argsort(distances)[:n_neighbors]
    return X[indices]

# Main function
def main():
    # Generate synthetic data
    X, y = generate_data()

    # Choose a random point
    point = X[np.random.choice(len(X))]

    # Find the nearest neighbors
    neighbors = rag_nearest_neighbors(X, point)

    # Print the point and its neighbors
    print("Point:", point)
    print("Neighbors:")
    print(neighbors)

    # Plot the data and the point with its neighbors
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(point[0], point[1], c='r')
    plt.scatter(neighbors[:, 0], neighbors[:, 1], c='g')
    plt.show()

if __name__ == "__main__":
    main()