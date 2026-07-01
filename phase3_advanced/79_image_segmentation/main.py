# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data for image segmentation demo
def generate_synthetic_data():
    # Create blobs with different colors
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
    return X, y

# Apply KMeans clustering for segmentation (simple approach)
def apply_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans.labels_

# Main function
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Apply KMeans clustering for segmentation
    segmented_labels = apply_kmeans_clustering(X, n_clusters=3)

    # Print segmented labels
    print("Segmented Labels:")
    print(segmented_labels)

    # Plot original and segmented data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original Data")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=segmented_labels)
    plt.title("Segmented Data")
    plt.show()