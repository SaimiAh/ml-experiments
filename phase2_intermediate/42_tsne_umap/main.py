# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.manifold import UMAP

# Load iris dataset
iris = load_iris()
data = iris.data

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# Apply UMAP
umap = UMAP(n_components=2, random_state=42)
data_umap = umap.fit_transform(data)

# Print shapes of original and reduced data
print("Original data shape:", data.shape)
print("t-SNE reduced data shape:", data_tsne.shape)
print("UMAP reduced data shape:", data_umap.shape)

# Plot t-SNE and UMAP results
if __name__ == "__main__":
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=iris.target)
    plt.title("t-SNE")
    plt.subplot(1, 2, 2)
    plt.scatter(data_umap[:, 0], data_umap[:, 1], c=iris.target)
    plt.title("UMAP")
    plt.show()