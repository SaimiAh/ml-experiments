# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Apply PCA
def apply_pca(X, n_components):
    """Apply PCA to reduce dimensions"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

# Main function
if __name__ == "__main__":
    # Load iris dataset
    X, y = load_data()
    print("Original data shape:", X.shape)
    
    # Apply PCA to reduce dimensions to 2
    X_pca = apply_pca(X, n_components=2)
    print("Data shape after PCA:", X_pca.shape)
    
    # Print explained variance ratio
    pca = PCA(n_components=2)
    pca.fit(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    # Plot the data after PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()