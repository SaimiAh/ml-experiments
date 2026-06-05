# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Generate synthetic data
if __name__ == "__main__":
    # Create a sample dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

    # Print original data stats
    print("Original Data Stats:")
    print("Mean: ", np.mean(X, axis=0))
    print("Std Dev: ", np.std(X, axis=0))

    # Apply batch normalization
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Print normalized data stats
    print("\nNormalized Data Stats:")
    print("Mean: ", np.mean(X_normalized, axis=0))
    print("Std Dev: ", np.std(X_normalized, axis=0))

    # Train a simple MLP classifier on original and normalized data
    clf_original = MLPClassifier(max_iter=1000)
    clf_normalized = MLPClassifier(max_iter=1000)
    clf_original.fit(X, y)
    clf_normalized.fit(X_normalized, y)

    # Print training accuracy
    print("\nTraining Accuracy:")
    print("Original Data: ", clf_original.score(X, y))
    print("Normalized Data: ", clf_normalized.score(X_normalized, y))

    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original Data")
    plt.show()

    plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y)
    plt.title("Normalized Data")
    plt.show()