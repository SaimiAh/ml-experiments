# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_data():
    # Use make_circles to generate a dataset with non-linear relationship
    X, y = make_circles(n_samples=200, factor=0.5, noise=0.05)
    return X, y

# Apply Radial Basis Function (RBF) kernel
def apply_rbf(X):
    # Use RBFSampler to transform data using RBF kernel
    rbf = RBFSampler(gamma=0.1, random_state=42)
    X_rbf = rbf.fit_transform(X)
    return X_rbf

# Train a simple neural network
def train_network(X, y):
    # Create a simple neural network regressor
    nn = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
    # Train the network
    nn.fit(X, y)
    return nn

# Main function
if __name__ == "__main__":
    # Generate data
    X, y = generate_data()
    
    # Apply RBF kernel
    X_rbf = apply_rbf(X)
    
    # Train a neural network on original and RBF-transformed data
    nn_original = train_network(X, y)
    nn_rbf = train_network(X_rbf, y)
    
    # Print scores
    print("Original data score:", nn_original.score(X, y))
    print("RBF-transformed data score:", nn_rbf.score(X_rbf, y))
    
    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()