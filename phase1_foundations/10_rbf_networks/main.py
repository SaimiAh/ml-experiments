# Import necessary libraries
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.rbf import RBF
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for demonstration
if __name__ == "__main__":
    # Create a sample dataset
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=0, cluster_std=1.0)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define the RBF network
    rbf = RBF(1, 10)
    mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
    
    # Fit the model
    mlp.fit(X_scaled, y)
    
    # Print the training accuracy
    print("Training Accuracy:", mlp.score(X_scaled, y))
    
    # Plot the data
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
    plt.show()