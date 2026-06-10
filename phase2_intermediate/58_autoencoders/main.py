import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X = iris.data

# Apply standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder for dimensionality reduction
class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.weights = np.random.rand(input_dim, encoding_dim)
        self.bias = np.zeros((1, encoding_dim))

    def encode(self, X):
        return np.tanh(np.dot(X, self.weights) + self.bias)

    def decode(self, encoded_X):
        return np.tanh(np.dot(encoded_X, self.weights.T))

# Initialize autoencoder
autoencoder = Autoencoder(input_dim=4, encoding_dim=2)

# Train autoencoder (simple gradient descent)
learning_rate = 0.1
iterations = 1000
for _ in range(iterations):
    encoded_X = autoencoder.encode(X_scaled)
    decoded_X = autoencoder.decode(encoded_X)
    error = np.mean((X_scaled - decoded_X) ** 2)
    d_weights = -2 * np.dot(X_scaled.T, (X_scaled - decoded_X)) * learning_rate
    d_bias = -2 * np.sum((X_scaled - decoded_X), axis=0, keepdims=True) * learning_rate
    autoencoder.weights += d_weights
    autoencoder.bias += d_bias

# Encode and decode data
encoded_X = autoencoder.encode(X_scaled)
decoded_X = autoencoder.decode(encoded_X)

if __name__ == "__main__":
    print("Original data shape:", X.shape)
    print("Encoded data shape:", encoded_X.shape)
    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_X[:, 0], encoded_X[:, 1], c=iris.target)
    plt.show()
    print("First 5 original data points:")
    print(X[:5])
    print("First 5 encoded data points:")
    print(encoded_X[:5])