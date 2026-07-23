# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    return X

# Feature scaling using StandardScaler
def standard_scale(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Feature scaling using MinMaxScaler
def min_max_scale(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

if __name__ == "__main__":
    # Load iris dataset
    X = load_data()

    # Apply StandardScaler
    X_std = standard_scale(X)
    print("StandardScaler mean: ", np.mean(X_std, axis=0))
    print("StandardScaler std: ", np.std(X_std, axis=0))

    # Apply MinMaxScaler
    X_min_max = min_max_scale(X)
    print("MinMaxScaler min: ", np.min(X_min_max, axis=0))
    print("MinMaxScaler max: ", np.max(X_min_max, axis=0))

    # Plot histograms for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(X[:, 0], bins=10, alpha=0.5, label='Original')
    plt.hist(X_std[:, 0], bins=10, alpha=0.5, label='StandardScaler')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(X[:, 0], bins=10, alpha=0.5, label='Original')
    plt.hist(X_min_max[:, 0], bins=10, alpha=0.5, label='MinMaxScaler')
    plt.legend()
    plt.show()