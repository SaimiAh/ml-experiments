# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    return X

# Apply StandardScaler
def apply_standard_scaler(X):
    """Apply StandardScaler to scale features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Apply MinMaxScaler
def apply_min_max_scaler(X):
    """Apply MinMaxScaler to scale features"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Main function
if __name__ == "__main__":
    X = load_data()
    print("Original Data:")
    print("Mean: ", np.mean(X, axis=0))
    print("Std: ", np.std(X, axis=0))

    X_standard = apply_standard_scaler(X)
    print("\nStandardScaler:")
    print("Mean: ", np.mean(X_standard, axis=0))
    print("Std: ", np.std(X_standard, axis=0))

    X_min_max = apply_min_max_scaler(X)
    print("\nMinMaxScaler:")
    print("Min: ", np.min(X_min_max, axis=0))
    print("Max: ", np.max(X_min_max, axis=0))