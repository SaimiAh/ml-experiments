# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_moons

# Generate synthetic data
def generate_synthetic_data():
    # Generate moons dataset with outliers
    X, y = make_moons(n_samples=200, noise=0.05)
    # Add some outliers
    X_outliers = np.random.uniform(-2, 2, size=(5, 2))
    X = np.vstack((X, X_outliers))
    return X

# Train Isolation Forest model
def train_model(X):
    # Initialize Isolation Forest model
    model = IsolationForest(contamination=0.1)
    # Fit the model to the data
    model.fit(X)
    return model

# Detect anomalies
def detect_anomalies(model, X):
    # Predict anomalies
    predictions = model.predict(X)
    return predictions

# Main function
if __name__ == "__main__":
    # Generate synthetic data
    X = generate_synthetic_data()
    
    # Train Isolation Forest model
    model = train_model(X)
    
    # Detect anomalies
    predictions = detect_anomalies(model, X)
    
    # Print the number of anomalies
    num_anomalies = np.sum(predictions == -1)
    print(f"Number of anomalies: {num_anomalies}")
    
    # Plot the data
    plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], c='blue')
    plt.scatter(X[predictions == -1, 0], X[predictions == -1, 1], c='red')
    plt.show()