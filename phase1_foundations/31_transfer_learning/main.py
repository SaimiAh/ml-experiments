# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split data into training and testing sets
def split_data(X, y):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Scale data using StandardScaler
def scale_data(X_train, X_test):
    """Scale data using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train model using SVC
def train_model(X_train_scaled, y_train):
    """Train model using SVC"""
    model = SVC(kernel='linear', C=1)
    model.fit(X_train_scaled, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model"""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    # Plot data (optional)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    # plt.show()