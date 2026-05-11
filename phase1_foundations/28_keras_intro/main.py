# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train test split
def split_data(X, y):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Simple neural network using scikit-learn
from sklearn.neural_network import MLPClassifier
def train_model(X_train, y_train):
    """Train a simple neural network"""
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)