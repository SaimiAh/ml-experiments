# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train a random forest classifier
def train_model(X_train, y_train):
    """Train a random forest classifier"""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print("Random Forest Accuracy:", accuracy)

    # Print feature importances
    feature_importances = model.feature_importances_
    print("Feature Importances:")
    for i, importance in enumerate(feature_importances):
        print(f"Feature {i+1}: {importance:.3f}")