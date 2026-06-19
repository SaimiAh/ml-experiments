# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train a logistic regression model
def train_model(X_train, y_train):
    """Train a logistic regression model"""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function
if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Print real output
    print("Model Coefficients:")
    print(model.coef_)
    print("Model Intercept:")
    print(model.intercept_)