# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train a model
def train_model(X, y):
    """Train a logistic regression model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Create a simple API using FastAPI is not allowed, so we simulate it
def simulate_api(model):
    """Simulate a simple API"""
    # Simulate API request
    input_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal length, sepal width, petal length, petal width
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Simulate API request
    prediction = simulate_api(model)
    print(f"API prediction: {prediction}")