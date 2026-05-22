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
    """Load iris dataset and split into features and target."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split data into training and test sets
def split_data(X, y):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train a model
def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy score."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Occam's Razor principle: Simple models are preferred over complex ones
    # Sampling principle: Models should be trained and tested on different data
    # Snooping principle: Models should not be overtrained on a specific dataset
    print("Three learning principles:")
    print("1. Occam's Razor: Prefer simple models")
    print("2. Sampling: Train and test on different data")
    print("3. Snooping: Avoid overtraining on a specific dataset")

if __name__ == "__main__":
    main()