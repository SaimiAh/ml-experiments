# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Prepare data
def prepare_data(X, y):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Scale data
def scale_data(X_train, X_test):
    """Scale data using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train model
def train_model(X_train_scaled, y_train):
    """Train logistic regression model"""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate model using accuracy score"""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    print("Model Accuracy:", accuracy)