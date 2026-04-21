# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    return load_iris()

# Train and test split
def split_data(data):
    """Split data into train and test sets"""
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
def scale_data(X_train, X_test):
    """Scale data using StandardScaler"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Train KNN model
def train_knn(X_train, y_train):
    """Train KNN classifier"""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

# Make predictions and evaluate
def evaluate_model(model, X_test, y_test):
    """Make predictions and evaluate the model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Scale data
    X_train, X_test = scale_data(X_train, X_test)
    
    # Train KNN model
    model = train_knn(X_train, y_train)
    
    # Make predictions and evaluate
    evaluate_model(model, X_test, y_test)