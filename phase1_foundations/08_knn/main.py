# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train KNN classifier
def train_knn(X_train, y_train, k=5):
    # Initialize KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the model
    knn.fit(X_train, y_train)
    return knn

# Evaluate KNN classifier
def evaluate_knn(knn, X_test, y_test):
    # Make predictions
    y_pred = knn.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

# Main function
if __name__ == "__main__":
    X, y = load_data()
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train and evaluate KNN classifier
    knn = train_knn(X_train, y_train)
    accuracy = evaluate_knn(knn, X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")