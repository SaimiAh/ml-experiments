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
    # Using iris dataset for demonstration
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train a model
def train_model(X_train, y_train):
    # Using logistic regression as a simple model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Make predictions and evaluate
def evaluate_model(model, X_test, y_test):
    # Predicting on test set
    y_pred = model.predict(X_test)
    # Evaluating model performance using accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to demonstrate end-to-end production ML system
def main():
    X, y = load_data()
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = train_model(X_train, y_train)
    
    # Evaluate model performance
    accuracy = evaluate_model(model, X_test, y_test)
    print("Model Accuracy:", accuracy)

    # Plotting a simple histogram for feature 0 of the iris dataset
    plt.hist(X[:, 0], bins=10)
    plt.title('Feature 0 Distribution')
    plt.show()

if __name__ == "__main__":
    main()