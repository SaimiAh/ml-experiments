# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Federated learning with 3 clients
def federated_learning(X, y):
    # Split data into 3 clients
    X1, X2, X3 = np.split(X, 3)
    y1, y2, y3 = np.split(y, 3)

    # Train local models
    model1 = LogisticRegression()
    model2 = LogisticRegression()
    model3 = LogisticRegression()
    model1.fit(X1, y1)
    model2.fit(X2, y2)
    model3.fit(X3, y3)

    # Aggregate models using simple averaging
    weights = (model1.coef_ + model2.coef_ + model3.coef_) / 3

    # Create global model
    global_model = LogisticRegression()
    global_model.coef_ = weights

    return global_model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    global_model = federated_learning(X_train, y_train)
    accuracy = global_model.score(X_test, y_test)
    print("Federated Learning Model Accuracy:", accuracy)