# Import necessary libraries
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    return load_iris()

# Train model with K-Fold Cross Validation
def train_model(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
    return np.mean(accuracies)

if __name__ == "__main__":
    # Load dataset
    iris = load_data()
    X, y = iris.data, iris.target
    
    # Train model with K-Fold Cross Validation
    accuracy = train_model(X, y)
    print("Average accuracy with K-Fold Cross Validation:", accuracy)