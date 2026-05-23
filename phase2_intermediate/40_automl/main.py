# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Perform AutoML using GridSearchCV
def automl(X_train, y_train):
    # Define hyperparameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10]
    }
    
    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

# Evaluate model
def evaluate_model(model, X_test, y_test):
    yPred = model.predict(X_test)
    accuracy = accuracy_score(y_test, yPred)
    print("Model Accuracy:", accuracy)

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_model = automl(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)