# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter space for GridSearch
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 15, 20]
}

# Perform GridSearch
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Define hyperparameter space for RandomSearch
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 15, 20]
}

# Perform RandomSearch
random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=5, n_iter=10)
random_search.fit(X_train, y_train)

# Compare the results
print("GridSearch Best Parameters: ", grid_search.best_params_)
print("GridSearch Best Score: ", grid_search.best_score_)
print("RandomSearch Best Parameters: ", random_search.best_params_)
print("RandomSearch Best Score: ", random_search.best_score_)

if __name__ == "__main__":
    # Train a model with the best hyperparameters from GridSearch
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))