# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter space for Bayesian optimisation
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 15, 20]
}

# Perform Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Plot the scores for each combination of hyperparameters
plt.figure(figsize=(10, 6))
for i, depth in enumerate(param_grid['max_depth']):
    scores = [grid_search.cv_results_[f'param_n_estimators_{n}'][f'mean_test_score_{i}'] for n in param_grid['n_estimators']]
    plt.plot(param_grid['n_estimators'], scores, label=f'Depth: {depth}')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Bayesian Optimisation for Hyperparameters')
plt.legend()
plt.show()