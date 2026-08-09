# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Ridge and Lasso regression models with L1 and L2 regularization
ridge_model = Ridge(alpha=0.1)  # L2 regularization
lasso_model = Lasso(alpha=0.1)  # L1 regularization

# Train the models
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Print the coefficients
print("Ridge Coefficients:", ridge_model.coef_)
print("Lasso Coefficients:", lasso_model.coef_)

# Plot the coefficients
plt.bar(range(len(ridge_model.coef_)), ridge_model.coef_, label='Ridge')
plt.bar(range(len(lasso_model.coef_)), lasso_model.coef_, label='Lasso')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("L1 (Lasso) vs L2 (Ridge) Regularization Demo")