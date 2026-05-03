# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
if __name__ == "__main__":
    # Generate data using make_friedman1
    X, y = make_friedman1(n_samples=100, n_features=5, noise=0.1, random_state=0)

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    # Train a linear regression model on the original data
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print("MSE with original features:", mean_squared_error(y, y_pred))

    # Train a linear regression model on the polynomial features
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    print("MSE with polynomial features:", mean_squared_error(y, y_pred_poly))

    # Plot the data
    plt.scatter(y, y_pred, label='Original Features')
    plt.scatter(y, y_pred_poly, label='Polynomial Features')
    plt.legend()
    plt.show()