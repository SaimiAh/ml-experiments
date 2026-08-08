# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Create a pipeline with polynomial features and linear regression
pipeline = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Train the model
pipeline.fit(X, y)

# Print coefficients
print("Coefficients:", pipeline.named_steps['linearregression'].coef_)

# Predict and plot
y_pred = pipeline.predict(X)
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, label='Model', color='red')
plt.legend()
plt.show()

# Evaluate the model
score = pipeline.score(X, y)
print("R2 Score:", score)