import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Bayesian Ridge regression model
model = BayesianRidge()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the coefficients (mean and std) of the model
print("Coefficients (mean):", model.coef_)
print("Coefficients (std):", np.sqrt(model.sigma_))

# Plot the data and the predictions
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted")
plt.legend()
plt.show()