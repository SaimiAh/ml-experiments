# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data for demonstration
# We will create a simple regression dataset with a causal relationship between X and y
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients (slope and intercept) to understand the causal relationship
print("Coefficients (slope and intercept): ", model.coef_, model.intercept_)

# Use the model to predict values for the test set
y_pred = model.predict(X_test)

# Plot the data to visualize the causal relationship
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.show()

# Print the actual and predicted values to see the causal inference in action
print("Actual values: ", y_test)
print("Predicted values: ", y_pred)

# Check the model's performance using a simple metric (mean squared error)
mse = np.mean((y_test - y_pred) ** 2)
print("Mean Squared Error: ", mse)