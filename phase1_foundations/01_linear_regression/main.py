import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of the data
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Initialize weights and bias
w = np.random.rand(1)
b = np.random.rand(1)

# Define the linear regression model
def linear_regression(x, w, b):
    return x * w + b

# Define the cost function (Mean Squared Error)
def cost(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Define the gradient descent update rule
def update(w, b, x, y, learning_rate=0.01):
    y_pred = linear_regression(x, w, b)
    dw = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Train the model
for i in range(1000):
    w, b = update(w, b, X_train, y_train)

# Make predictions on the test set
y_pred = linear_regression(X_test, w, b)

# Print the final weights and bias
print(f"Final weights: {w}, Final bias: {b}")

# Print the test set cost
print(f"Test set cost: {cost(y_pred, y_test)}")

# Plot the data and the regression line
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='r')
plt.show()