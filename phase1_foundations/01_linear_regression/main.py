import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Add a column of ones to X for the bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize weights
weights = np.random.rand(2)

# Define the learning rate and number of iterations
alpha = 0.01
n_iterations = 1000

# Train the model
for _ in range(n_iterations):
    # Compute predictions
    predictions = np.dot(X, weights)

    # Compute the gradient
    gradient = np.dot(X.T, (predictions - y)) / len(y)

    # Update the weights
    weights -= alpha * gradient

# Make predictions
predictions = np.dot(X, weights)

# Compute MSE
mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the data and the best fit line
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], predictions, color='red')
plt.show()