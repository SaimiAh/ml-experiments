# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data for demonstration
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Define learning rate and number of iterations
learning_rate = 0.01
n_iterations = 1000

# Initialize weights and bias
weight = 0
bias = 0

# Define the cost function (Mean Squared Error)
def cost_function(weight, bias, X, y):
    return np.mean((weight * X + bias - y) ** 2)

# Define Gradient Descent function
def gradient_descent(X, y, weight, bias, learning_rate):
    n_samples = len(X)
    # Calculate predictions
    predictions = weight * X + bias
    # Calculate gradients
    weight_gradient = (-2 / n_samples) * np.sum(X * (y - predictions))
    bias_gradient = (-2 / n_samples) * np.sum(y - predictions)
    # Update weights and bias
    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient
    return weight, bias

# Train the model
costs = []
for i in range(n_iterations):
    weight, bias = gradient_descent(X[:, 0], y, weight, bias, learning_rate)
    cost = cost_function(weight, bias, X[:, 0], y)
    costs.append(cost)

# Plot the cost over iterations
plt.plot(costs)
plt.show()

# Print final weights and bias
print(f"Final weight: {weight}, Final bias: {bias}")
print(f"Final cost: {cost}")

# Plot the data and the best fit line
plt.scatter(X[:, 0], y)
plt.plot(X[:, 0], weight * X[:, 0] + bias, 'r')
plt.show()