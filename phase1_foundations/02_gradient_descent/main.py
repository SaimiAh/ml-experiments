import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Add bias term (intercept) to X
X = np.c_[np.ones(X.shape[0]), X]

# Initialize model parameters (weights)
np.random.seed(0)
w = np.random.rand(X.shape[1])

# Define learning rate and number of iterations
lr = 0.01
n_iter = 1000

# Initialize loss list to store the losses at each iteration
losses = []

# Gradient Descent algorithm
for _ in range(n_iter):
    # Compute the predicted values
    y_pred = np.dot(X, w)
    
    # Compute the loss (mean squared error)
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)
    
    # Compute the gradients of the loss with respect to the model parameters
    gradients = 2 * np.dot(X.T, (y_pred - y)) / X.shape[0]
    
    # Update the model parameters using the gradients and learning rate
    w -= lr * gradients

# Print the final model parameters
print("Final weights:", w)

# Plot the data and the best fit line
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, w))
plt.show()

# Plot the loss at each iteration
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()