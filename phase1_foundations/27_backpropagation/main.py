# Import necessary libraries
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Define a simple neural network with one input, one hidden, and one output layer
n_inputs = 1
n_hidden = 2
n_outputs = 1

# Initialize weights randomly
np.random.seed(42)
weights1 = np.random.rand(n_inputs, n_hidden)
weights2 = np.random.rand(n_hidden, n_outputs)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the loss function
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Perform backpropagation
def backpropagation(X, y, weights1, weights2, learning_rate=0.01):
    # Forward pass
    hidden_layer = sigmoid(np.dot(X, weights1))
    output_layer = np.dot(hidden_layer, weights2)

    # Backward pass
    output_error = y - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)

    hidden_error = output_delta.dot(weights2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    # Weight updates
    weights2 += learning_rate * hidden_layer.T.dot(output_delta)
    weights1 += learning_rate * X.T.dot(hidden_delta)

    return weights1, weights2

# Train the model
for i in range(1000):
    weights1, weights2 = backpropagation(X, y.reshape(-1, 1), weights1, weights2)

# Make predictions
hidden_layer = sigmoid(np.dot(X, weights1))
output_layer = np.dot(hidden_layer, weights2)

# Print the final loss
print("Final Loss: ", mse(output_layer, y.reshape(-1, 1)))

# Plot the predictions
plt.scatter(X, y)
plt.plot(X, output_layer)
plt.show()

if __name__ == "__main__":
    pass