# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Define the sigmoid function
def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    """The derivative of the sigmoid function."""
    return x * (1 - x)

# Initialize weights and bias
weights = np.array([0.5, 0.5])
bias = 0.5

# Define the learning rate
learning_rate = 0.1

# Train the model using backpropagation
for i in range(100):
    # Forward pass
    output = sigmoid(np.dot(X, weights) + bias)
    
    # Calculate the error
    error = y - output
    
    # Backward pass
    d_output = error * sigmoid_derivative(output)
    d_weights = np.dot(X.T, d_output)
    d_bias = np.sum(d_output)
    
    # Update the weights and bias
    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

# Print the final weights and bias
print("Final weights: ", weights)
print("Final bias: ", bias)

# Plot the decision boundary
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_values = np.linspace(x_min, x_max, 100)
y_values = (-weights[0] * x_values - bias) / weights[1]
plt.plot(x_values, y_values, label='Decision boundary')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()

if __name__ == "__main__":
    pass