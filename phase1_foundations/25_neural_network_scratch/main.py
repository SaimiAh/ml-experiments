import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the sigmoid function for activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and bias
np.random.seed(42)
weights = np.random.rand(2)
bias = np.random.rand(1)

# Define the learning rate and number of epochs
learning_rate = 0.1
epochs = 1000

# Train the neural network
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X_train, weights) + bias
    a = sigmoid(z)

    # Backward pass
    error = y_train - a
    d_z = error * sigmoid_derivative(a)
    d_weights = np.dot(X_train.T, d_z)
    d_bias = np.sum(d_z)

    # Update weights and bias
    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

# Make predictions on the test set
z = np.dot(X_test, weights) + bias
predictions = sigmoid(z)

# Print some predictions
print("Predictions:", predictions[:5])

# Evaluate the model
accuracy = np.mean((predictions > 0.5) == y_test)
print("Accuracy:", accuracy)

# Plot the data and the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()