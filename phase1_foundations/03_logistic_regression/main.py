import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the sigmoid function for logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression model
def logistic_regression(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    return sigmoid(linear_model)

# Define the cost function (binary cross-entropy)
def cost(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize the weights and bias
weights = np.zeros(X.shape[1])
bias = 0

# Train the model using gradient descent
for _ in range(100):
    y_pred = logistic_regression(X_train, weights, bias)
    loss = cost(y_pred, y_train)
    
    # Compute gradients
    weights_grad = np.dot(X_train.T, (y_pred - y_train)) / len(y_train)
    bias_grad = np.mean(y_pred - y_train)
    
    # Update weights and bias
    weights -= 0.01 * weights_grad
    bias -= 0.01 * bias_grad

# Print the trained model's performance
y_pred = logistic_regression(X_test, weights, bias)
print("Trained Model's Accuracy:", np.mean((y_pred > 0.5) == y_test))

if __name__ == "__main__":
    print("Running Logistic Regression from Scratch")
    y_pred = logistic_regression(X_test, weights, bias)
    print("Predicted Probabilities:", y_pred)

    # Plot the data and decision boundary
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred)
    plt.show()