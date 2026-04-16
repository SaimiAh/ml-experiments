import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define sigmoid function
def sigmoid(x):
    # This function maps any real number to a value between 0 and 1
    return 1 / (1 + np.exp(-x))

# Define logistic regression model
def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    # Train the model
    for _ in range(num_iterations):
        # Calculate the predicted probabilities
        predictions = sigmoid(np.dot(X, weights) + bias)

        # Calculate the gradients
        dw = (1 / X.shape[0]) * np.dot(X.T, (predictions - y))
        db = (1 / X.shape[0]) * np.sum(predictions - y)

        # Update the weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Train the model
weights, bias = logistic_regression(X_train, y_train)

# Make predictions on the test set
predictions = sigmoid(np.dot(X_test, weights) + bias)
predicted_classes = (predictions > 0.5).astype(int)

# Print the metrics
print("Accuracy:", accuracy_score(y_test, predicted_classes))
print("Classification Report:\n", classification_report(y_test, predicted_classes))

# Print the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

if __name__ == "__main__":
    print("Logistic Regression from Scratch")