import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network with dropout
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# Define neural network with dropout
mlp_dropout = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42, alpha=0.1)

# Train without dropout
mlp.fit(X_train, y_train)
print("Accuracy without dropout: ", mlp.score(X_test, y_test))

# Train with dropout
mlp_dropout.fit(X_train, y_train)
print("Accuracy with dropout: ", mlp_dropout.score(X_test, y_test))

if __name__ == "__main__":
    # Run demo
    print("Running demo...")
    print("Iris dataset shape: ", X.shape)
    print("Training set shape: ", X_train.shape)
    print("Test set shape: ", X_test.shape)