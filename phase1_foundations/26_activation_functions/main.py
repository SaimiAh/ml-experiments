# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define activation functions to compare
activation_functions = ['identity', 'relu', 'tanh']

# Train neural network with different activation functions
for activation in activation_functions:
    # Create a neural network regressor with one hidden layer and specified activation function
    model = MLPRegressor(hidden_layer_sizes=(10,), activation=activation, max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print training score
    print(f"Training score with {activation} activation: {model.score(X_train, y_train)}")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Print test score
    print(f"Test score with {activation} activation: {model.score(X_test, y_test)}")

# Visualize predictions for different activation functions
plt.scatter(X_test, y_test, label='Actual')
for activation in activation_functions:
    model = MLPRegressor(hidden_layer_sizes=(10,), activation=activation, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, label=activation)

plt.legend()
plt.show()