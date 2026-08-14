import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Define activation functions to compare
activation_functions = ['relu', 'tanh', 'logistic']

# Train and evaluate models with different activation functions
for activation in activation_functions:
    # Initialize and train a neural network regressor
    model = MLPRegressor(hidden_layer_sizes=(10,), activation=activation, random_state=42)
    model.fit(X, y)
    
    # Predict and evaluate the model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Print the results
    print(f"Activation: {activation}, MSE: {mse:.2f}")

# For demo, plot a simple sigmoid (logistic) activation function
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.title('Sigmoid (Logistic) Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()