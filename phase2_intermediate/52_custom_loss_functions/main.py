# Import necessary libraries
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom mean squared error loss function
def custom_mse_loss(y_true, y_pred):
    """
    Custom mean squared error loss function.
    
    Parameters:
    y_true (numpy array): True values.
    y_pred (numpy array): Predicted values.
    
    Returns:
    float: Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)

# Train a simple linear regression model
# For simplicity, we'll just use numpy for linear regression
X_train_with_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_with_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# Calculate coefficients
coefficients = np.linalg.inv(X_train_with_bias.T @ X_train_with_bias) @ X_train_with_bias.T @ y_train

# Predict values
y_pred = X_test_with_bias @ coefficients

# Calculate custom MSE loss
loss = custom_mse_loss(y_test, y_pred)
print("Custom MSE Loss:", loss)

if __name__ == "__main__":
    # Plot data
    plt.scatter(X_test, y_test, label='True values')
    plt.scatter(X_test, y_pred, label='Predicted values')
    plt.legend()
    plt.show()