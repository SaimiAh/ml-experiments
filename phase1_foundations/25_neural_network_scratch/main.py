import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
n_inputs = X.shape[1]
n_outputs = 1

# Initialize weights and biases
weights = np.random.rand(n_inputs)
bias = np.random.rand(1)

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network forward pass
def forward_pass(X):
    linear_output = np.dot(X, weights) + bias
    output = sigmoid(linear_output)
    return output

# Define the loss function (binary cross-entropy)
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Train the neural network
for epoch in range(100):
    output = forward_pass(X_train)
    loss_value = loss(y_train, output)
    # Print loss at every 10th epoch
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss_value:.4f}')

# Make predictions on test set
y_pred = forward_pass(X_test)

# Print accuracy
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()