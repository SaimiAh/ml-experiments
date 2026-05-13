import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic sequential data
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create a simple RNN-like structure
def rnn(x, weights, bias):
    # Simulate a simple RNN cell
    output = np.zeros_like(x)
    hidden_state = np.zeros_like(x)
    for i in range(len(x)):
        hidden_state[i] = np.tanh(np.dot(x[i], weights) + bias)
        output[i] = hidden_state[i]
    return output

# Initialize weights and bias for the RNN
weights = np.random.rand(1, 1)
bias = np.random.rand(1)

# Run the RNN on the scaled data
output = rnn(X_scaled, weights, bias)

# Print the output
print("RNN Output:")
print(output[:5])

# Plot the output
plt.plot(output)
plt.show()

if __name__ == "__main__":
    print("RNN demo running...")
    print("Output shape:", output.shape)