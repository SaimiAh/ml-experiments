import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_blobs(n_samples=100, centers=2, n_features=5, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define attention weights
attention_weights = np.random.rand(X_train.shape[0], X_train.shape[1])

# Compute attention output
attention_output = np.sum(X_train * attention_weights, axis=1)

# Print attention output shape and values
print("Attention Output Shape:", attention_output.shape)
print("Attention Output Values:")
print(attention_output)

# Plot attention weights
plt.bar(range(X_train.shape[1]), np.mean(attention_weights, axis=0))
plt.xlabel("Feature Index")
plt.ylabel("Attention Weight")
plt.show()

if __name__ == "__main__":
    # Print real output
    print("Attention Mechanism Demo")
    print("------------------------")
    print("Training Data Shape:", X_train.shape)
    print("Attention Weights Shape:", attention_weights.shape)
    print("Attention Output Shape:", attention_output.shape)