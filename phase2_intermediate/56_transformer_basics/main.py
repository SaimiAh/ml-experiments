import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Simple self-attention mechanism
def self_attention(query, key, value):
    """
    Simple self-attention mechanism.
    
    Parameters:
    query (np.array): Query vector
    key (np.array): Key vector
    value (np.array): Value vector
    
    Returns:
    np.array: Weighted sum of value vector
    """
    # Calculate attention weights
    attention_weights = np.dot(query, key) / np.sqrt(key.shape[0])
    # Calculate weighted sum of value vector
    output = np.dot(attention_weights, value)
    return output

# Demo self-attention mechanism
if __name__ == "__main__":
    # Generate random query, key, and value vectors
    query = np.random.rand(1, 4)
    key = np.random.rand(4, 4)
    value = np.random.rand(4, 4)
    
    # Calculate self-attention output
    output = self_attention(query, key, value)
    print("Self-Attention Output:")
    print(output)