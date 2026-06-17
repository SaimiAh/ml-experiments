# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Print model coefficients
print("Model Coefficients:", model.coef_)

# Model pruning: Remove features with small coefficients
threshold = 0.1
pruned_model = LogisticRegression(max_iter=1000)
pruned_model.fit(X_train[:, np.abs(model.coef_[0]) > threshold], y_train)

# Print pruned model coefficients
print("Pruned Model Coefficients:", pruned_model.coef_)

# Model quantization: Reduce precision of model coefficients
quantized_model = LogisticRegression(max_iter=1000)
quantized_model.coef_ = np.round(model.coef_, 2)

# Print quantized model coefficients
print("Quantized Model Coefficients:", quantized_model.coef_)

if __name__ == "__main__":
    # Train and print accuracy of original model
    print("Original Model Accuracy:", model.score(X_test, y_test))
    
    # Train and print accuracy of pruned model
    pruned_model = LogisticRegression(max_iter=1000)
    pruned_model.fit(X_train[:, np.abs(model.coef_[0]) > threshold], y_train)
    print("Pruned Model Accuracy:", pruned_model.score(X_test[:, np.abs(model.coef_[0]) > threshold], y_test))
    
    # Train and print accuracy of quantized model
    quantized_model = LogisticRegression(max_iter=1000)
    quantized_model.coef_ = np.round(model.coef_, 2)
    quantized_model.fit(X_train, y_train)
    print("Quantized Model Accuracy:", quantized_model.score(X_test, y_test))