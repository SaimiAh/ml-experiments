# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, n_repeated=0, n_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

# Print the model's accuracy
print("Model Accuracy:", accuracy)

# In zero-shot classification, we would typically use a transformer-based model
# However, since we are limited to using only numpy, pandas, matplotlib, and scikit-learn,
# we will simulate this by using the logistic regression model as a proxy.

if __name__ == "__main__":
    print("Running Zero-Shot Classification Demo...")
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
    print("Model Accuracy:", accuracy)
    # Plot a simple bar chart to visualize the class distribution
    plt.bar(range(len(set(y_train))), [np.sum(y_train == i) for i in set(y_train)])
    plt.xlabel("Class Labels")
    plt.ylabel("Class Counts")
    plt.title("Class Distribution")
    plt.show()