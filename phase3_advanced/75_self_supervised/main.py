# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Self-supervised learning: create a new task (predicting the sign of the data)
X_self_supervised = np.sign(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_self, X_test_self, _, _ = train_test_split(X_self_supervised, y, test_size=0.2, random_state=42)

# Train a model on the original task
model = LogisticRegression()
model.fit(X_train, y_train)

# Train a model on the self-supervised task
model_self = LogisticRegression()
model_self.fit(X_train_self, y_train)

# Evaluate the models
y_pred = model.predict(X_test)
y_pred_self = model_self.predict(X_test_self)
print("Original task accuracy:", accuracy_score(y_test, y_pred))
print("Self-supervised task accuracy:", accuracy_score(y_test, y_pred_self))

if __name__ == "__main__":
    print("Self-supervised learning example")
    print("Original task accuracy:", accuracy_score(y_test, y_pred))
    print("Self-supervised task accuracy:", accuracy_score(y_test, y_pred_self))