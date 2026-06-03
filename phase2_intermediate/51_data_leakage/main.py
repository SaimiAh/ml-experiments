import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Introduce data leakage by adding a feature that is highly correlated with the target
X_leaked = np.hstack((X, y[:, None]))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_leaked, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the leaked data
model_leaked = LogisticRegression()
model_leaked.fit(X_train, y_train)

# Train a logistic regression model on the original data
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42)
model_original = LogisticRegression()
model_original.fit(X_train_original, y_train_original)

# Evaluate the models
y_pred_leaked = model_leaked.predict(X_test)
y_pred_original = model_original.predict(X_test_original)

print("Accuracy with leaked data:", accuracy_score(y_test, y_pred_leaked))
print("Accuracy with original data:", accuracy_score(y_test_original, y_pred_original))

if __name__ == "__main__":
    print("Running data leakage demo...")
    print("Accuracy with leaked data:", accuracy_score(y_test, y_pred_leaked))
    print("Accuracy with original data:", accuracy_score(y_test_original, y_pred_original))