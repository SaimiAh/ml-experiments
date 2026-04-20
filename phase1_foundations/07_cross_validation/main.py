# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize KFold with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store accuracy scores
train_acc = []
test_acc = []

# Perform K-Fold Cross Validation
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict and calculate accuracy scores
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc.append(accuracy_score(y_train, y_pred_train))
    test_acc.append(accuracy_score(y_test, y_pred_test))

# Print accuracy scores
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Print mean accuracy scores
print("Mean Train Accuracy:", np.mean(train_acc))
print("Mean Test Accuracy:", np.mean(test_acc))

# Plot a simple bar chart
plt.bar(range(5), train_acc, label='Train Accuracy')
plt.bar(range(5), test_acc, label='Test Accuracy')
plt.legend()
plt.show()