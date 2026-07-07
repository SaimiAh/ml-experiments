# Import necessary libraries
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple prompt engineering task: 
# Given a set of features, predict the class label
def prompt_engineering_example(X_train, y_train, X_test):
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    return y_pred

# Make predictions and evaluate the model
y_pred = prompt_engineering_example(X_train, y_train, X_test)
print("Predicted labels:", y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Visualize the dataset (first two features)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Iris dataset")
plt.show()

if __name__ == "__main__":
    print("Running prompt engineering example...")
    print("Dataset shape:", X.shape, y.shape)
    print("Test set shape:", X_test.shape, y_test.shape)