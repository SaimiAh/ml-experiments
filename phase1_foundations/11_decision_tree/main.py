# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Print classification report
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Decision Tree Classifier Demo")
    print("Dataset:", iris.target_names)
    print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred))