# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print("Model Accuracy:", accuracy)

# This is a simplified representation of CLIP, where we use text labels 
# (class names) to connect with image features (in this case, iris features)
class_names = iris.target_names

if __name__ == "__main__":
    # Print class names (text) and their corresponding features (images)
    print("Class Names (Text):", class_names)
    print("Features (Images):", X_test[:5])

    # Show a simple plot to represent image features
    plt.scatter(X_test[:, 0], X_test[:, 1])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Print model predictions
    print("Model Predictions:", y_pred[:5])