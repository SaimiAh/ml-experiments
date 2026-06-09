# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Generate synthetic data for text classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_classes=2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple text classification model using Naive Bayes
def text_classification_model(X_train, y_train):
    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Train the model and make predictions
model = text_classification_model(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")

# Plot a simple bar chart to compare predicted and actual labels
plt.bar(range(len(y_test)), y_test, label='Actual')
plt.bar(range(len(y_pred)), y_pred, label='Predicted')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("Text Classification using Naive Bayes:")
    print(f"Model Accuracy: {accuracy:.3f}")
    print("Plotting actual and predicted labels...")