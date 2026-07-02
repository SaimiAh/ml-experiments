# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset (using it for demo purposes, as speech data is complex)
# We'll treat the 'species' column as our "speech" labels
iris = load_iris()
X = iris.data
y = iris.target

# Convert labels to text (like speech)
labels = iris.target_names
label_encoder = LabelEncoder()
y_text = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_text, test_size=0.2, random_state=42)

# Simple "speech recognition" model: nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Speech recognition accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    # Demo
    print("Iris dataset 'species' as speech labels:")
    print(labels)
    print("\nPredictions:")
    print(y_pred)
    print("\nActual labels:")
    print(y_test)
    print(f"\nAccuracy: {accuracy:.2f}")