import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate synthetic data with 5 features and 2 classes
X, y = make_classification(n_samples=100, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, n_classes=2)

# Split data into training and testing sets (few-shot learning uses very few samples for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model with few-shot learning (only 5 samples)
few_shot_X_train = X_train[:5]
few_shot_y_train = y_train[:5]
model.fit(few_shot_X_train, few_shot_y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Few-shot learning accuracy:", accuracy)

# Train the model with full dataset for comparison
model.fit(X_train, y_train)
full_dataset_predictions = model.predict(X_test)
full_dataset_accuracy = accuracy_score(y_test, full_dataset_predictions)
print("Full dataset accuracy:", full_dataset_accuracy)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()