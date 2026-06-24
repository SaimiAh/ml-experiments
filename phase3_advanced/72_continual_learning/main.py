# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=0, n_classes=2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
print("Initial Model Accuracy:", accuracy_score(y_test, y_pred))

# Simulate new data for continual learning
new_X, new_y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, n_classes=2)

# Update the model with new data
model.fit(np.concatenate((X_train, new_X)), np.concatenate((y_train, new_y)))

# Evaluate updated model on test set
new_y_pred = model.predict(X_test)
print("Updated Model Accuracy:", accuracy_score(y_test, new_y_pred))

# Continual learning without forgetting
# We can use a simple approach like re-training the model with all the data we have

# Generate more new data
more_new_X, more_new_y = make_classification(n_samples=50, n_features=5, n_informative=3, n_redundant=0, n_classes=2)

# Update the model with all data
model.fit(np.concatenate((X_train, new_X, more_new_X)), np.concatenate((y_train, new_y, more_new_y)))

# Evaluate updated model on test set
final_y_pred = model.predict(X_test)
print("Final Model Accuracy:", accuracy_score(y_test, final_y_pred))

if __name__ == "__main__":
    print("Continual Learning Demo")
    print("------------------------")