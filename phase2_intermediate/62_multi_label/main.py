import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate synthetic data for multi-label classification
X, y = make_multilabel_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a multi-output classifier with a random forest classifier as the base estimator
clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy score
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    print("Running demo...")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))