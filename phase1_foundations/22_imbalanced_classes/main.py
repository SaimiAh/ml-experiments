# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Generate synthetic imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=0, n_classes=2, weights=[0.1, 0.9], random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print class distribution before SMOTE
print("Class distribution before SMOTE:", np.bincount(y_train))

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:", np.bincount(y_train_smote))

# Train a classifier on the original and SMOTE-balanced data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Original data
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy on original data:", accuracy_score(y_test, y_pred))

# SMOTE-balanced data
model_smote = LogisticRegression()
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)
print("Accuracy on SMOTE-balanced data:", accuracy_score(y_test, y_pred_smote))

# Print classification reports
print("Classification report on original data:\n", classification_report(y_test, y_pred))
print("Classification report on SMOTE-balanced data:\n", classification_report(y_test, y_pred_smote))

if __name__ == "__main__":
    pass