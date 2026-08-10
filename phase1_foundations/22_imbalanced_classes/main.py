# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Generate synthetic imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, n_classes=2, weights=[0.1, 0.9])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to oversample minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print counts of each class before and after SMOTE
print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_smote))

# Train a model on original and SMOTE datasets
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model_smote = LogisticRegression()
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

# Print classification reports
print("Original Dataset:")
print(classification_report(y_test, y_pred))
print("SMOTE Dataset:")
print(classification_report(y_test, y_pred_smote))