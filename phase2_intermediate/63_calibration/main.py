# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=3, n_classes=2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create and train a calibrated classifier
calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
calibrated_model.fit(X_train, y_train)

# Print accuracy and calibration before and after calibration
print("Model Accuracy:", model.score(X_test, y_test))
print("Calibrated Model Accuracy:", calibrated_model.score(X_test, y_test))

# Plot calibration curves
y_pred = model.predict_proba(X_test)[:, 1]
y_pred_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Plot data
plt.figure(figsize=(10, 5))
plt.hist(y_pred, bins=10, alpha=0.5, label='Predicted Probabilities')
plt.hist(y_pred_calibrated, bins=10, alpha=0.5, label='Calibrated Probabilities')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("Model calibration example using synthetic data.")
    print("Running the script will display two histograms showing predicted probabilities and calibrated probabilities.")