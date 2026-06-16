import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generate synthetic data with bias
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, n_redundant=0, n_repeated=0, n_classes=2)

# Introduce bias by making feature 0 dependent on the target variable
X[:, 0] = y * np.random.randn(1000) + np.random.randn(1000)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# Print accuracy and confusion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)

# Detect bias by checking the importance of the biased feature
importances = model.coef_[0]
print("Feature Importances:", importances)

# Plot feature importances
plt.bar(range(10), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

if __name__ == "__main__":
    print("Running demo...")