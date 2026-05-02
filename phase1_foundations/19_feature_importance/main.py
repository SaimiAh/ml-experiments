# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = rf.feature_importances_

# Print feature importance
print("Feature Importance:")
for i, feature in enumerate(iris.feature_names):
    print(f"{feature}: {feature_importance[i]:.3f}")

# Plot feature importance
plt.bar(iris.feature_names, feature_importance)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

if __name__ == "__main__":
    # Train and print feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importance = rf.feature_importances_
    for i, feature in enumerate(iris.feature_names):
        print(f"{feature}: {feature_importance[i]:.3f}")