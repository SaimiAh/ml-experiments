# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate synthetic data for classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbc.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    # Run the demo
    print("Gradient Boosting Classifier Demo")
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
    print("Number of Estimators:", gbc.n_estimators)
    print("Learning Rate:", gbc.learning_rate)

    # Plot feature importances
    feature_importances = gbc.feature_importances_
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.show()