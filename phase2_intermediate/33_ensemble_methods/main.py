import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
log_reg = LogisticRegression(max_iter=1000)
dec_tree = DecisionTreeClassifier()

# Define ensemble model using stacking
ensemble = VotingClassifier(estimators=[('logreg', log_reg), ('decision_tree', dec_tree)])

# Train base models and ensemble model
log_reg.fit(X_train, y_train)
dec_tree.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

# Make predictions
y_pred_logreg = log_reg.predict(X_test)
y_pred_dectree = dec_tree.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)

# Evaluate models
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dectree))
print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))

if __name__ == "__main__":
    print("Starting ensemble demo...")
    # Print accuracy scores
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dectree))
    print("Ensemble Model Accuracy:", accuracy_score(y_test, y_pred_ensemble))