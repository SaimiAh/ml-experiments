# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, n_repeated=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Plot a simple ROC curve (not exactly applicable but for demonstration)
y_pred_proba = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 100)
tpr = [sum((y_pred_proba >= t) & (y_test == 1)) / sum(y_test == 1) for t in thresholds]
fpr = [sum((y_pred_proba >= t) & (y_test == 0)) / sum(y_test == 0) for t in thresholds]
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

if __name__ == "__main__":
    print("LLM Evaluation Metrics Demo")