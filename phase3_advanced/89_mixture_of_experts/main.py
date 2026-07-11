import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple Mixture of Experts model
class MixtureOfExperts:
    def __init__(self, n_experts, n_classes):
        self.n_experts = n_experts
        self.n_classes = n_classes
        self.experts = [LogisticRegression(max_iter=1000) for _ in range(n_experts)]

    def fit(self, X, y):
        for i in range(self.n_experts):
            # Each expert is trained on a subset of the data
            idx = np.random.choice(len(X), size=len(X), replace=True)
            self.experts[i].fit(X[idx], y[idx])

    def predict(self, X):
        predictions = np.zeros((len(X), self.n_classes))
        for i in range(self.n_experts):
            predictions += self.experts[i].predict_proba(X)
        return np.argmax(predictions, axis=1)

# Train a Mixture of Experts model with 3 experts
moe = MixtureOfExperts(n_experts=3, n_classes=3)
moe.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = moe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    print("Mixture of Experts Demo")
    print("Training Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    plt.show()