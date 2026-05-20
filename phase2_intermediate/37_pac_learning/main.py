# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the test accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the test accuracy
print("Test Accuracy:", accuracy)

# PAC learning implies that the model will eventually learn the data
# As sample size increases, the model's accuracy should improve
sample_sizes = [10, 50, 100, 200, 500, 1000]
accuracies = []
for size in sample_sizes:
    X_train_pac, X_test_pac, y_train_pac, y_test_pac = train_test_split(X, y, test_size=0.2, train_size=size)
    model_pac = LogisticRegression()
    model_pac.fit(X_train_pac, y_train_pac)
    y_pred_pac = model_pac.predict(X_test_pac)
    accuracy_pac = accuracy_score(y_test_pac, y_pred_pac)
    accuracies.append(accuracy_pac)

# Plot the accuracy vs sample size
plt.plot(sample_sizes, accuracies)
plt.xlabel("Sample Size")
plt.ylabel("Accuracy")
plt.show()

if __name__ == "__main__":
    print("PAC Learning Demo")