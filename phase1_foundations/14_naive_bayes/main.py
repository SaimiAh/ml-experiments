# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load iris dataset
# We will use the iris dataset as it's a classic multi-class classification problem
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
# We'll use 80% of the data for training and the remaining 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
# We'll use Gaussian Naive Bayes as it's suitable for continuous data
gnb = GaussianNB()

# Train the classifier
# We'll train the classifier using the training data
gnb.fit(X_train, y_train)

# Make predictions on the test set
# We'll use the trained classifier to make predictions on the test data
y_pred = gnb.predict(X_test)

# Evaluate the classifier
# We'll evaluate the classifier by calculating its accuracy and printing a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

if __name__ == "__main__":
    # Run the code
    print("Running Naive Bayes classifier demo...")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    # Plot a simple histogram to visualize the data
    plt.hist(X[:, 0], bins=10)
    plt.title("Sepal length distribution")
    plt.show()