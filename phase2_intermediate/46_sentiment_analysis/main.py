# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Load a sample dataset (we'll use iris for simplicity, but any text dataset would work)
# For sentiment analysis, we'd typically use a text dataset
iris = load_iris()
# Generate simple sentiment-like data (i.e., positive or negative text)
# We'll use the first 50 samples as positive and the next 50 as negative
texts = np.array(["I love this" if i < 50 else "I hate that" for i in range(100)])
labels = np.array([1 if i < 50 else 0 for i in range(100)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()

# Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Create a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier using the training data
clf.fit(X_train_count, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_count)

# Print the accuracy of the classifier
print("Accuracy:", clf.score(X_test_count, y_test))

if __name__ == "__main__":
    # Run the demo
    print("Texts:")
    print(texts)
    print("Labels:")
    print(labels)
    print("Predictions:")
    print(y_pred)