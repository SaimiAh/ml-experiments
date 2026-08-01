# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train Naive Bayes classifier
def train_model(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

# Evaluate model
def evaluate_model(gnb, X_test, y_test):
    y_pred = gnb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main function
if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gnb = train_model(X_train, y_train)
    evaluate_model(gnb, X_test, y_test)