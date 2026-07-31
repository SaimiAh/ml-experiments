# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    # Load iris dataset
    iris = datasets.load_iris()
    # Convert to pandas dataframe
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

# Train SVM model
def train_model(df):
    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create SVM classifier
    classifier = svm.SVC()
    # Train the model
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test

# Evaluate SVM model
def evaluate_model(classifier, X_test, y_test):
    # Predict the response for test dataset
    y_pred = classifier.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Other metrics
    print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))

# Main function
if __name__ == "__main__":
    # Load data
    df = load_data()
    # Train model
    classifier, X_test, y_test = train_model(df)
    # Evaluate model
    evaluate_model(classifier, X_test, y_test)