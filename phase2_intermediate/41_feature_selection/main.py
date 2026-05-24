# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split the data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Perform feature selection
def select_features(X_train, y_train, k):
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected

# Train a logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Compare feature selection techniques
def compare_feature_selection():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    k_values = range(1, 5)
    accuracies = []
    
    for k in k_values:
        X_train_selected = select_features(X_train, y_train, k)
        X_test_selected = SelectKBest(chi2, k=k).fit_transform(X_train, y_train).transform(X_test)
        model = train_model(X_train_selected, y_train)
        accuracy = evaluate_model(model, X_test_selected, y_test)
        accuracies.append(accuracy)
        print(f'k={k}, Accuracy: {accuracy:.3f}')
        
    plt.plot(k_values, accuracies)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    compare_feature_selection()