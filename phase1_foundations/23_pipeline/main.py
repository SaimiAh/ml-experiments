import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Split data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Create a pipeline with StandardScaler and LogisticRegression
def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # scale data
        ('classifier', LogisticRegression())  # classify data
    ])
    return pipeline

# Train the pipeline and make predictions
def train_and_predict(pipeline, X_train, X_test, y_train):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return y_pred

# Evaluate the pipeline
def evaluate_pipeline(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = create_pipeline()
    y_pred = train_and_predict(pipeline, X_train, X_test, y_train)
    evaluate_pipeline(y_test, y_pred)