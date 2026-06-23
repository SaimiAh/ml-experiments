# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset and create a dataframe
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

# Split data into training and testing sets
def split_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

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

# A/B testing
def ab_testing(df):
    # Version A: Using all features
    X_train, X_test, y_train, y_test = split_data(df)
    model_a = train_model(X_train, y_train)
    accuracy_a = evaluate_model(model_a, X_test, y_test)

    # Version B: Using only two features
    df_b = df[['sepal length (cm)', 'sepal width (cm)', 'target']]
    X_train_b, X_test_b, y_train_b, y_test_b = split_data(df_b)
    model_b = train_model(X_train_b, y_train_b)
    accuracy_b = evaluate_model(model_b, X_test_b, y_test_b)

    print(f"Accuracy of Version A: {accuracy_a:.3f}")
    print(f"Accuracy of Version B: {accuracy_b:.3f}")

    # Plotting the results
    labels = ['Version A', 'Version B']
    accuracy = [accuracy_a, accuracy_b]
    plt.bar(labels, accuracy)
    plt.xlabel('Version')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    df = load_data()
    ab_testing(df)