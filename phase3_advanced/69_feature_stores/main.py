# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# What: Feature stores are centralized repositories that store and manage features for machine learning models
# Why: They simplify feature engineering, reduce data duplication, and improve model reproducibility

if __name__ == "__main__":
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a simple feature store with pandas DataFrame
    feature_store = pd.DataFrame(X_train, columns=iris.feature_names)

    # Print feature store
    print("Feature Store:")
    print(feature_store.head())

    # Train a logistic regression model using the feature store
    model = LogisticRegression(max_iter=1000)
    model.fit(feature_store, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Plot the first two features
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()