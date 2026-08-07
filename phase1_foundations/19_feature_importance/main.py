# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    """Load iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train a random forest classifier
def train_model(X_train, y_train):
    """Train a random forest classifier"""
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Get feature importance
def get_feature_importance(model, feature_names):
    """Get feature importance"""
    importance = model.feature_importances_
    return dict(zip(feature_names, importance))

# Main function
if __name__ == "__main__":
    X, y = load_data()
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    importance = get_feature_importance(model, feature_names)
    
    # Print feature importance
    print("Feature Importance:")
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.2f}")
    
    # Plot feature importance
    plt.bar(importance.keys(), importance.values())
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()