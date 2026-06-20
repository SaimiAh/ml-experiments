# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load iris dataset
def load_data():
    """Loads the iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

# Define ETL pipeline
def etl_pipeline(df):
    """Defines the ETL pipeline"""
    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data using StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_test, y_test

# Main function
if __name__ == "__main__":
    # Load data
    df = load_data()
    print("Data Loaded:")
    print(df.head())
    
    # Define and run ETL pipeline
    pipeline, X_test, y_test = etl_pipeline(df)
    print("\nETL Pipeline Built and Run")
    print("Model Score:", pipeline.score(X_test, y_test))