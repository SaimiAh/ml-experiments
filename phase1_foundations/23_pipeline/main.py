# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and SVC
# This pipeline will first scale the data and then classify it
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling is a common preprocessing step
    ('clf', SVC())  # Using SVM classifier
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    # Train the pipeline and make predictions
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Print accuracy score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Pipeline steps:", pipeline.steps)
    print("Pipeline named steps:", pipeline.named_steps)