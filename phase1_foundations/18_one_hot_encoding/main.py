# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import load_iris

# Load iris dataset
def load_data():
    """Load iris dataset"""
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

# One-hot encoding and label encoding
def encode_data(X, y, feature_names, target_names):
    """One-hot encoding and label encoding"""
    # Create a DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Label encoding for target variable
    le = LabelEncoder()
    df['target_le'] = le.fit_transform(df['target'])
    
    # One-hot encoding for a categorical feature (assuming one of the features is categorical)
    # For demonstration purposes, we'll create a categorical feature
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(X))
    ohe = OneHotEncoder(sparse=False)
    ohe_df = pd.DataFrame(ohe.fit_transform(df[['category']]), columns=ohe.get_feature_names_out())
    df = pd.concat([df, ohe_df], axis=1)
    
    return df

# Main function
if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    df = encode_data(X, y, feature_names, target_names)
    print("Original target values:")
    print(np.unique(y))
    print("\nLabel encoded target values:")
    print(np.unique(df['target_le']))
    print("\nOne-hot encoded category values:")
    print(df.iloc[:, -3:].head())