# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Introduce missing values
X_missing = X.copy()
X_missing[np.random.randint(0, X.shape[0], 10), np.random.randint(0, X.shape[1], 10)] = np.nan

# Create a DataFrame
df = pd.DataFrame(X_missing, columns=iris.feature_names)

# Print the number of missing values
print("Missing values count:", df.isnull().sum())

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_missing)

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

if __name__ == "__main__":
    # Run the demo
    print("Demo Output:")
    print("Missing values count:", df.isnull().sum())
    print("Mean Squared Error:", mse)