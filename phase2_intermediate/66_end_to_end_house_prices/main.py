# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for house price prediction
# We're generating 1000 samples, with 10 features, and a target variable (house price)
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Convert data to pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 11)])
df['price'] = y

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.show()

if __name__ == "__main__":
    print("House Price Prediction Model")
    print("-------------------------------")
    print("Mean Squared Error:", mse)