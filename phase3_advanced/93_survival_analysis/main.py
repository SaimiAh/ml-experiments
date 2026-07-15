# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Generate synthetic data for regression
# We will use this data to create a simple survival analysis model
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Create a simple survival time variable
# For simplicity, let's assume that the event occurs when y > 10
event = (y > 10).astype(int)
time_to_event = np.where(event == 1, y, 20 - y)  # arbitrary time-to-event values

# Create a dataframe
df = pd.DataFrame({'X': X.flatten(), 'y': y, 'event': event, 'time_to_event': time_to_event})

# Print the first few rows of the dataframe
print("Dataframe:")
print(df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, time_to_event, test_size=0.2, random_state=42)

# Create a linear regression model for time-to-event prediction
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the mean absolute error of the model
print("\nMean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Plot the data
plt.scatter(X, time_to_event)
plt.xlabel('X')
plt.ylabel('Time to Event')
plt.show()