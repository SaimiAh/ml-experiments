# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(0)
data = np.cumsum(np.random.normal(size=100))

# Create a pandas Series
series = pd.Series(data)

# Plot the original time series data
plt.figure(figsize=(10,6))
plt.plot(series)
plt.title('Original Time Series')
plt.show()

# Split data into training and testing sets
train, test = train_test_split(series, test_size=0.2, shuffle=False)

# Create and fit the ARIMA model
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Generate forecast for the test data
forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')

# Print the forecasted values
print("Forecasted values:", forecast)

# Print the mean squared error
mse = mean_squared_error(test, forecast)
print("Mean Squared Error:", mse)

# Plot the forecasted values
plt.figure(figsize=(10,6))
plt.plot(train, label='Training')
plt.plot([None for i in train] + [x for x in test], label='Testing')
plt.plot([None for i in train] + [x for x in forecast], label='Forecast', color='red')
plt.title('Time Series Forecast')
plt.legend()
plt.show()