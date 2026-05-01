# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()

# Create a dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Print original dataframe
print("Original DataFrame:")
print(df.head())

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
ohe_target = ohe.fit_transform(df[['target']])

# Print One-Hot Encoded dataframe
print("\nOne-Hot Encoded DataFrame:")
print(ohe_target[:5])

# Label Encoding
le = LabelEncoder()
le_target = le.fit_transform(df['target'])

# Print Label Encoded dataframe
print("\nLabel Encoded DataFrame:")
print(le_target[:5])

# Plot the data
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=le_target)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset')
plt.show()

if __name__ == "__main__":
    print("One-Hot Encoding and Label Encoding Demo")