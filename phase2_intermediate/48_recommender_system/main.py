# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Generate synthetic user-item interaction data
np.random.seed(0)
user_item_data = np.random.randint(0, 2, size=(10, 5))  # 10 users, 5 items

# Create a pandas dataframe for better readability
df = pd.DataFrame(user_item_data, columns=['Item1', 'Item2', 'Item3', 'Item4', 'Item5'])

# Print the user-item interaction data
print("User-Item Interaction Data:")
print(df)

# Create a nearest neighbors model for collaborative filtering
model = NearestNeighbors(n_neighbors=3)

# Fit the model to the user-item interaction data
model.fit(user_item_data)

# Find the nearest neighbors for a given user (e.g., user 0)
distances, indices = model.kneighbors([user_item_data[0]])

# Print the recommended items for the given user
print("\nRecommended Items for User 0:")
print("Indices:", indices[0])
print("Distances:", distances[0])

# Use the indices to get the recommended item names
recommended_items = df.columns[indices[0]]
print("Recommended Item Names:", recommended_items)

if __name__ == "__main__":
    pass  # No need for extra code here, the script runs standalone