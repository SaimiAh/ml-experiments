import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform data
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create a DataFrame with transformed data
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['species'] = y

# Plot data
plt.figure(figsize=(8, 6))
for species in np.unique(y):
    plt.scatter(X_pca[y == species, 0], X_pca[y == species, 1], label=iris.target_names[species])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset with PCA')
plt.legend()
plt.show()

if __name__ == "__main__":
    print("PCA Experiment Completed")