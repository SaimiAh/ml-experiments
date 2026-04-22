# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generate synthetic data
X, y = make_circles(n_samples=200, factor=.2, noise=.05, random_state=0)

# Explain the kernel trick: instead of computing the dot product in the original space
# we compute it in a higher-dimensional space using a kernel function
# Here, we use the radial basis function (RBF) kernel

# Train a SVM model with the RBF kernel
svm = SVC(kernel='rbf', gamma=1)
svm.fit(X, y)

# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Kernel Method: SVM with RBF kernel")
plt.show()

# Print the support vectors
print("Support Vectors:")
print(svm.support_vectors_)

if __name__ == "__main__":
    print("Kernel Methods Demo")
    print("------------------")
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Number of support vectors:", len(svm.support_vectors_))