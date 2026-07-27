# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
def load_data():
    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Kernel trick with linear and polynomial kernels
def kernel_trick(X, y):
    # Train a linear SVM model
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X, y)
    linear_accuracy = accuracy_score(y, linear_svm.predict(X))
    
    # Train a polynomial SVM model (using the kernel trick)
    poly_svm = SVC(kernel='poly', degree=2)
    poly_svm.fit(X, y)
    poly_accuracy = accuracy_score(y, poly_svm.predict(X))
    
    return linear_accuracy, poly_accuracy

if __name__ == "__main__":
    # Load iris dataset
    X, y = load_data()
    
    # Apply kernel trick
    linear_accuracy, poly_accuracy = kernel_trick(X, y)
    
    # Print the results
    print(f"Linear SVM Accuracy: {linear_accuracy}")
    print(f"Polynomial SVM Accuracy: {poly_accuracy}")
    
    # Visualize the first two features
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()