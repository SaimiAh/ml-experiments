# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a Support Vector Machine classifier
classifier = svm.SVC(kernel='linear')  # using linear kernel

# Train the classifier
classifier.fit(X_train, y_train)

# Print accuracy
print("Accuracy:", classifier.score(X_test, y_test))

# Plot the decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
w = classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
yy = a * xx - (classifier.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')
plt.show()

if __name__ == "__main__":
    print("Support Vector Machine experiment")