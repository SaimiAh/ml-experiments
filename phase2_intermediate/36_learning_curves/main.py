# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load iris dataset
def load_data():
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Train model and plot learning curve
def plot_learning_curve(X, y):
    # Initialize model
    model = LogisticRegression(max_iter=1000)
    
    # Generate learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_sizes, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.plot(train_sizes, train_mean, color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), color='red', label='Cross-validation Score')
    plt.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1), np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.2, color='red')
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = load_data()
    plot_learning_curve(X, y)
    print("Learning Curve has been plotted. High bias is indicated by low training score and low variance is indicated by small gap between training and cross-validation scores.")