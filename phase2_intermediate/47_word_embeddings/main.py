# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load 20 newsgroups dataset
def load_data():
    # Load dataset
    newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    return newsgroups.data, newsgroups.target_names

# Generate word embeddings using Word2Vec intuition (Bag-of-Words)
def generate_word_embeddings(data):
    # Create a CountVectorizer to generate word embeddings
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    return X.toarray(), vectorizer.get_feature_names_out()

# Apply dimensionality reduction using t-SNE
def reduce_dimensions(data, target_names):
    # Apply t-SNE to reduce dimensions
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

# Plot word embeddings
def plot_word_embeddings(data, target_names):
    # Plot data
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1])
    for i, target in enumerate(target_names):
        plt.annotate(target, (data[i, 0], data[i, 1]))
    plt.show()

if __name__ == "__main__":
    # Load data
    data, target_names = load_data()
    # Generate word embeddings
    word_embeddings, feature_names = generate_word_embeddings(data[:10])
    print("Word Embeddings Shape:", word_embeddings.shape)
    print("Feature Names:", feature_names[:5])
    # Reduce dimensions
    reduced_word_embeddings = reduce_dimensions(word_embeddings, target_names)
    print("Reduced Word Embeddings Shape:", reduced_word_embeddings.shape)
    # Plot word embeddings
    plot_word_embeddings(reduced_word_embeddings, target_names[:10])