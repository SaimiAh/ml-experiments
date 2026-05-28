import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

# Load the dataset
dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the data and transform it into a matrix of TF-IDF features
X = vectorizer.fit_transform(dataset.data)

# Print the shape of the transformed data
print("Shape of transformed data:", X.shape)

# Print the feature names (i.e., the words in the vocabulary)
print("Feature names:", vectorizer.get_feature_names_out()[:10])

# Get the document with the most features (i.e., the longest document)
doc_lengths = np.array(X.sum(axis=1)).ravel()
longest_doc_idx = np.argmax(doc_lengths)
print("Longest document index:", longest_doc_idx)
print("Longest document length:", doc_lengths[longest_doc_idx])

if __name__ == "__main__":
    print("Demo:")
    print("Shape of transformed data:", X.shape)
    print("Feature names:", vectorizer.get_feature_names_out()[:10])
    print("Longest document index:", longest_doc_idx)
    print("Longest document length:", doc_lengths[longest_doc_idx])