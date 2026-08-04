import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Introduce missing values
df_missing = df.copy()
df_missing.iloc[0:10, 0] = np.nan  # introduce missing values in column 0

# Strategy 1: Drop rows with missing values
df_drop = df_missing.dropna()

# Strategy 2: Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
df_impute = pd.DataFrame(imputer.fit_transform(df_missing), columns=df_missing.columns)

# Compare strategies
X_drop = df_drop.drop('target', axis=1)
y_drop = df_drop['target']
X_impute = df_impute.drop('target', axis=1)
y_impute = df_impute['target']

X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_drop, y_drop, test_size=0.2, random_state=42)
X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(X_impute, y_impute, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train_drop, y_train_drop)
y_pred_drop = model.predict(X_test_drop)
model.fit(X_train_impute, y_train_impute)
y_pred_impute = model.predict(X_test_impute)

print("Drop rows with missing values: ", accuracy_score(y_test_drop, y_pred_drop))
print("Impute missing values with mean: ", accuracy_score(y_test_impute, y_pred_impute))

if __name__ == "__main__":
    # Run demo
    print("Running demo...")
    load_iris()
    print("Iris dataset loaded.")
    print("Missing values introduced and strategies compared.")
    print("Accuracy scores printed above.")