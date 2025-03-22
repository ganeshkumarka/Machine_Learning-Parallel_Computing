import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models for feature reduction
reducers = {
    "PCA_2": PCA(n_components=2),
    "PCA_3": PCA(n_components=3),
    "LDA_2": LDA(n_components=2),
    "LDA_3": LDA(n_components=3),
    "TSNE_2": TSNE(n_components=2, random_state=42),
    "TSNE_3": TSNE(n_components=3, random_state=42),
    "SVD_2": TruncatedSVD(n_components=2),
    "SVD_3": TruncatedSVD(n_components=3)
}

# Define classifier and cross-validation
clf = SVC(kernel='linear')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform feature extraction and cross-validation
results = {}
for name, reducer in reducers.items():
    try:
        X_reduced = reducer.fit_transform(X_scaled, y) if 'LDA' in name else reducer.fit_transform(X_scaled)
        scores = cross_val_score(clf, X_reduced, y, cv=skf, scoring='accuracy')
        results[name] = scores.mean()
        print(f"{name}: Accuracy = {scores.mean():.4f}")
    except Exception as e:
        print(f"{name} failed: {e}")

# Print best performing method
best_method = max(results, key=results.get)
print(f"\nBest performing method: {best_method} with accuracy {results[best_method]:.4f}")
