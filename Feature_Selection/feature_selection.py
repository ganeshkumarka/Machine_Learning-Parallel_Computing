import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Perform exploratory data analysis (EDA)
print("Dataset Head:")
print(X.head())
print("\nDataset Description:")
print(X.describe())
print("\nClass Distribution:")
print(y.value_counts())

# Visualize the dataset
sns.pairplot(pd.concat([X, y], axis=1), hue='target')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Univariate Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Feature Importance using Random Forest
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importances (Random Forest):")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# Recursive Feature Elimination (RFE) using Support Vector Machine (SVM)
svc = SVC(kernel="linear")
rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Evaluate the performance of the selected features using a classification model
# Using Logistic Regression for evaluation
log_reg = LogisticRegression(max_iter=1000)

# Without feature selection
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy_all_features = accuracy_score(y_test, y_pred)
print(f"\nAccuracy with all features: {accuracy_all_features:.4f}")

# With Univariate Feature Selection
log_reg.fit(X_train_selected, y_train)
y_pred = log_reg.predict(X_test_selected)
accuracy_univariate = accuracy_score(y_test, y_pred)
print(f"Accuracy with Univariate Feature Selection: {accuracy_univariate:.4f}")

# With RFE
log_reg.fit(X_train_rfe, y_train)
y_pred = log_reg.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred)
print(f"Accuracy with RFE: {accuracy_rfe:.4f}")

# Compare the model performance before and after feature selection
print("\nModel Performance Comparison:")
print(f"All Features: {accuracy_all_features:.4f}")
print(f"Univariate Feature Selection: {accuracy_univariate:.4f}")
print(f"RFE: {accuracy_rfe:.4f}")
