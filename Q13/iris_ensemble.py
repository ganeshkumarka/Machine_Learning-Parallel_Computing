from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Folder name
folder_name = "clustering_algorithms"
os.makedirs(folder_name, exist_ok=True)

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis
iris_df = pd.DataFrame(X, columns=data.feature_names)
iris_df['species'] = y
sns.pairplot(iris_df, hue='species', palette='tab10')
plt.show()

# List of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "MLP Classifier": MLPClassifier(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SGD": SGDClassifier(max_iter=1000, random_state=42)
}

# Evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = (accuracy, precision, recall, f1)
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Select best 4 classifiers based on accuracy
top_classifiers = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:4]
best_models = [classifiers[name] for name, _ in top_classifiers]

# Ensemble methods
bagging = BaggingClassifier(estimator=best_models[0], n_estimators=10, random_state=42)
boosting = AdaBoostClassifier(estimator=best_models[1], n_estimators=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
stacking = StackingClassifier(estimators=[(name, clf) for name, clf in zip([t[0] for t in top_classifiers], best_models)], final_estimator=LogisticRegression())

ensembles = {
    "Bagging": bagging,
    "Boosting": boosting,
    "Gradient Boosting": gb,
    "Stacking": stacking
}

# Evaluate ensemble methods
for name, ensemble in ensembles.items():
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Ensemble - Accuracy: {accuracy:.4f}")
