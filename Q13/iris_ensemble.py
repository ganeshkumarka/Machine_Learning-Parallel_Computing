import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of classifiers to evaluate
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Evaluate classifiers and store results
results = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append((name, accuracy, precision, recall, f1))

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 Score"])
print("Classifier Performance:")
print(results_df)

# Select the best 4 classifiers based on F1 Score
best_classifiers = results_df.nlargest(4, 'F1 Score')['Classifier'].values
print("\nBest 4 Classifiers based on F1 Score:")
print(best_classifiers)

# Initialize the selected classifiers
selected_classifiers = [classifiers[name] for name in best_classifiers]

# Apply Bagging
print("\nApplying Bagging...")
for clf in selected_classifiers:
    bagging = BaggingClassifier(base_estimator=clf, n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Bagging with {clf.__class__.__name__}: Accuracy = {accuracy:.4f}")

# Apply Boosting (AdaBoost)
print("\nApplying Boosting...")
for clf in selected_classifiers:
    boosting = AdaBoostClassifier(base_estimator=clf, n_estimators=10, random_state=42)
    boosting.fit(X_train, y_train)
    y_pred = boosting.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Boosting with {clf.__class__.__name__}: Accuracy = {accuracy:.4f}")

# Apply Stacking
print("\nApplying Stacking...")
estimators = [(name, classifiers[name]) for name in best_classifiers]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking: Accuracy = {accuracy:.4f}")
