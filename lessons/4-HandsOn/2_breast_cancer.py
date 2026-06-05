# 2_breast_cancer.py
# Reference: https://www.geeksforgeeks.org/machine-learning/ml-cancer-cell-classification-using-scikit-learn/

# 0. import library
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()

# 1. Exploring the dataset with Pandas
# Each row is one tumor sample; each column is a measured cell feature.
df = pd.DataFrame(data.data, columns=data.feature_names)

# Print the first 5 samples so students see the exact same rows each run.
print("First 5 samples of data:")
print(df.head(5))

print("Info of data:")
df.info()

print("Statistics of data:")
print(df.describe())

# Analyze data.target to understand the distribution of malignant and benign
# cases, since class imbalance can affect model performance.
target_names = {index: name for index, name in enumerate(data.target_names)}
df2 = pd.DataFrame(data.target, columns=["target"])
df2["diagnosis"] = df2["target"].map(target_names)

print("Class label mapping:")
print(target_names)

class_counts = df2["diagnosis"].value_counts()
plt.pie(
    class_counts,
    labels=class_counts.index,
    autopct="%1.2f%%",
    colors=["green", "red"],
)
plt.title("Breast cancer diagnosis distribution")
plt.savefig("2_breast_cancer_class_distribution.png")

# Split dataset into training and testing sets
# stratify keeps the malignant/benign ratio similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.33, random_state=42, stratify=data.target
)

# Train the Gaussian Naive Bayes model
model = GaussianNB()  # Try other models: MLP, SVC, DecisionTree, RandomForest, etc.
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#
