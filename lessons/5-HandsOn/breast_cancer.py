# breast_cancer.py
# Reference: https://www.geeksforgeeks.org/machine-learning/ml-cancer-cell-classification-using-scikit-learn/

# 0. import library
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()

# 1. Exploring the dataset with Pandas
df=pd.DataFrame(data.data,columns=data.feature_names)

# print first 4 samples
print("First 5 samples of data:")
print(df.sample(5))

print("Info of data:")
print(df.info())

print("Statistics of data:")
print(df.describe())

# Analyze data.target to understand the distribution of malignant and benign cases as class imbalance can affect model performance.
df2=pd.DataFrame(data.target,columns=['target'])
df2.sample(5)

class_counts=df2["target"].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.2f%%', colors=['red', 'green'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

# Train the Gaussian Naive Bayes model
model = GaussianNB() # ganti model yang lain: MLP, SVC, DecisionTree, RandomForest, dll
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#