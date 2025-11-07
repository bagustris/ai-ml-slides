# diabetes_1.py

# 0. Load required packages/library
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# 1. Load dataset
X, y = datasets.load_diabetes(return_X_y=True)
print("Shape of Raw Input: ")
print(X.shape)
print("First Sample: ")
print(X[0])

# 2. Selecting the x-rd feature
X = X[:, 9]
print("Shape of feature (old, 1D):")
print(X.shape)

# Reshaping to get a 2D array
X = X.reshape(-1, 1)
print("Shape of feature (new, 2D):")
print(X.shape)

# 3. Split into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

# 4. Train
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Plot
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()
