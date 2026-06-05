# diabetes_1.py

# 0. Load required packages/library
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# 1. Load dataset
# X contains 10 standardized input features; y is the disease progression score.
X, y = datasets.load_diabetes(return_X_y=True)
print("Shape of Raw Input: ")
print(X.shape)
print("First Sample: ")
print(X[0])

# 2. Select the 3rd feature (BMI) so a simple 2D plot is possible.
X = X[:, 2]
print("Shape of feature (old, 1D):")
print(X.shape)

# scikit-learn estimators expect a 2D feature matrix: (n_samples, n_features).
X = X.reshape(-1, 1)
print("Shape of feature (new, 2D):")
print(X.shape)

# 3. Split into train and test
# random_state makes the result reproducible for classroom demonstrations.
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 4. Train
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Plot
# Sort by the x-axis before drawing the line; otherwise matplotlib connects
# predictions in random test-set order, producing a misleading zig-zag line.
sort_index = np.argsort(X_test.ravel())
plt.scatter(X_test, y_test, color="black", label="Actual test data")
plt.plot(
    X_test[sort_index],
    y_pred[sort_index],
    color="blue",
    linewidth=3,
    label="Prediction",
)
plt.xlabel("BMI (standardized)")
plt.ylabel("Diabetes progression")
plt.title("Linear regression using one diabetes feature")
plt.legend()
plt.show()
