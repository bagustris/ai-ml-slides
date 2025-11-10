import time

import numpy as np
import shap
from sklearn.model_selection import train_test_split

X, y = shap.datasets.diabetes()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
X_train_summary = shap.kmeans(X_train, 10)


def print_error(f):
    print(f"Root mean squared test error = {np.sqrt(np.mean((f(X_test) - y_test) ** 2))}")
    time.sleep(0.5)  # to let the print get out before any progress bars


from sklearn import linear_model

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train, y_train)

print_error(lin_regr.predict)

# explain the model's single predictions using SHAP
ex = shap.KernelExplainer(lin_regr.predict, X_train_summary)
shap_values = ex.shap_values(X_test.iloc[0, :])
shap.force_plot(ex.expected_value, shap_values, X_test.iloc[0, :])

# explain the model's predictions on the whole test set
shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# plot the SHAP values for a single feature (bmi)
shap.dependence_plot("bmi", shap_values, X_test)

# force plot for the whole test set
shap.force_plot(ex.expected_value, shap_values, X_test)

# Challenge: try using a different models and see how the explanations differ!
# For example, try a decision tree, random forest, or NN.