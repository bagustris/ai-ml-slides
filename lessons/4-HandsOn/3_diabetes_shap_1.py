import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent


def finish_plot(filename):
    """Save plots in headless mode, otherwise show them interactively."""
    if plt.get_backend().lower() == "agg":
        output_path = SCRIPT_DIR / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close()
    else:
        plt.show()


def save_force_plot(filename, force_plot):
    """Write SHAP force plots to HTML files for script-based runs."""
    output_path = SCRIPT_DIR / filename
    with output_path.open("w", encoding="utf-8") as html_file:
        shap.save_html(html_file, force_plot)
    print(f"Saved force plot to {output_path}")

# X is a pandas DataFrame of diabetes features; y is the target progression score.
X, y = shap.datasets.diabetes()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Kernel SHAP can be slow if it uses every training row as the background data.
# kmeans creates 10 representative background points for a faster classroom demo.
X_train_summary = shap.kmeans(X_train, 10)


def print_error(predict_fn):
    """Print RMSE so students can connect model accuracy with explanations."""
    y_pred = predict_fn(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"Root mean squared test error = {rmse:.2f}")
    time.sleep(0.5)  # to let the print get out before any progress bars


from sklearn import linear_model

lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train, y_train)

print_error(lin_regr.predict)

# Explain one prediction first; this shows how each feature moves the prediction
# away from the model's average prediction.
ex = shap.KernelExplainer(lin_regr.predict, X_train_summary)
shap_values = ex.shap_values(X_test.iloc[0, :])
single_force_plot = shap.force_plot(ex.expected_value, shap_values, X_test.iloc[0, :])
save_force_plot("3_diabetes_shap_1_force_single.html", single_force_plot)

# Explain the model's predictions on the whole test set.
shap_values = ex.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
finish_plot("3_diabetes_shap_1_summary.png")

# plot the SHAP values for a single feature (bmi)
shap.dependence_plot("bmi", shap_values, X_test, show=False)
finish_plot("3_diabetes_shap_1_bmi_dependence.png")

# Force plot for the whole test set. In a notebook this renders inline; when
# running as a script, save the returned HTML object if you need a shareable file.
force_plot = shap.force_plot(ex.expected_value, shap_values, X_test)
save_force_plot("3_diabetes_shap_1_force_all.html", force_plot)

# Challenge: try using a different models and see how the explanations differ!
# For example, try a decision tree, random forest, or NN.
