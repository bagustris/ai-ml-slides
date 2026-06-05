# Hands-On AI Projects

This folder contains small Python examples for practicing basic machine learning
and explainable AI workflows.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu, install Tkinter if you want Matplotlib figures to open in an
interactive window:

```bash
sudo apt install python3-tk
```

## Running The Examples

Run the scripts in this order:

```bash
python3 1_diabetes_1.py
python3 2_breast_cancer.py
python3 3_diabetes_shap_1.py
```

The examples are:

1. `1_diabetes_1.py` - linear regression using one feature from the diabetes
   dataset.
2. `2_breast_cancer.py` - breast cancer classification using Gaussian Naive
   Bayes.
3. `3_diabetes_shap_1.py` - SHAP explanations for diabetes model predictions.

## Matplotlib Backend

If figures do not appear on Linux, run with an interactive backend:

```bash
MPLBACKEND=TkAgg python3 1_diabetes_1.py
```

For headless environments such as servers, CI, or remote shells without a GUI,
use `Agg`. The first script saves the plot to `1_diabetes_1_plot.png` when
`Agg` is active:

```bash
MPLBACKEND=Agg python3 1_diabetes_1.py
```

The SHAP script also saves plots when `Agg` is active:

- `3_diabetes_shap_1_summary.png`
- `3_diabetes_shap_1_bmi_dependence.png`
- `3_diabetes_shap_1_force_single.html`
- `3_diabetes_shap_1_force_all.html`

## Notebooks

The notebooks provide alternate interactive versions of the exercises:

- `diabetes_1.ipynb`
- `XAITraining.ipynb`
