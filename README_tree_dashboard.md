# Tree Model Pipelines Dashboard

This package includes Python code for **five tree-model pipelines** and a **Streamlit dashboard**
to compare them for both:

- a **continuous response** (`y_continuous`) using regression models
- a **binary response** (`y_binary`) using classification models

## Included pipeline families

1. Decision Tree
2. Random Forest
3. Extra Trees
4. Gradient Boosting
5. HistGradient Boosting

Each family is implemented in both regression and classification form.

## Files

- `tree_dashboard_app.py` → Streamlit dashboard
- `tree_pipelines.py` → all pipeline code and evaluation functions
- `requirements_tree_dashboard.txt` → dependencies
- `simulated_dataset.csv` → dataset

## How to run

```bash
pip install -r requirements_tree_dashboard.txt
streamlit run tree_dashboard_app.py
```

## Dataset requirements

Your CSV should contain:
- predictor columns
- `y_continuous`
- `y_binary`

If `simulated_dataset.csv` is not found, the dashboard can generate a sample dataset automatically.

## What the dashboard shows

- dataset preview
- regression pipeline comparison using MAE, RMSE, and R²
- classification pipeline comparison using Accuracy, F1, and ROC-AUC
- best pipeline summary
