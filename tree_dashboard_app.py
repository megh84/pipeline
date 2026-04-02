import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tree_pipelines import (
    generate_dataset,
    load_or_generate_data,
    evaluate_regression,
    evaluate_classification,
)

st.set_page_config(page_title="Tree Model Pipelines Dashboard", layout="wide")

st.title("Tree Model Pipelines Dashboard")
st.write(
    "This dashboard compares five tree-based pipelines for both a continuous target "
    "(regression tree pipelines) and a binary target (classification tree pipelines)."
)

DEFAULT_PATH = "simulated_dataset.csv"

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    use_generated = st.checkbox("Generate sample data if CSV is unavailable", value=True)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

def read_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if os.path.exists(DEFAULT_PATH):
        return pd.read_csv(DEFAULT_PATH)

    if use_generated:
        df = generate_dataset(n_samples=500, n_predictors=50, random_state=random_state)
        df.to_csv(DEFAULT_PATH, index=False)
        return df

    st.stop()

df = read_data()

required_cols = {"y_continuous", "y_binary"}
if not required_cols.issubset(set(df.columns)):
    st.error("Dataset must contain y_continuous and y_binary columns.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Predictors", df.shape[1] - 2)

reg_results, _ = evaluate_regression(df, test_size=test_size, random_state=random_state)
clf_results, _ = evaluate_classification(df, test_size=test_size, random_state=random_state)

tab1, tab2, tab3 = st.tabs(["Regression Pipelines", "Classification Pipelines", "Summary"])

with tab1:
    st.subheader("Regression Tree Pipelines")
    st.dataframe(reg_results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(reg_results["Pipeline"], reg_results["RMSE"])
    ax.set_title("Regression Comparison by RMSE")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

    st.markdown(
        """
        **Included regression pipelines**
        - Decision Tree Regressor
        - Random Forest Regressor
        - Extra Trees Regressor
        - Gradient Boosting Regressor
        - HistGradient Boosting Regressor
        """
    )

with tab2:
    st.subheader("Classification Tree Pipelines")
    st.dataframe(clf_results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(clf_results["Pipeline"], clf_results["ROC_AUC"])
    ax.set_title("Classification Comparison by ROC-AUC")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

    st.markdown(
        """
        **Included classification pipelines**
        - Decision Tree Classifier
        - Random Forest Classifier
        - Extra Trees Classifier
        - Gradient Boosting Classifier
        - HistGradient Boosting Classifier
        """
    )

with tab3:
    st.subheader("Best Models at a Glance")

    best_reg = reg_results.iloc[0]
    best_clf = clf_results.iloc[0]

    left, right = st.columns(2)
    with left:
        st.success(
            f"Best regression pipeline: **{best_reg['Pipeline']}** "
            f"(RMSE = {best_reg['RMSE']:.3f}, R² = {best_reg['R2']:.3f})"
        )
    with right:
        st.success(
            f"Best classification pipeline: **{best_clf['Pipeline']}** "
            f"(ROC-AUC = {best_clf['ROC_AUC']:.3f}, Accuracy = {best_clf['Accuracy']:.3f})"
        )

    merged = reg_results[["Pipeline", "RMSE", "R2"]].merge(
        clf_results[["Pipeline", "Accuracy", "F1", "ROC_AUC"]],
        on="Pipeline",
        how="inner",
    )
    st.dataframe(merged, use_container_width=True)

st.caption("Built with Streamlit and scikit-learn. The app supports both regression-tree and classification-tree pipelines.")
