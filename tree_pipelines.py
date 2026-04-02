import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)

RANDOM_STATE = 42


def generate_dataset(n_samples=500, n_predictors=50, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_predictors))
    predictor_names = [f"X{i+1}" for i in range(n_predictors)]

    y_cont = (
        3 * X[:, 0]
        - 2 * X[:, 1]
        + 1.5 * X[:, 2]
        + 0.8 * X[:, 3] * X[:, 4]
        - 1.2 * (X[:, 5] ** 2)
        + rng.normal(0, 2, n_samples)
    )

    logit = (
        1.2 * X[:, 0]
        - 1.5 * X[:, 1]
        + 1.0 * X[:, 6]
        - 0.7 * X[:, 7] * X[:, 8]
        + 0.5 * (X[:, 9] > 0).astype(int)
    )
    prob = 1 / (1 + np.exp(-logit))
    y_bin = rng.binomial(1, prob, n_samples)

    df = pd.DataFrame(X, columns=predictor_names)
    df["y_continuous"] = y_cont
    df["y_binary"] = y_bin
    return df


def load_or_generate_data(csv_path="simulated_dataset.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = generate_dataset()
        df.to_csv(csv_path, index=False)
    return df


def get_feature_columns(df, continuous_target="y_continuous", binary_target="y_binary"):
    return [c for c in df.columns if c not in [continuous_target, binary_target]]


def make_preprocessor(feature_columns):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_columns),
        ],
        remainder="drop",
    )


def get_regression_pipelines(feature_columns, random_state=RANDOM_STATE):
    preprocessor = make_preprocessor(feature_columns)

    return {
        "Decision Tree": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state,
                )),
            ]
        ),
        "HistGradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=250,
                    random_state=random_state,
                )),
            ]
        ),
    }


def get_classification_pipelines(feature_columns, random_state=RANDOM_STATE):
    preprocessor = make_preprocessor(feature_columns)

    return {
        "Decision Tree": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=random_state,
                )),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state,
                )),
            ]
        ),
        "HistGradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=250,
                    random_state=random_state,
                )),
            ]
        ),
    }


def evaluate_regression(df, test_size=0.2, random_state=RANDOM_STATE):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["y_continuous"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipelines = get_regression_pipelines(feature_cols, random_state=random_state)
    rows = []

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        rows.append({
            "Pipeline": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse,
            "R2": r2_score(y_test, preds),
        })

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, pipelines


def evaluate_classification(df, test_size=0.2, random_state=RANDOM_STATE):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["y_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipelines = get_classification_pipelines(feature_cols, random_state=random_state)
    rows = []

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        try:
            proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                proba = 1 / (1 + np.exp(-scores))
            else:
                proba = preds

        rows.append({
            "Pipeline": name,
            "Accuracy": accuracy_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "ROC_AUC": roc_auc_score(y_test, proba),
        })

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    return results, pipelines
