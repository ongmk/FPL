from datetime import datetime
from functools import partial
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.pipeline import Pipeline
from typing import Any
from src.fpl.pipelines.model_pipeline.ensemble import EnsembleModel, Model
import textwrap
from sklearn.metrics import r2_score


logger = logging.getLogger(__name__)
color_pal = sns.color_palette()
plt.style.use("ggplot")


def ordered_set(input_list):
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def get_transformed_columns(
    sklearn_pipeline: Pipeline, categorical_features: list[str]
) -> list[str]:
    encoder = (
        sklearn_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["one_hot_encoder"]
    )
    pca = (
        sklearn_pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["pca"]
    )
    n_pca_components = pca.n_components_
    pca_cols = np.array([f"pca_{i}" for i in range(n_pca_components)])

    encoded_cat_cols = encoder.get_feature_names_out(
        input_features=categorical_features
    )

    return encoded_cat_cols.tolist() + pca_cols.tolist()


def plot_feature_importance(
    model: Model, X: np.ndarray, y: np.ndarray, columns: list[str]
) -> Figure:
    feature_importances = model.get_feature_importance(X, y)
    columns = [textwrap.fill(label.replace("_", " "), width=10) for label in columns]

    feature_importances = pd.DataFrame(
        data=feature_importances,
        index=columns,
        columns=["importance"],
    )
    feature_importances = feature_importances.sort_values(
        by="importance", ascending=False
    ).head(10)

    ax = feature_importances.sort_values("importance").plot(
        kind="barh", title=f"{model.id} Feature Importance"
    )
    return ax.get_figure()


def evaluate_feature_importance(
    sklearn_pipeline: Pipeline,
    categorical_features: list[str],
    ensemble_model: EnsembleModel,
    X: np.ndarray,
    y: np.ndarray,
    evaluation_set: str,
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
):
    transformed_columns = get_transformed_columns(
        sklearn_pipeline=sklearn_pipeline, categorical_features=categorical_features
    )

    feature_importance_plots = {
        f"{start_time}__{evaluation_set}_{model.id}_fi.png": plot_feature_importance(
            model=model,
            X=X,
            y=y,
            columns=transformed_columns,
        )
        for model in ensemble_model.models
    }
    return feature_importance_plots


def plot_residual_histogram(
    ax: Axes, col: str, errors: np.ndarray, color: str = None
) -> float:
    errors.hist(ax=ax, bins=np.arange(-3.5, 3.5, 0.1), color=color)
    mae = errors.abs().mean()
    ax.set_title(f"{col} MAE: {mae:.2f}")
    ax.set_xlabel(f"{col}_error")
    return mae


def plot_residual_scatter(
    ax: Axes, col: str, prediction: np.ndarray, target: np.ndarray, color: str = None
) -> float:
    has_prediction = ~prediction.isna()
    prediction = prediction[has_prediction]
    target = target[has_prediction]
    residual = target - prediction
    ax.scatter(prediction, residual, alpha=0.5, color=color)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel(f"{col}_predictions")
    ax.set_ylabel(f"residual")
    r2 = r2_score(target, prediction)
    ax.set_title(f"{col} R^2: {r2:.2f}")
    return r2


def evaluate_residuals(
    output_df: pd.DataFrame,
    eval_cols: list[str],
    target: str,
    evaluation_set: str,
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
) -> tuple[dict[str, float], dict[str, Figure]]:
    fig, axes = plt.subplots(
        nrows=2, ncols=len(eval_cols), figsize=(20, 10), sharey="row"
    )

    for i, col in enumerate(eval_cols):
        output_df[f"{col}_error"] = output_df[col] - output_df[target]
        mae = plot_residual_histogram(
            ax=axes[0, i], col=col, errors=output_df[f"{col}_error"], color=color_pal[i]
        )
        r2 = plot_residual_scatter(
            ax=axes[1, i],
            col=col,
            prediction=output_df[col],
            target=output_df[target],
            color=color_pal[i],
        )

    error_metrics = {
        f"{evaluation_set}_mae": mae,
        f"{evaluation_set}_r2": r2,
    }
    for metric, score in error_metrics.items():
        logger.info(f"{metric} = {score}")
    plt.subplots_adjust(wspace=0.1)
    error_plot = {f"{start_time}__{evaluation_set}_residuals.png": fig}
    return error_metrics, error_plot


def evaluate_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: EnsembleModel,
    sklearn_pipeline: Pipeline,
    experiment_id: int,
    start_time: str,
    parameters: dict[str, Any],
    evaluation_set: str,
) -> tuple[float, pd.DataFrame, dict[str, Figure], tuple[int, dict[str, float]]]:
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    baseline_columns = parameters["baseline_columns"]

    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_train_preprocessed = sklearn_pipeline.transform(X_train)
    X_test = test_data[numerical_features + categorical_features]
    X_test_preprocessed = sklearn_pipeline.transform(X_test)

    output_plots = {}

    output_plots.update(
        evaluate_feature_importance(
            sklearn_pipeline=sklearn_pipeline,
            categorical_features=categorical_features,
            ensemble_model=model,
            X=X_train_preprocessed.toarray(),
            y=y_train,
            evaluation_set=evaluation_set,
            start_time=start_time,
        )
    )

    test_predictions = model.predict(X=X_test_preprocessed)
    output_cols = ordered_set(
        ["id"] + numerical_features + categorical_features + [target] + baseline_columns
    )
    output_df = test_data[output_cols].copy()
    output_df["prediction"] = test_predictions

    output_metrics, error_plot = evaluate_residuals(
        output_df=output_df,
        eval_cols=["prediction"] + baseline_columns,
        target=target,
        evaluation_set=evaluation_set,
        start_time=start_time,
    )
    output_plots.update(error_plot)

    output_df["experiment_id"] = experiment_id
    output_df["start_time"] = start_time
    columns = ["experiment_id", "start_time"] + [
        col for col in output_df.columns if col not in ("experiment_id", "start_time")
    ]
    output_df = output_df[columns]

    return (
        output_df,
        output_plots,
        (experiment_id, output_metrics),
    )


evaluate_model_holdout = partial(evaluate_model, evaluation_set="holdout")


if __name__ == "__main__":
    import sqlite3
    import pandas as pd
    import yaml
    import pickle

    filename = "train_model_output.pkl"
    with open(filename, "rb") as file:
        model, pipeline = pickle.load(file)

    connection = sqlite3.connect("./data/fpl.db")
    holdout_data = pd.read_sql_query("SELECT * FROM holdout_data", connection)
    with open("./conf/base/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)
        parameters = parameters["model"]
