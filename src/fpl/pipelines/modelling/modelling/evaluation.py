import logging
import textwrap
from datetime import datetime
from functools import partial
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from fpl.pipelines.modelling.modelling.ensemble import EnsembleModel, Model

matplotlib.use("Agg")
logger = logging.getLogger(__name__)
color_pal = sns.color_palette()
plt.style.use("ggplot")


def get_transformed_columns(
    sklearn_pipeline: Pipeline,
    categorical_features: list[str],
    numerical_features: list[str],
) -> list[str]:
    encoder = (
        sklearn_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["one_hot_encoder"]
    )
    numerical_processor_steps = (
        sklearn_pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps
    )
    if "pca" in numerical_processor_steps:
        n_pca_components = numerical_processor_steps["pca"].n_components_
        numerical_features = np.array(
            [f"pca_{i}" for i in range(n_pca_components)]
        ).tolist()

    encoded_cat_cols = encoder.get_feature_names_out(
        input_features=categorical_features
    )

    return encoded_cat_cols.tolist() + numerical_features


def plot_feature_importance(
    model: Model, X: np.ndarray, y: np.ndarray, columns: list[str]
) -> Figure:
    feature_importances = model.get_feature_importance(X, y)
    columns = [textwrap.fill(label.replace("_", " "), width=10) for label in columns]

    feature_importances = pd.DataFrame(
        data=feature_importances.flatten(),
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
    numerical_features: list[str],
    ensemble_model: EnsembleModel,
    X: np.ndarray,
    y: np.ndarray,
    evaluation_set: str,
    start_time: str = datetime.now().strftime("%Y%m%d_%H%M%S"),
):
    transformed_columns = get_transformed_columns(
        sklearn_pipeline=sklearn_pipeline,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
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
) -> None:
    errors.hist(ax=ax, bins=np.arange(-3.5, 3.5, 0.1), color=color)
    mae = errors.abs().mean()
    ax.set_title(f"{col} MAE: {mae:.2f}")
    ax.set_xlabel(f"{col}_error")
    return None


def plot_residual_scatter(
    ax,
    col: str,
    target: np.ndarray,
    prediction: np.ndarray,
    color: str = None,
) -> None:
    r2 = calculate_r2(target, prediction)
    residual = target - prediction
    ax.scatter(prediction, residual, alpha=0.5, color=color)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel(f"{col}_predictions")
    ax.set_ylabel(f"residual")
    ax.set_title(f"{col} R^2: {r2:.2f}")
    return None


def calculate_r2(target, prediction):
    not_nan = ~prediction.isna() & ~target.isna()
    prediction = prediction[not_nan]
    target = target[not_nan]
    return r2_score(target, prediction)


def calculate_mae(target, prediction):
    not_nan = ~prediction.isna() & ~target.isna()
    prediction = prediction[not_nan]
    target = target[not_nan]
    return (prediction - target).abs().mean()


metrics_dict = {"r2": calculate_r2, "mae": calculate_mae}


def evaluate_residuals(
    inference_results: pd.DataFrame,
    prediction_col: str,
    target: str,
    metrics: list[str],
    baseline_cols: list[str],
    evaluation_set: str,
    start_time: str = datetime.now().strftime("%Y%m%d_%H%M%S"),
) -> tuple[dict[str, float], dict[str, Figure]]:
    eval_cols = [prediction_col] + baseline_cols
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(eval_cols),
        figsize=(20, 13),
        sharey="row",
        gridspec_kw={"hspace": 0.4},
    )

    for i, col in enumerate(eval_cols):
        inference_results[f"{col}_error"] = (
            inference_results[col] - inference_results[target]
        )
        plot_residual_histogram(
            ax=axes[0, i],
            col=col,
            errors=inference_results[f"{col}_error"],
            color=color_pal[i],
        )
        plot_residual_scatter(
            ax=axes[1, i],
            col=col,
            target=inference_results[target],
            prediction=inference_results[col],
            color=color_pal[i],
        )

    error_metrics = {
        f"{evaluation_set}_{m}": metrics_dict[m](
            inference_results[target], inference_results[prediction_col]
        )
        for m in metrics
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
    metrics = parameters["metrics"]

    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_train_preprocessed = sklearn_pipeline.transform(X_train)
    X_test = test_data[numerical_features + categorical_features]
    X_test.columns = X_test.columns.astype(str)
    X_test_preprocessed = sklearn_pipeline.transform(X_test)

    output_plots = {}

    output_plots.update(
        evaluate_feature_importance(
            sklearn_pipeline=sklearn_pipeline,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            ensemble_model=model,
            X=X_train_preprocessed,
            y=y_train,
            evaluation_set=evaluation_set,
            start_time=start_time,
        )
    )

    test_predictions = model.predict(X=X_test_preprocessed)
    inference_results = pd.DataFrame(index=test_data.index)
    inference_results["experiment_id"] = experiment_id
    inference_results["start_time"] = start_time
    inference_results = test_data.join(inference_results)
    inference_results["prediction"] = test_predictions

    output_metrics, error_plot = evaluate_residuals(
        inference_results=inference_results,
        prediction_col="prediction",
        target=target,
        metrics=metrics,
        baseline_cols=baseline_columns,
        evaluation_set=evaluation_set,
        start_time=start_time,
    )
    output_plots.update(error_plot)

    return (
        inference_results,
        output_plots,
        (experiment_id, output_metrics),
    )


evaluate_model_holdout = partial(evaluate_model, evaluation_set="holdout")


if __name__ == "__main__":
    import pickle
    import sqlite3

    import pandas as pd
    import yaml

    filename = "train_model_output.pkl"
    with open(filename, "rb") as file:
        model, pipeline = pickle.load(file)

    connection = sqlite3.connect("./data/fpl.db")
    holdout_data = pd.read_sql_query("SELECT * FROM holdout_data", connection)
    with open("./conf/base/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)
        parameters = parameters["model"]
