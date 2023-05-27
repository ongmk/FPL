import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.pipeline import Pipeline
from typing import Any
from src.fpl.pipelines.model_pipeline.training import ensemble_predict

logger = logging.getLogger(__name__)
color_pal = sns.color_palette()
plt.style.use("ggplot")


def _ordered_set(input_list):
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


def evaluate_model(
    holdout_data: pd.DataFrame,
    models: dict[str, tuple[float, Any]],
    sklearn_pipeline: Pipeline,
    experiment_id: int,
    start_time: str,
    parameters: dict[str, Any],
) -> tuple[float, pd.DataFrame, dict[str, Figure], tuple[int, dict[str, float]]]:
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    baseline_columns = parameters["baseline_columns"]
    output_plots = {}
    transformed_columns = get_transformed_columns(
        sklearn_pipeline=sklearn_pipeline, categorical_features=categorical_features
    )

    for model_id, (_, model) in models.items():
        # Feature Importance
        features_importance = pd.DataFrame(
            data=model.feature_importances_,
            index=transformed_columns,
            columns=["importance"],
        )
        features_importance = features_importance.sort_values(
            by="importance", ascending=False
        ).head(10)

        ax = features_importance.sort_values("importance").plot(
            kind="barh", title="Feature Importance"
        )
        output_plots[f"{start_time}__{model_id}_fi.png"] = ax.get_figure()

    X_holdout = holdout_data[numerical_features + categorical_features]
    X_holdout_preprocessed = sklearn_pipeline.transform(X_holdout)
    holdout_predictions = ensemble_predict(models, X_holdout_preprocessed)
    output_cols = _ordered_set(
        ["id"] + numerical_features + categorical_features + [target] + baseline_columns
    )
    output_df = holdout_data[output_cols].copy()
    eval_cols = ["prediction"] + baseline_columns
    output_df["prediction"] = holdout_predictions

    fig, axes = plt.subplots(
        nrows=1, ncols=len(eval_cols), figsize=(20, 5), sharey=True
    )

    for i, col in enumerate(eval_cols):
        output_df[f"{col}_error"] = output_df[col] - output_df[target]
        output_df[f"{col}_error"].hist(
            ax=axes[i], bins=np.arange(-3.5, 3.5, 0.1), color=color_pal[i]
        )
        mae = output_df[f"{col}_error"].abs().mean()
        axes[i].set_title(f"{col} MAE: {mae:.2f}")
        axes[i].set_xlabel(f"{col}_error")
    output_df.head()
    output_metrics = {"mae": output_df["prediction_error"].abs().mean()}
    for metric, score in output_metrics.items():
        logger.info(f"{metric} = {score}")
    plt.subplots_adjust(wspace=0.1)
    output_plots[f"{start_time}__errors.png"] = fig

    output_df["experiment_id"] = experiment_id
    output_df["start_time"] = start_time
    columns = ["experiment_id", "start_time"] + [
        col for col in output_df.columns if col not in ("experiment_id", "start_time")
    ]
    output_df = output_df[columns]
    # plt.close("all")

    return (
        output_metrics["mae"],
        output_df,
        output_plots,
        (experiment_id, output_metrics),
    )


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
