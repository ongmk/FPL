import logging
from typing import Any

import numpy as np
import pandas as pd
from pycaret.regression import compare_models, pull, setup
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.fpl.pipelines.model_pipeline.modelling.ensemble import EnsembleModel
from src.fpl.pipelines.model_pipeline.modelling.evaluation import evaluate_model

logger = logging.getLogger(__name__)


def get_all_numerical_features(parameters: dict[str, Any]) -> list[str]:
    numerical_features = parameters["numerical_features"]
    ma_lag = parameters["ma_lag"]
    ma_features = parameters["ma_features"]
    for i in range(ma_lag):
        numerical_features += [f"{f}_ma{i+1}" for f in ma_features]
    return numerical_features


def filter_train_val_data(
    train_val_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    group_by = parameters["group_by"]
    target = parameters["target"]
    categorical_features = parameters["categorical_features"]
    numerical_features = get_all_numerical_features(parameters)

    train_val_data = train_val_data[
        [group_by] + categorical_features + numerical_features + [target]
    ]
    train_val_data = train_val_data.dropna(subset=numerical_features)
    return train_val_data


def create_sklearn_pipeline(
    train_val_data: pd.DataFrame, parameters: dict[str, Any]
) -> Pipeline:
    categorical_features = parameters["categorical_features"]
    numerical_features = get_all_numerical_features(parameters)

    numerical_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=parameters["pca_components"])),
        ]
    )

    categorical_data = train_val_data[categorical_features]
    categories = [
        np.append(categorical_data[col].unique(), "Unknown")
        for col in categorical_data.columns
    ]
    categorical_pipeline = Pipeline(
        [
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    categories=categories,
                    min_frequency=1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    sklearn_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    return sklearn_pipeline


def pycaret_compare_models(
    train_val_data: pd.DataFrame, sklearn_pipeline: Pipeline, parameters: dict[str, Any]
) -> pd.DataFrame:
    target = parameters["target"]
    groups = train_val_data[parameters["group_by"]]
    n_splits = groups.nunique()
    logger.info(f"GroupKFold splits = {n_splits}")

    setup(
        data=train_val_data,
        target=target,
        preprocess=False,
        custom_pipeline=sklearn_pipeline,
        fold_strategy="groupkfold",
        fold=n_splits,
        fold_groups=parameters["group_by"],
    )

    compare_models(sort=parameters["sort_models"])
    pycaret_result = pull()
    return pycaret_result


def get_mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    mean_dict = {}

    for key in metrics_list[0].keys():
        scores = [metric_dict[key] for metric_dict in metrics_list]
        avg_score = np.mean(scores)
        mean_dict[key] = avg_score
        logger.info(f"Cross validation {key} = {scores}")
        logger.info(f"Average score = {avg_score}")

    return mean_dict


def cross_validation(
    train_val_data: pd.DataFrame,
    model: EnsembleModel,
    sklearn_pipeline: Pipeline,
    experiment_id: int,
    start_time: str,
    parameters: dict[str, Any],
) -> tuple[float, tuple[int, dict[str, float]]]:
    groups = train_val_data[parameters["group_by"]]
    group_kfold = GroupKFold(n_splits=groups.nunique())

    all_folds_metrics = []
    for train_index, val_index in group_kfold.split(X=train_val_data, groups=groups):
        train_data = train_val_data.loc[train_index]
        val_data = train_val_data.loc[val_index]
        model, sklearn_pipeline = train_model(
            train_data=train_data,
            model=model,
            sklearn_pipeline=sklearn_pipeline,
            parameters=parameters,
        )
        _, last_fold_evaluation_plots, (_, fold_metrics) = evaluate_model(
            train_data=train_data,
            test_data=val_data,
            model=model,
            sklearn_pipeline=sklearn_pipeline,
            experiment_id=experiment_id,
            start_time=start_time,
            parameters=parameters,
            evaluation_set="val",
        )
        all_folds_metrics.append(fold_metrics)

    mean_metrics = get_mean_metrics(all_folds_metrics)

    val_score = mean_metrics["val_r2"]

    return (val_score, (experiment_id, mean_metrics), last_fold_evaluation_plots)


def train_model(
    train_data: pd.DataFrame,
    model: EnsembleModel,
    sklearn_pipeline: Pipeline,
    parameters: dict[str, str],
) -> tuple[EnsembleModel, Pipeline]:
    logger.info(f"Training model: {model}")
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    X_train = train_data[numerical_features + categorical_features]
    y_train = train_data[target]
    X_train_preprocessed = sklearn_pipeline.fit_transform(X_train)
    model.fit(
        X=X_train_preprocessed,
        y=y_train,
    )
    return model, sklearn_pipeline
