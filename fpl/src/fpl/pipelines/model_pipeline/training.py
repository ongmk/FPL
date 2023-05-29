from typing import Any
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from pycaret.regression import setup, compare_models, pull
import inspect
from src.fpl.pipelines.model_pipeline.all_models.regression import (
    get_regression_model_instance,
)
import logging

logger = logging.getLogger(__name__)


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]
    train_val_data = processed_data[processed_data["season"] != holdout_year]
    holdout_data = processed_data[processed_data["season"] == holdout_year]

    return train_val_data, holdout_data


def create_sklearn_pipeline(
    train_val_data: pd.DataFrame, parameters: dict[str, Any]
) -> Pipeline:
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    numerical_pipeline = Pipeline(
        [
            ("num_imputer", SimpleImputer(strategy="constant", fill_value=-999)),
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
            ("cat_imputer", SimpleImputer(strategy="constant", fill_value="missing")),
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


def has_fit_parameter(cls, param_name):
    fit_method = getattr(cls, "fit", None)
    method_signature = inspect.signature(fit_method)
    return param_name in method_signature.parameters


def ensemble_fit(
    models: list[Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict[str, Any],
) -> None:
    for model in models:
        fit_params = {}
        if has_fit_parameter(model, "early_stopping_rounds"):
            fit_params["early_stopping_rounds"] = parameters["early_stopping_rounds"]
        if has_fit_parameter(model, "eval_set"):
            fit_params["eval_set"] = [(X_val, y_val)]
        if has_fit_parameter(model, "verbose"):
            fit_params["verbose"] = parameters["verbose"]

        model.fit(X_train, y_train, **fit_params)
    return None


def ensemble_predict(
    models: list[Any],
    weights: list[float],
    X: pd.DataFrame,
) -> np.ndarray:
    model_predictions = []
    for model in models:
        model_predictions.append(model.predict(X))
    ensemble_predictions = np.average(model_predictions, axis=0, weights=weights)
    return ensemble_predictions


def model_selection(parameters: dict[str, Any]) -> list[Any]:
    models = []
    for model_id in parameters["models"]:
        model_params = parameters[f"{model_id}_params"]
        model = get_regression_model_instance(
            model_id=model_id, model_params=model_params
        )
        models.append(model)
    return models


def cross_validation(
    train_val_data: pd.DataFrame,
    models: list[Any],
    sklearn_pipeline: Pipeline,
    experiment_id: int,
    parameters: dict[str, Any],
) -> tuple[float, tuple[int, dict[str, float]]]:
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    model_weights = parameters["model_weights"]

    X_train_val = train_val_data[numerical_features + categorical_features]
    y_train_val = train_val_data[target]
    groups = train_val_data[parameters["group_by"]]
    group_kfold = GroupKFold(n_splits=groups.nunique())

    scores = []
    for train_index, val_index in group_kfold.split(X_train_val, y_train_val, groups):
        X_train, X_val = X_train_val.loc[train_index], X_train_val.loc[val_index]
        X_train_preprocessed = sklearn_pipeline.fit_transform(X_train)
        X_val_preprocessed = sklearn_pipeline.transform(X_val)
        y_train, y_val = y_train_val.loc[train_index], y_train_val.loc[val_index]

        ensemble_fit(
            models=models,
            X_train=X_train_preprocessed,
            y_train=y_train,
            X_val=X_val_preprocessed,
            y_val=y_val,
            parameters=parameters,
        )
        y_pred = ensemble_predict(
            models=models,
            weights=model_weights,
            X=X_val_preprocessed,
        )

        score = r2_score(y_val, y_pred)
        scores.append(score)
    logger.info(f"Cross validation scores = {scores}")
    avg_score = np.mean(scores)
    logger.info(f"Average score = {avg_score}")
    output_metrics = {"val_r2": avg_score}
    return avg_score, (experiment_id, output_metrics)


def train_model(
    train_val_data: pd.DataFrame,
    models: list[Any],
    sklearn_pipeline: Pipeline,
    parameters: dict[str, str],
) -> dict[str, Any]:
    logger.info(f"Training models: {models}")
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    X_train_val = train_val_data[numerical_features + categorical_features]
    y_train_val = train_val_data[target]
    X_train_val_preprocessed = sklearn_pipeline.fit_transform(X_train_val)
    ensemble_fit(
        models=models,
        X_train=X_train_val_preprocessed,
        y_train=y_train_val,
        X_val=X_train_val_preprocessed,
        y_val=y_train_val,
        parameters=parameters,
    )
    return models, sklearn_pipeline
