import inspect
from src.fpl.pipelines.model_pipeline.all_models.regression import (
    get_regression_model_instance,
)
import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)


def has_fit_parameter(cls, param_name):
    fit_method = getattr(cls, "fit", None)
    method_signature = inspect.signature(fit_method)
    return param_name in method_signature.parameters


@dataclass
class Model:
    id: int
    model: Any
    weight: float

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        early_stopping_rounds: int = None,
        verbose: bool = True,
    ) -> None:
        fit_params = {}
        if has_fit_parameter(self.model, "early_stopping_rounds"):
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        if has_fit_parameter(self.model, "eval_set"):
            if X_val is None:
                fit_params["eval_set"] = [(X, y)]
            else:
                fit_params["eval_set"] = [(X_val, y_val)]
        if has_fit_parameter(self.model, "verbose"):
            fit_params["verbose"] = verbose
        self.model.fit(X, y, **fit_params)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)
        elif hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        else:
            result = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=0
            )
            return result.importances_mean


class EnsembleModel:
    def __init__(self, models: list[Model]):
        self.models = models
        self.is_ensemble = len(models) > 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        early_stopping_rounds: int = None,
        verbose: bool = None,
    ):
        if self.is_ensemble:
            self.ensemble_fit(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        else:
            self.models[0].fit(
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )

    def predict(self, X):
        if self.is_ensemble:
            return self.ensemble_predict(X)
        else:
            return self.models[0].predict(X)

    def ensemble_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:
        for model in self.models:
            model.fit(X, y, **kwargs)
        return None

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        weights = []
        for model in self.models:
            predictions.append(model.predict(X))
            weights.append(model.weight)

        ensembled_predictions = np.average(predictions, axis=0, weights=weights)
        return ensembled_predictions


def model_selection(parameters: dict[str, Any]) -> EnsembleModel:
    models = []
    model_ids = parameters["models"]
    model_weights = parameters["model_weights"]
    if len(model_weights) > len(model_ids):
        logger.warning(
            "Number of model weights > number of models. All model weights will be set to 1."
        )
        model_weights = len(model_ids) * [1]

    for model_id, model_weight in zip(model_ids, model_weights):
        model_params = parameters[f"{model_id}_params"]
        model = get_regression_model_instance(
            model_id=model_id, model_params=model_params
        )
        models.append(Model(id=model_id, model=model, weight=model_weight))

    return EnsembleModel(models=models)
