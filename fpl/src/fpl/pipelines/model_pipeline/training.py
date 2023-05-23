from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]
    train_val_data = processed_data[processed_data["season"] != holdout_year]
    holdout_data = processed_data[processed_data["season"] == holdout_year]

    return train_val_data, holdout_data


def _encode_features(
    X: np.ndarray,
    categorical_features: list[str],
    numerical_features: list[str],
    encoder: OneHotEncoder,
) -> np.ndarray:
    X_cat = X[categorical_features]
    X_num = X[numerical_features]
    X_encoded = np.hstack([encoder.transform(X_cat).toarray(), X_num])
    return X_encoded


def train_model(train_val_data, parameters):
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]

    X_train_val = train_val_data[numerical_features + categorical_features]
    y_train_val = train_val_data[target]
    groups = train_val_data[parameters["group_by"]]

    n_splits = groups.nunique()
    logger.info(f"{groups.unique() = }")
    group_kfold = GroupKFold(n_splits=n_splits)

    X_train_val_cat = X_train_val[categorical_features]
    categories = [
        np.append(X_train_val_cat[col].unique(), "Unknown")
        for col in X_train_val_cat.columns
    ]
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist", categories=categories, min_frequency=1
    )
    encoder.fit(X_train_val_cat)

    xgb_parameters = parameters["xgboost_params"]
    xgb_parameters["n_estimators"] = int(xgb_parameters["n_estimators"])
    model = XGBRegressor(
        random_state=parameters["random_seed"],
        **xgb_parameters,
    )

    cross_val_scores = []
    for train_index, val_index in group_kfold.split(X_train_val, y_train_val, groups):
        # Remove outliers
        X_train, y_train = X_train_val.iloc[train_index], y_train_val.iloc[train_index]
        outlier = y_train > 4
        X_train, y_train = X_train.loc[~outlier], y_train.loc[~outlier]

        X_train_encoded = _encode_features(
            X_train, categorical_features, numerical_features, encoder
        )
        X_val_encoded = _encode_features(
            X_train_val.iloc[val_index],
            categorical_features,
            numerical_features,
            encoder,
        )
        y_val = y_train_val.iloc[val_index]

        model.fit(
            X_train_encoded,
            y_train,
            eval_set=[(X_train_encoded, y_train), (X_val_encoded, y_val)],
            verbose=100,
        )

        val_predictions = model.predict(X_val_encoded)
        val_accuracy = mean_squared_error(y_val, val_predictions)
        cross_val_scores.append(val_accuracy)

    avg_cv_accuracy = sum(cross_val_scores) / n_splits
    logger.info(f"Average cross-validation accuracy: {avg_cv_accuracy}")
    logger.info(cross_val_scores)
    return model, encoder
