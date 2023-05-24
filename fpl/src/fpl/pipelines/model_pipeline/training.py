from typing import Any
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
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


def train_model(
    train_val_data: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[Any, Pipeline]:
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]

    X_train_val = train_val_data[numerical_features + categorical_features]
    y_train_val = train_val_data[target]
    groups = train_val_data[parameters["group_by"]]

    n_splits = groups.nunique()
    logger.info(f"{groups.unique() = }")
    group_kfold = GroupKFold(n_splits=n_splits)

    numerical_pipeline = Pipeline(
        [
            ("num_imputer", SimpleImputer(strategy="constant", fill_value=-999)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=parameters["pca_components"])),
        ]
    )

    categorical_data = X_train_val[categorical_features]
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
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )

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

        X_train_preprocessed = pipeline.fit_transform(X_train)
        X_val_preprocessed = pipeline.transform(X_train_val.iloc[val_index])

        y_val = y_train_val.iloc[val_index]

        model.fit(
            X_train_preprocessed,
            y_train,
            eval_set=[(X_train_preprocessed, y_train), (X_val_preprocessed, y_val)],
            verbose=100,
        )

        val_predictions = model.predict(X_val_preprocessed)
        val_accuracy = mean_squared_error(y_val, val_predictions)
        cross_val_scores.append(val_accuracy)

    avg_cv_accuracy = sum(cross_val_scores) / n_splits
    logger.info(f"Average cross-validation accuracy: {avg_cv_accuracy}")
    logger.info(cross_val_scores)
    return model, pipeline


if __name__ == "__main__":
    import sqlite3
    import pandas as pd
    import yaml

    # Connect to the SQLite database
    connection = sqlite3.connect("./data/fpl.db")
    train_val_data = pd.read_sql_query("SELECT * FROM train_val_data", connection)
    with open("./conf/base/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)
        parameters = parameters["model"]

    outputs = train_model(train_val_data=train_val_data, parameters=parameters)
    import pickle

    filename = "train_model_output.pkl"

    # Save the pipeline object to a file
    with open(filename, "wb") as file:
        pickle.dump(outputs, file)
