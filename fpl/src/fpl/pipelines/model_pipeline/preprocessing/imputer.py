import sqlite3
from datetime import timedelta
from typing import Any

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm


def impute_with_knn(missing_data, existing_data, col, column_type):
    if len(missing_data) > 0 and len(existing_data) > 0:
        model = (
            KNeighborsClassifier
            if column_type == "categorical"
            else KNeighborsRegressor
        )
        predictor = model(n_neighbors=min(5, len(existing_data)))
        predictor.fit(
            existing_data[["value"]],
            existing_data[col],
        )
        return predictor.predict(missing_data[["value"]])
    else:
        return None


def impute_past_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    excluded = ["season", "date", "round", "fpl_name", "fpl_points", "value", "player"]
    numerical_columns = [
        col
        for col in data.select_dtypes(include=["int", "float"]).columns
        if col not in excluded
    ]
    categorical_columns = [
        col
        for col in data.select_dtypes(exclude=["int", "float"]).columns
        if col not in excluded
    ]

    for season in tqdm(data["season"].dropna().unique(), "Imputing Season"):
        season_filter = data["season"] == season
        for round in tqdm(
            data.loc[season_filter, "round"].dropna().unique(), "Round", leave=False
        ):
            season_round_filter = season_filter & (data["round"] == round)
            for col in categorical_columns:
                missing = data[col].isnull()
                data.loc[season_round_filter & missing, col] = impute_with_knn(
                    data.loc[season_round_filter & missing],
                    data.loc[season_round_filter & ~missing],
                    col,
                    "categorical",
                )
            for col in numerical_columns:
                missing = data[col].isnull()
                if (season_round_filter & missing).sum() > 0:
                    data.loc[season_round_filter & missing, col] = impute_with_knn(
                        data.loc[season_round_filter & missing],
                        data.loc[season_round_filter & ~missing],
                        col,
                        "numerical",
                    )
    return data


def impute_missing_values(
    data: pd.DataFrame, read_processed_data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    if parameters["use_cache"]:
        cached_data = read_processed_data.loc[read_processed_data["cached"] == True]
    data = data.sort_values(["date", "fpl_name"])
    data["value"] = data.groupby("fpl_name")["value"].fillna(method="ffill")

    player_log_max_date = data.loc[data["player"].notnull(), "date"].max()

    data.loc[data["date"] < player_log_max_date] = impute_past_data(
        data.loc[data["date"] < player_log_max_date]
    )

    # TODO: ffill data
    # TODO: check elo rating exist for next week's match
    data.loc[data["date"] < player_log_max_date - timedelta(days=3), "cached"] = True

    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
