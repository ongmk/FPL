import re
import sqlite3
import statistics
from typing import Any

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm
import multiprocessing as mp
import logging
logger = logging.getLogger(__name__)


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
    

def impute_round(data, season_round_pos_filter, categorical_columns, numerical_columns, season, pos, _round):
    for col in categorical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[
                season_round_pos_filter & missing, col
            ] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[season_round_pos_filter & ~missing],
                col,
                "categorical",
            )
    for col in numerical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[
                season_round_pos_filter & missing, col
            ] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[season_round_pos_filter & ~missing],
                col,
                "numerical",
            )
        # If still missing, impute with other season's data
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            other_season_pos_filter = (data["season"] != season) & (
                data["pos"] == pos
            )
            data.loc[
                season_round_pos_filter & missing, col
            ] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[other_season_pos_filter & ~missing],
                col,
                "numerical",
            )
    return logger.info(f"Imputed {season=}, {pos=}, {_round=}")


def impute_past_data(data: pd.DataFrame, n_cores) -> pd.DataFrame:
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

    with mp.Pool(n_cores) as pool:
        total = len(data.groupby(['season', 'round', 'pos']).size().reset_index())
            
        inputs = []
        for season in data["season"].dropna().unique():
            season_filter = data["season"] == season
            for _round in data.loc[season_filter, "round"].dropna().unique():
                season_round_filter = season_filter & (data["round"] == _round)
                for pos in data.loc[season_round_filter, "pos"].dropna().unique():
                    season_round_pos_filter = season_round_filter & (data["pos"] == pos)
                    inputs.append((data, season_round_pos_filter, categorical_columns, numerical_columns, season, pos, _round))
        _ = pool.starmap(impute_round, tqdm(inputs, total=total))
    return data


def ffill_future_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("date")
    ma_cols = data.filter(regex=r"\w+_ma\d+").columns.tolist()
    ma_features = set([re.sub(r"_ma\d+", "", col) for col in ma_cols])
    ma_lags = [int(re.findall(r"\d+", string)[0]) for string in ma_cols]
    lag = statistics.mode(ma_lags)
    for feature in ma_features:
        data[f"{feature}_ma{lag}"] = data[f"{feature}_ma{lag}"].ffill()

    excluded = [
        "season",
        "date",
        "round",
        "fpl_name",
        "fpl_points",
        "value",
        "player",
        "cached",
    ]
    columns = [col for col in data.columns if col not in excluded + ma_cols]
    for col in columns:
        data[col] = data[col].ffill()

    return data


def impute_missing_values(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    n_cores = parameters["n_cores"]
    
    data = data.sort_values(["date", "fpl_name"])
    data["value"] = data.groupby("fpl_name")["value"].fillna(method="ffill")

    data.loc[data["cached"] == True] = impute_past_data(
        data.loc[data["cached"] == True],
        n_cores=n_cores
    )

    data = ffill_future_data(data)
    # TODO: check elo rating exist for next week's match

    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
