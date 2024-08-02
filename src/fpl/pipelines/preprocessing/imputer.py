import logging
import multiprocessing as mp
import re
import sqlite3
import statistics
from typing import Any, Callable

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm

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
            existing_data[
                [
                    "value",
                    "home_total_att_elo",
                    "away_total_att_elo",
                    "total_def_elo",
                    "home_total_def_elo",
                    "away_total_def_elo",
                ]
            ],
            existing_data[col],
        )
        return predictor.predict(missing_data[["value"]])
    else:
        return None


def impute_with_all_seasons(
    data,
    season_round_pos_filter,
    categorical_columns,
    numerical_columns,
    season,
    pos,
    _round,
):
    for col in categorical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[season_round_pos_filter & ~missing],
                col,
                "categorical",
            )

    for col in numerical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() == 0:
            continue
        data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
            data.loc[season_round_pos_filter & missing],
            data.loc[season_round_pos_filter & ~missing],
            col,
            "numerical",
        )
    impute_with_other_rounds(
        data,
        season_round_pos_filter,
        categorical_columns,
        numerical_columns,
        season,
        _round,
    )
    impute_with_other_seasons(
        data,
        season_round_pos_filter,
        categorical_columns,
        numerical_columns,
        season,
        pos,
        _round,
    )
    return None


def impute_with_other_rounds(
    data,
    season_round_pos_filter,
    categorical_columns,
    numerical_columns,
    season,
    _round,
):
    other_rounds_filter = (data["season"] == season) & (data["round"] != _round)
    for col in categorical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[other_rounds_filter & missing],
                data.loc[other_rounds_filter & ~missing],
                col,
                "categorical",
            )

    for col in numerical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[other_rounds_filter & ~missing],
                col,
                "numerical",
            )
    return None


def impute_with_other_seasons(
    data,
    season_round_pos_filter,
    categorical_columns,
    numerical_columns,
    season,
    pos,
):
    other_seasons_filter = (data["season"] != season) & (data["pos"] == pos)
    for col in categorical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[other_seasons_filter & missing],
                data.loc[other_seasons_filter & ~missing],
                col,
                "categorical",
            )

    for col in numerical_columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[other_seasons_filter & ~missing],
                col,
                "numerical",
            )
    return None


GK_ONLY_COLS = [
    "sota",
    "ga",
    "saves",
    "savepct",
    "cs",
    "psxg",
]
OUTFIELD_ONLY_COLS = [
    "gls",
    "ast",
    "pk",
    "sh",
    "sot",
    "touches",
    "xg",
    "npxg",
    "xag",
    "sca",
    "gca",
]


def parallel_impute_handler(
    data: pd.DataFrame,
    impute_func: Callable,
    n_cores: int,
    ma_lag: int,
    excluded: list[str],
) -> pd.DataFrame:
    data = data.copy()
    excluded, numerical_columns, categorical_columns = split_columns(
        data, ma_lag, excluded
    )

    with mp.Pool(n_cores) as pool:
        total = len(data.groupby(["season", "round", "pos"]).size().reset_index())

        inputs = []
        for season in data["season"].dropna().unique():
            season_filter = data["season"] == season
            for _round in data.loc[season_filter, "round"].dropna().unique():
                season_round_filter = season_filter & (data["round"] == _round)
                for pos in data.loc[season_round_filter, "pos"].dropna().unique():
                    season_round_pos_filter = season_round_filter & (data["pos"] == pos)
                    inputs.append(
                        (
                            data,
                            season_round_pos_filter,
                            categorical_columns,
                            numerical_columns,
                            season,
                            pos,
                            _round,
                        )
                    )
        _ = pool.starmap(impute_func, tqdm(inputs, total=total))
    return data


def sequential_impute_handler(
    data: pd.DataFrame,
    impute_func: Callable,
    ma_lag: int,
    excluded: list[str],
) -> pd.DataFrame:
    data = data.copy()
    excluded, numerical_columns, categorical_columns = split_columns(
        data, ma_lag, excluded
    )

    for season in data["season"].dropna().unique():
        season_filter = data["season"] == season
        for _round in data.loc[season_filter, "round"].dropna().unique():
            season_round_filter = season_filter & (data["round"] == _round)
            for pos in data.loc[season_round_filter, "pos"].dropna().unique():
                season_round_pos_filter = season_round_filter & (data["pos"] == pos)
                impute_func(
                    data,
                    season_round_pos_filter,
                    categorical_columns,
                    numerical_columns,
                    season,
                    pos,
                    _round,
                )
    return data


def split_columns(
    data: pd.DataFrame, ma_lag: int, excluded: list[str]
) -> tuple[list[str], list[str], list[str]]:
    gk_only_plus_ma_cols = GK_ONLY_COLS + [f"{col}_ma{ma_lag}" for col in GK_ONLY_COLS]
    outfield_only_plus_ma_cols = OUTFIELD_ONLY_COLS + [
        f"{col}_ma{ma_lag}" for col in OUTFIELD_ONLY_COLS
    ]
    data.loc[data["pos"] == "GK", outfield_only_plus_ma_cols] = data.loc[
        data["pos"] == "GK", outfield_only_plus_ma_cols
    ].fillna(-1)
    data.loc[data["pos"] != "GK", gk_only_plus_ma_cols] = data.loc[
        data["pos"] != "GK", gk_only_plus_ma_cols
    ].fillna(-1)
    excluded = excluded + gk_only_plus_ma_cols + outfield_only_plus_ma_cols

    numerical_columns, categorical_columns = [], []
    for col in data.columns:
        if col in excluded:
            continue

        if data[col].dtype in [int, float]:
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)
    return excluded, numerical_columns, categorical_columns


def impute_missing_values(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    n_cores = parameters["n_cores"]
    excluded = parameters["do_not_impute"]
    ma_lag = parameters["ma_lag"]

    data = data.sort_values(["date", "fpl_name"])
    data["value"] = data.groupby("fpl_name")["value"].fillna(method="ffill")

    data = parallel_impute_handler(
        data,
        impute_with_all_seasons,
        n_cores=n_cores,
        ma_lag=ma_lag,
        excluded=excluded,
    )
    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
