import logging
import multiprocessing as mp
import sqlite3
from typing import Any, Literal

import pandas as pd
from pandas.core.indexing import _LocIndexer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm

logger = logging.getLogger(__name__)


KNN_FEATURES = [
    "value",
    "home_total_att_elo",
    "away_total_att_elo",
    "total_def_elo",
    "home_total_def_elo",
    "away_total_def_elo",
]


def impute_with_knn(
    missing_data: pd.DataFrame,
    existing_data: pd.DataFrame,
    columns: list[str],
    column_type: Literal["categorical", "numerical"],
):
    if len(existing_data) == 0:
        return missing_data
    model = (
        KNeighborsClassifier if column_type == "categorical" else KNeighborsRegressor
    )
    predictor = model(n_neighbors=min(5, len(existing_data)))

    predictor.fit(
        existing_data[KNN_FEATURES],
        existing_data[columns],
    )
    imputed_values = predictor.predict(missing_data[KNN_FEATURES])
    imputed_values = pd.DataFrame(
        imputed_values, columns=columns, index=missing_data.index
    )
    missing_data.loc[:, columns] = missing_data.loc[:, columns].fillna(imputed_values)
    return missing_data


def impute_columns(
    data: pd.DataFrame,
    season_pos_round_mask: _LocIndexer,
    columns: list[str],
    column_type: Literal["categorical", "numerical"],
    season: int,
    pos: str,
    _round: int,
):
    missing = data[columns].isna().any(axis=1)
    data.loc[season_pos_round_mask & missing] = impute_with_knn(
        data.loc[season_pos_round_mask & missing],
        data.loc[season_pos_round_mask & ~missing],
        columns,
        column_type,
    )
    missing = data[columns].isna().any(axis=1)
    if (season_pos_round_mask & missing).sum() == 0:
        logger.info(f"Imputed {season} {pos} {_round} with same SEASON ROUND POS")
        return data

    other_rounds_filter = (data["season"] == season) & (data["round"] != _round)
    data.loc[season_pos_round_mask & missing] = impute_with_knn(
        data.loc[season_pos_round_mask & missing],
        data.loc[other_rounds_filter & ~missing],
        columns,
        column_type,
    )
    missing = data[columns].isna().any(axis=1)
    if (season_pos_round_mask & missing).sum() == 0:
        logger.info(
            f"Imputed {season} {pos} {_round} with same SEASON POS different ROUND"
        )
        return data

    other_seasons_filter = (data["season"] != season) & (data["pos"] == pos)
    data.loc[season_pos_round_mask & missing] = impute_with_knn(
        data.loc[season_pos_round_mask & missing],
        data.loc[other_seasons_filter & ~missing],
        columns,
        column_type,
    )
    missing = data[columns].isna().any(axis=1)
    if (season_pos_round_mask & missing).sum() == 0:
        logger.info(f"Imputed {season} {pos} {_round} with same POS different SEASON")
        return data

    na_cols = data[columns].isna().any()
    logger.warn(f"Could not impute {na_cols} for {season} {pos} {_round}")
    return data


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


def find_missing(data, columns):
    contains_na = data.groupby(["season", "round", "pos"], group_keys=True)[
        columns
    ].apply(lambda x: x.isna().any())
    contains_na = contains_na[contains_na.any(axis=1)]
    contains_na = contains_na.loc[contains_na.any(axis=1)]
    contains_na["missing_cols"] = contains_na.apply(
        lambda x: [col for col in x.index if x[col] == True], axis=1
    )
    contains_na = contains_na.drop(columns=columns)
    return contains_na


def parallel_impute_handler(
    data: pd.DataFrame,
    missing_data: pd.DataFrame,
    n_cores: int,
) -> pd.DataFrame:
    with mp.Pool(n_cores) as pool:
        for season, season_df in missing_data.groupby("season"):
            inputs = [
                (
                    data[(KNN_FEATURES + missing_cols + ["season", "round", "pos"])],
                    (
                        (data["season"] == season)
                        & (data["pos"] == pos)
                        & (data["round"] == _round)
                    ),
                    missing_cols,
                    col_type,
                    season,
                    pos,
                    _round,
                )
                for pos, _round, col_type, missing_cols in zip(
                    season_df["pos"],
                    season_df["round"],
                    season_df["level_0"],
                    season_df["missing_cols"],
                )
            ]
            values = pool.starmap(
                impute_columns,
                tqdm(inputs, desc=f"Imputing {season}", total=len(inputs)),
                chunksize=1,
            )
            for mask, val in zip(
                [i[1] for i in inputs],
                values,
            ):
                data.loc[mask, val.columns] = val
    return data


def sequential_impute_handler(
    data: pd.DataFrame,
    missing_data: pd.DataFrame,
    n_cores: int,
) -> pd.DataFrame:
    for season, season_df in missing_data.groupby("season"):
        for pos, _round, col_type, missing_cols in tqdm(
            zip(
                season_df["pos"],
                season_df["round"],
                season_df["level_0"],
                season_df["missing_cols"],
            ),
            desc=f"Imputing {season}",
            total=len(season_df),
        ):
            season_pos_round_mask = (
                (data["season"] == season)
                & (data["pos"] == pos)
                & (data["round"] == _round)
            )
            relevant_cols = ["season", "round", "pos"] + KNN_FEATURES + missing_cols
            data[relevant_cols] = impute_columns(
                data[relevant_cols],
                season_pos_round_mask,
                missing_cols,
                col_type,
                season,
                pos,
                _round,
            )
    return data


def impute_missing_values(
    data: pd.DataFrame, data_params: dict[str, Any], model_params: dict[str, Any]
) -> pd.DataFrame:
    n_cores = data_params["n_cores"]
    ma_lag = data_params["ma_lag"]
    excluded = KNN_FEATURES + ["pos", "round"]
    categorical_columns = [
        c for c in model_params["categorical_features"] if c not in excluded
    ]
    numerical_columns = [
        n for n in model_params["numerical_features"] if n not in excluded
    ]

    data = data.sort_values(["date", "fpl_name"])
    data[KNN_FEATURES] = data.groupby("fpl_name")[KNN_FEATURES].fillna(method="ffill")

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
    missing_categorical = find_missing(data, categorical_columns)
    missing_numerical = find_missing(data, numerical_columns)
    missing_data = (
        pd.concat(
            [missing_categorical, missing_numerical], keys=["categorical", "numerical"]
        )
        .reset_index()
        .sort_values(["season", "round", "pos"])
    )
    data = parallel_impute_handler(
        data,
        missing_data,
        n_cores=n_cores,
    )
    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
