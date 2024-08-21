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
    col: str,
    column_type: Literal["categorical", "numerical"],
):
    if len(missing_data) > 0 and len(existing_data) > 0:
        model = (
            KNeighborsClassifier
            if column_type == "categorical"
            else KNeighborsRegressor
        )
        predictor = model(n_neighbors=min(5, len(existing_data)))

        predictor.fit(
            existing_data[KNN_FEATURES],
            existing_data[col],
        )
        return predictor.predict(missing_data[KNN_FEATURES])
    else:
        return None


def impute_columns(
    data: pd.DataFrame,
    season_round_pos_filter: _LocIndexer,
    columns: list[str],
    column_type: Literal["categorical", "numerical"],
    season: int,
    pos: str,
    _round: int,
):
    for col in columns:
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() == 0:
            continue
        data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
            data.loc[season_round_pos_filter & missing],
            data.loc[season_round_pos_filter & ~missing],
            col,
            column_type,
        )
        if (season_round_pos_filter & missing).sum() == 0:
            logger.info(
                f"Imputed {col} for {season} {pos} {_round} with same SEASON ROUND POS"
            )
            continue

        other_rounds_filter = (data["season"] == season) & (data["round"] != _round)
        missing = data[col].isnull()
        data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
            data.loc[season_round_pos_filter & missing],
            data.loc[other_rounds_filter & ~missing],
            col,
            column_type,
        )
        if (season_round_pos_filter & missing).sum() == 0:
            logger.info(
                f"Imputed {col} for {season} {pos} {_round} with same SEASON POS different ROUND"
            )
            continue

        other_seasons_filter = (data["season"] != season) & (data["pos"] == pos)
        missing = data[col].isnull()
        data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
            data.loc[season_round_pos_filter & missing],
            data.loc[other_seasons_filter & ~missing],
            col,
            column_type,
        )
        if (season_round_pos_filter & missing).sum() == 0:
            logger.info(
                f"Imputed {col} for {season} {pos} {_round} with same POS different SEASON"
            )
            continue
        else:
            logger.warn(f"Could not impute {col} for {season} {pos} {_round}")

    return data.loc[season_round_pos_filter]


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
    n_cores: int,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    data = data.copy()

    with mp.Pool(n_cores) as pool:
        for season in data["season"].dropna().unique():
            season_filter = data["season"] == season
            total = len(
                data.loc[season_filter]
                .groupby(["season", "round", "pos"])
                .size()
                .reset_index()
            )
            inputs = []
            for _round in data.loc[season_filter, "round"].dropna().unique():
                season_round_filter = season_filter & (data["round"] == _round)
                for pos in data.loc[season_round_filter, "pos"].dropna().unique():
                    season_round_pos_filter = season_round_filter & (data["pos"] == pos)
                    inputs.append(
                        (
                            data,
                            season_round_pos_filter,
                            categorical_columns,
                            "categorical",
                            season,
                            pos,
                            _round,
                        )
                    )
                    inputs.append(
                        (
                            data,
                            season_round_pos_filter,
                            numerical_columns,
                            "numerical",
                            season,
                            pos,
                            _round,
                        )
                    )
            values = pool.starmap(
                impute_columns,
                tqdm(inputs, total=total * 2, desc=f"Imputing {season}"),
                chunksize=1,
            )
            for f, v in tqdm(
                zip([i[1] for i in inputs], values),
                total=total * 2,
                desc=f"Mapping {season}",
            ):
                data.loc[f] = v
    return data


def sequential_impute_handler(
    data: pd.DataFrame,
    n_cores: int,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    data = data.copy()

    for season in data["season"].dropna().unique()[-1:]:
        season_filter = data["season"] == season
        for _round in data.loc[season_filter, "round"].dropna().unique()[:10]:
            season_round_filter = season_filter & (data["round"] == _round)
            for pos in data.loc[season_round_filter, "pos"].dropna().unique():
                season_round_pos_filter = season_round_filter & (data["pos"] == pos)
                impute_columns(
                    data,
                    season_round_pos_filter,
                    categorical_columns,
                    "categorical",
                    season,
                    pos,
                    _round,
                )
                impute_columns(
                    data,
                    season_round_pos_filter,
                    numerical_columns,
                    "numerical",
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
    categorical_features = model_params["categorical_features"]
    numerical_features = model_params["numerical_features"]
    excluded = KNN_FEATURES + ["pos"]

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

    data = parallel_impute_handler(
        data,
        n_cores=n_cores,
        categorical_columns=[c for c in categorical_features if c not in excluded],
        numerical_columns=[n for n in numerical_features if n not in excluded],
    )
    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
