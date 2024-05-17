import logging
import multiprocessing as mp
import re
import sqlite3
import statistics
from typing import Any

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
            existing_data[["value"]],
            existing_data[col],
        )
        return predictor.predict(missing_data[["value"]])
    else:
        return None


def impute_round(
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
        # If still missing, impute with other season's data
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            other_season_pos_filter = (data["season"] != season) & (data["pos"] == pos)
            data.loc[season_round_pos_filter & missing, col] = impute_with_knn(
                data.loc[season_round_pos_filter & missing],
                data.loc[other_season_pos_filter & ~missing],
                col,
                "numerical",
            )
        missing = data[col].isnull()
        if (season_round_pos_filter & missing).sum() > 0:
            logger.info(f"Cannot impute {col} for {season=}, {pos=}, {_round=}")
            pass
    logger.info(f"Imputed {season=}, {pos=}, {_round=}")
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


def impute_past_data(data: pd.DataFrame, n_cores, ma_lag, excluded=[]) -> pd.DataFrame:
    data = data.copy()

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
        _ = pool.starmap(impute_round, tqdm(inputs, total=total))
    return data


def ffill_future_data(data: pd.DataFrame, excluded=[]) -> pd.DataFrame:
    data = data.sort_values("date")
    ma_cols = data.filter(regex=r"\w+_ma\d+").columns.tolist()
    ma_features = set([re.sub(r"_ma\d+", "", col) for col in ma_cols])
    ma_lags = [int(re.findall(r"\d+", string)[0]) for string in ma_cols]
    lag = statistics.mode(ma_lags)
    for feature in ma_features:
        data[f"{feature}_ma{lag}"] = data[f"{feature}_ma{lag}"].ffill()

    columns = [col for col in data.columns if col not in excluded + ma_cols]
    for col in columns:
        data[col] = data[col].ffill()

    return data


def impute_missing_values(
    data: pd.DataFrame, parameters: dict[str, Any]
) -> pd.DataFrame:
    n_cores = parameters["n_cores"]
    excluded = parameters["do_not_impute"]
    ma_lag = parameters["ma_lag"]

    data = data.sort_values(["date", "fpl_name"])
    data["value"] = data.groupby("fpl_name")["value"].fillna(method="ffill")

    data.loc[data["cached"] == True] = impute_past_data(
        data.loc[data["cached"] == True],
        n_cores=n_cores,
        ma_lag=ma_lag,
        excluded=excluded,
    )

    data = ffill_future_data(data, excluded=excluded)

    return data


if __name__ == "__main__":
    conn = sqlite3.connect("./data/fpl.db")
    combined_data = pd.read_sql(f"select * from processed_data", conn)
    conn.close()
    result = impute_missing_values(combined_data)
# result.query("fpl_name == 'Aaron Lennon'")
