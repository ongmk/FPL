import itertools
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm

logger = logging.getLogger(__name__)

HOME_ELO_COLS = [
    "att_elo",
    "home_att_elo",
    "away_att_elo",
    "def_elo",
    "home_def_elo",
    "away_def_elo",
]
AWAY_ELO_COLS = [
    "att_elo_opp",
    "home_att_elo_opp",
    "away_att_elo_opp",
    "def_elo_opp",
    "home_def_elo_opp",
    "away_def_elo_opp",
]


def agg_home_away_elo(data: pd.DataFrame) -> pd.DataFrame:
    data["total_att_elo"] = data.att_elo + data.def_elo_opp
    data["home_total_att_elo"] = data.home_att_elo + data.away_def_elo_opp
    data["away_total_att_elo"] = data.away_att_elo + data.home_def_elo_opp
    data["total_def_elo"] = data.def_elo + data.att_elo_opp
    data["home_total_def_elo"] = data.home_def_elo + data.away_att_elo_opp
    data["away_total_def_elo"] = data.away_def_elo + data.home_att_elo_opp
    data = data.drop(columns=HOME_ELO_COLS)
    data = data.drop(columns=AWAY_ELO_COLS)
    return data


def calculate_points(row):
    if pd.isna(row["team_gf"]) and pd.isna(row["team_ga"]):
        return None
    if row["team_gf"] > row["team_ga"]:
        return 3
    elif row["team_gf"] == row["team_ga"]:
        return 1
    else:
        return 0


def calculate_daily_rank(date_data: pd.DataFrame) -> pd.Series:
    date_data = date_data.sort_values(["pts_b4_match", "team"], ascending=[False, True])
    date_data["rank_b4_match"] = date_data.reset_index().index + 1
    return date_data


def features_from_pts(points_data: pd.DataFrame) -> pd.DataFrame:
    output_data = pd.DataFrame(
        columns=[
            "season",
            "team",
            "date",
            "pts_b4_match",
            "rank_b4_match",
            "pts_gap_above",
            "pts_gap_below",
        ]
    )
    for season, season_data in points_data.groupby("season"):
        team_date = pd.DataFrame(
            list(
                itertools.product(
                    [season], season_data["team"].unique(), season_data["date"].unique()
                )
            ),
            columns=["season", "team", "date"],
        )
        season_data["matchday"] = True
        fill_dates = team_date.merge(
            season_data, on=["season", "team", "date"], how="left"
        )

        fill_dates = fill_dates.sort_values("date")
        fill_dates["pts_b4_match"] = (
            fill_dates.groupby("team")["pts_b4_match"].ffill().fillna(0)
        )
        fill_dates = (
            fill_dates.groupby("date")
            .apply(calculate_daily_rank)
            .reset_index(drop=True)
        )

        fill_dates = fill_dates.sort_values(["date", "pts_b4_match"], ascending=False)
        fill_dates["pts_gap_above"] = (
            fill_dates.groupby("date")["pts_b4_match"].shift(1)
            - fill_dates["pts_b4_match"]
        )
        fill_dates["pts_gap_below"] = fill_dates["pts_b4_match"] - fill_dates.groupby(
            "date"
        )["pts_b4_match"].shift(-1)
        fill_dates["pts_gap_above"] = fill_dates.groupby("date")[
            "pts_gap_above"
        ].transform(lambda x: x.fillna(x.mean()))
        fill_dates["pts_gap_below"] = fill_dates.groupby("date")[
            "pts_gap_below"
        ].transform(lambda x: x.fillna(x.mean()))
        fill_dates = fill_dates.loc[fill_dates["matchday"] == True].drop(
            "matchday", axis=1
        )
        output_data = pd.concat([output_data, fill_dates])

    output_data = output_data.reset_index(drop=True)
    return output_data


def calculate_pts_data(data: pd.DataFrame, cached_data: pd.DataFrame) -> pd.DataFrame:
    pts_data = (
        data.copy()
        .dropna(subset=["season", "team", "date"])
        .drop_duplicates(["season", "team", "date"])
    )

    pts_data = pts_data[["season", "team", "date", "team_gf", "team_ga"]]
    if cached_data is not None:
        cached_pts_data = cached_data[
            ["season", "team", "date", "team_gf", "team_ga"]
        ].drop_duplicates(subset=["season", "team", "date"])
        pts_data = pd.concat([cached_pts_data, pts_data], ignore_index=True)

    tqdm.pandas(desc="Calculating league points")
    pts_data["match_points"] = pts_data.progress_apply(calculate_points, axis=1)
    pts_data = pts_data.drop(columns=["team_gf", "team_ga"])
    pts_data["league_points"] = (
        pts_data.groupby(["season", "team"])["match_points"]
    ).shift(1)
    pts_data["pts_b4_match"] = (
        pts_data.groupby(["season", "team"])["league_points"].cumsum().fillna(0)
    )

    pts_data = features_from_pts(pts_data)

    return data.merge(pts_data, on=["season", "team", "date"])


def create_lag_features(df: pd.DataFrame, match_stat_col: str, lag: int, drop=True):
    df = df.sort_values(by=["fpl_name", "date"])

    # Create lag features
    for i in range(1, lag + 1):
        shifted = df.groupby("fpl_name")[match_stat_col].shift(i)
        date_diff = df["date"] - df.groupby("fpl_name")["date"].shift(i)
        within_one_year = date_diff <= timedelta(days=365)
        df[match_stat_col + "_" + str(i)] = shifted.where(within_one_year, None)

    # Drop the original column
    if drop:
        df = df.drop(columns=[match_stat_col])
    return df


def single_ma_feature(
    data: pd.DataFrame, cached_data: pd.DataFrame, match_stat_col: str, lag: int
):
    # ma_cols = [f"{match_stat_col}_ma{i+1}" for i in range(lag)]
    ma_cols = [f"{match_stat_col}_ma{lag}"]
    if cached_data is not None:
        columns = ["fpl_name", "date", "cached"] + ma_cols
        cached_ma_data = cached_data[columns]
        ma_data = pd.concat([cached_ma_data, data], ignore_index=True)
    else:
        ma_data = data
        ma_data["cached"] = None
        cached_ma_data = pd.DataFrame(columns=ma_cols)

    ma_data = ma_data.sort_values(by=["date"])

    tqdm.pandas(leave=False, desc="Calculating row")
    ma_data = (
        ma_data.groupby("fpl_name")
        .parallel_apply(
            lambda player_df: calculate_single_lag_ma(player_df, match_stat_col, lag)
        )
        .droplevel(0)
    )
    ma_data = pd.concat([cached_ma_data[ma_cols], ma_data])
    data = data.drop(columns=ma_cols, errors="ignore")
    return pd.merge(data, ma_data, left_index=True, right_index=True)


def calculate_multi_lag_ma(group: DataFrameGroupBy, match_stat_col: str, max_lag: int):
    ma_df = pd.DataFrame(index=group[group["cached"] != True].index)

    for index in ma_df.index:
        i = group.index.get_loc(index)
        for lag in range(1, max_lag + 1):
            if i >= lag and group["date"].iloc[i] - group["date"].iloc[
                i - lag
            ] <= timedelta(days=365):
                ma_df.loc[group.index[i], f"{match_stat_col}_ma{lag}"] = (
                    group[match_stat_col].iloc[i - lag : i].mean()
                )

    return ma_df


def calculate_single_lag_ma(group: DataFrameGroupBy, match_stat_col: str, lag: int):
    ma_df = pd.DataFrame(index=group[group["cached"] != True].index)

    for index in ma_df.index:
        i = group.index.get_loc(index)
        if i >= lag and group["date"].iloc[i] - group["date"].iloc[
            i - lag
        ] <= timedelta(days=365):
            ma_df.loc[group.index[i], f"{match_stat_col}_ma{lag}"] = (
                group[match_stat_col].iloc[i - lag : i].mean()
            )

    return ma_df


def create_ma_features(
    data: pd.DataFrame, cached_data: pd.DataFrame, ma_lag: int, parameters: dict
) -> pd.DataFrame:
    excluded = ["pts_gap_above", "pts_gap_below", "pts_b4_match", "start", "round"]
    ma_features = [
        col
        for col in data.select_dtypes(include=["int", "float"]).columns
        if col not in excluded
    ]
    if parameters["debug_run"]:
        ma_features = ma_features[:3]
    for feature in tqdm(ma_features, desc="Creating MA features"):
        data = single_ma_feature(data, cached_data, feature, ma_lag)

    return data


def select_most_common(row, df_counts):
    if pd.isna(row["pos"]):
        player_df = df_counts.loc[df_counts["fpl_name"] == row["fpl_name"]]
        if len(player_df) > 0:
            return player_df.loc[player_df["counts"].idxmax(), "pos"]
        else:
            return row["pos"]
    else:
        positions = row["pos"].split(",")
        counts = [
            df_counts.loc[
                (df_counts["fpl_name"] == row["fpl_name"]) & (df_counts["pos"] == pos),
                "counts",
            ].values[0]
            for pos in positions
        ]
        return positions[np.argmax(counts)]


def extract_mode_pos(data: pd.DataFrame, cached_data: pd.DataFrame) -> pd.DataFrame:
    unpivot_pos = (
        data.assign(pos=data["pos"].str.split(","))
        .explode("pos")
        .dropna(subset=["pos"])
    )
    df_counts = (
        unpivot_pos.groupby(["fpl_name", "pos"]).size().reset_index(name="counts")
    )

    if cached_data is not None:
        cached_data = (
            cached_data.groupby(["fpl_name", "pos"]).size().reset_index(name="counts")
        )
        df_counts = pd.merge(
            df_counts, cached_data, on=["fpl_name", "pos"], how="outer"
        )
        df_counts["counts"] = df_counts["counts_x"].fillna(0) + df_counts[
            "counts_y"
        ].fillna(0)

    logger.info("Processing positions...")
    data["pos"] = data.parallel_apply(
        lambda row: select_most_common(row, df_counts), axis=1
    )
    data["pos"] = (
        data["pos"].map({"MF": "CM", "DF": "CB"}).fillna(data["pos"])
    )  # MF and DF are only used in 2016-2017
    logger.info("Done")
    return data


def feature_engineering(
    data: pd.DataFrame, read_processed_data: pd.DataFrame, parameters
) -> pd.DataFrame:
    use_cache = parameters["use_cache"]
    ma_lag = parameters["ma_lag"]
    n_cores = parameters["n_cores"]
    holdout_year = parameters["holdout_year"]

    pandarallel.initialize(progress_bar=True, nb_workers=n_cores)

    if parameters["use_cache"] and read_processed_data is not None:
        cached_data = read_processed_data.loc[read_processed_data["cached"] == True]
        cached_date = cached_data["date"].max()
        data = data[data["date"] > cached_date]
        if data.empty:
            return cached_data
    else:
        cached_data = None

    data = agg_home_away_elo(data)
    data = calculate_pts_data(data, cached_data)

    train_set = data["season"] < holdout_year
    data.loc[train_set] = data.loc[train_set].dropna(subset=["player"])

    data = extract_mode_pos(data, cached_data)
    data = create_ma_features(data, cached_data, ma_lag, parameters)
    # TODO: calculate total points per team
    # TODO: calculate % point of each player within the team

    if use_cache:
        data = pd.concat([cached_data, data], ignore_index=True)

    one_week_ago = datetime.now() - timedelta(days=7)
    data.loc[data["date"] <= one_week_ago, "cached"] = True
    data = reorder_columns(data)

    return data


def reorder_columns(data):
    start_cols = ["season", "fpl_name", "round", "date", "player"]
    end_cols = ["fpl_points", "cached"]
    new_columns = (
        start_cols
        + [col for col in data.columns if col not in start_cols + end_cols]
        + end_cols
    )
    data = data[new_columns]
    return data
