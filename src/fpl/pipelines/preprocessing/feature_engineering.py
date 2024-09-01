import itertools
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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

    pts_data.loc[pts_data["team_gf"] > pts_data["team_ga"], "match_points"] = 3
    pts_data.loc[pts_data["team_gf"] == pts_data["team_ga"], "match_points"] = 1
    pts_data.loc[pts_data["team_gf"] < pts_data["team_ga"], "match_points"] = 0

    pts_data = pts_data.drop(columns=["team_gf", "team_ga"])
    pts_data["league_points"] = (
        pts_data.groupby(["season", "team"])["match_points"]
    ).shift(1)
    pts_data["pts_b4_match"] = (
        pts_data.groupby(["season", "team"])["league_points"].cumsum().fillna(0)
    )

    pts_data = features_from_pts(pts_data)

    return data.merge(pts_data, on=["season", "team", "date"])


def create_ma_features(
    data: pd.DataFrame, cached_data: pd.DataFrame, ma_lag: int
) -> pd.DataFrame:
    if cached_data is not None:
        cached_data = cached_data.loc[
            cached_data["season"] == cached_data["season"].max()
        ]
        data = pd.concat([cached_data, data], keys=["cached", "new"])

    excluded = ["pts_gap_above", "pts_gap_below", "pts_b4_match", "start", "round"]
    numerical_columns = [
        col
        for col in data.select_dtypes(include=["int", "float"]).columns
        if col not in excluded
    ]

    ma_features = [f"{col}_ma{ma_lag}" for col in numerical_columns]
    data[ma_features] = data.groupby("fpl_name")[numerical_columns].transform(
        lambda x: x.rolling(ma_lag, closed="left", min_periods=ma_lag).mean()
    )
    n_matches_ago = data.groupby("fpl_name")["date"].shift(ma_lag)
    too_long_ago = data["date"] - n_matches_ago > timedelta(days=365)
    data.loc[too_long_ago, ma_features] = np.nan

    if isinstance(data.index, pd.MultiIndex):
        data = data.loc["new"]

    return data


def extract_mode_pos(data: pd.DataFrame, cached_data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing positions...")
    unpivot_pos = (
        data.assign(pos=data["pos"].str.split(","))
        .explode("pos")
        .dropna(subset=["pos"])
    )
    df_counts = (
        unpivot_pos.groupby(["fpl_name", "pos"]).size().reset_index(name="pos_counts")
    )

    if cached_data is not None:
        cached_data = (
            cached_data.groupby(["fpl_name", "pos"])
            .size()
            .reset_index(name="pos_counts")
        )
        df_counts = pd.merge(
            df_counts, cached_data, on=["fpl_name", "pos"], how="outer"
        )
        df_counts["pos_counts"] = df_counts["pos_counts_x"].fillna(0) + df_counts[
            "pos_counts_y"
        ].fillna(0)
        df_counts = df_counts.drop(columns=["pos_counts_x", "pos_counts_y"])

    data = data.assign(pos=data["pos"].str.split(",")).explode("pos")
    data = pd.merge(data, df_counts, on=["fpl_name", "pos"], how="left")
    data = data.sort_values(["date", "fpl_name", "pos_counts"]).drop_duplicates(
        ["fpl_name", "date"], keep="last"
    )
    data["pos"] = (
        data["pos"].map({"MF": "CM", "DF": "CB"}).fillna(data["pos"])
    )  # MF and DF are only used in 2016-2017
    assert all(data.groupby(["fpl_name", "date"]).size() == 1)
    return data


def feature_engineering(
    data: pd.DataFrame, read_processed_data: pd.DataFrame, parameters
) -> pd.DataFrame:
    use_cache = parameters["use_cache"]
    ma_lag = parameters["ma_lag"]
    holdout_year = parameters["holdout_year"]

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
    data = create_ma_features(data, cached_data, ma_lag)
    # TODO: calculate total points per team
    # TODO: calculate % point of each player within the team

    if use_cache:
        data = pd.concat([cached_data, data], ignore_index=True)

    one_week_ago = data.loc[~data["fpl_points"].isna(), "date"].max() - timedelta(
        days=7
    )
    data["cached"] = data["date"] <= one_week_ago
    data = reorder_columns(data)

    return data


def reorder_columns(data):
    start_cols = ["season", "fpl_name", "round", "date", "player"]
    end_cols = ["minutes", "fpl_points", "cached"]
    new_columns = (
        start_cols
        + [col for col in data.columns if col not in start_cols + end_cols]
        + end_cols
    )
    data = data[new_columns]
    return data
