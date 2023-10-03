import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm


def agg_home_away_elo(data: pd.DataFrame) -> pd.DataFrame:
    data["att_total"] = data.att_elo + data.def_elo_opp
    data["home_att_total"] = data.home_att_elo + data.away_def_elo_opp
    data["away_att_total"] = data.away_att_elo + data.home_def_elo_opp
    data["def_total"] = data.def_elo + data.att_elo_opp
    data["home_def_total"] = data.home_def_elo + data.away_att_elo_opp
    data["away_def_total"] = data.away_def_elo + data.home_att_elo_opp
    data = data.drop(data.filter(regex="_elo$").columns, axis=1)
    data = data.drop(data.filter(regex="_opp$").columns, axis=1)
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
    if cached_data:
        pts_data = cached_data
        raise Exception("TODO")

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


def single_ma_feature(df: pd.DataFrame, match_stat_col: str, lag: int):
    df = df.sort_values(by=["date"])

    # Create moving average
    ma_df = df.groupby("fpl_name").apply(
        lambda x: calculate_multi_lag_ma(x, match_stat_col, lag)
    )
    return pd.merge(df, ma_df, left_index=True, right_index=True)


def calculate_multi_lag_ma(group: DataFrameGroupBy, match_stat_col: str, max_lag: int):
    ma_df = pd.DataFrame(index=group.index)

    for i in range(len(group)):
        for lag in range(1, max_lag + 1):
            if i >= lag and group["date"].iloc[i] - group["date"].iloc[
                i - lag
            ] <= timedelta(days=365):
                ma_df.loc[group.index[i], f"{match_stat_col}_ma{lag}"] = (
                    group[match_stat_col].iloc[i - lag : i].mean()
                )

    return ma_df


def create_ma_features(
    data: pd.DataFrame, cached_data: pd.DataFrame, ma_lag: int
) -> (pd.DataFrame, [str]):
    if cached_data:
        raise Exception("TODO")
    excluded = ["pts_gap_above", "pts_gap_below", "pts_b4_match", "start", "round"]
    ma_features = [
        col
        for col in data.select_dtypes(include=["int", "float"]).columns
        if col not in excluded
    ]
    for feature in tqdm(ma_features, desc="Creating MA features"):
        data = single_ma_feature(data, feature, ma_lag)

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


def extract_mode_pos(df: pd.DataFrame, cached_data: pd.DataFrame) -> pd.DataFrame:
    if cached_data:
        raise Exception("TODO")
    # Step 1: Split 'pos' column into separate rows
    df_new = df.assign(pos=df["pos"].str.split(",")).explode("pos")

    # Step 2: Group by 'player' and 'pos' and count the number of each position for each player
    df_counts = df_new.groupby(["fpl_name", "pos"]).size().reset_index(name="counts")

    # # Step 3: Apply the function to each row
    tqdm.pandas(desc="Processing positions")
    df["pos"] = df.progress_apply(
        lambda row: select_most_common(row, df_counts), axis=1
    )
    return df


def feature_engineering(
    data: pd.DataFrame, read_processed_data: pd.DataFrame, parameters
) -> pd.DataFrame:
    use_cache = parameters["use_cache"]
    ma_lag = parameters["ma_lag"]
    cached_data = (
        read_processed_data.loc[read_processed_data["cached"] == True]
        if use_cache
        else None
    )

    data = agg_home_away_elo(data)
    data = calculate_pts_data(data, cached_data)
    data = extract_mode_pos(data, cached_data)
    data = create_ma_features(data, cached_data, ma_lag)

    if use_cache:
        data = pd.concat([cached_data, data])

    most_recent_fpl_data = data.loc[data["fpl_points"].notnull(), "date"].max()
    data.loc[data["date"] <= most_recent_fpl_data, "cached"] = True

    start_cols = ["season", "fpl_name", "round", "date", "player"]
    end_cols = ["fpl_points", "cached"]
    new_columns = (
        start_cols
        + [col for col in data.columns if col not in start_cols + end_cols]
        + end_cols
    )
    data = data[new_columns]

    return data
