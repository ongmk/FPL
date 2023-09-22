import itertools
import logging
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from thefuzz import process
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fuzzy_match_player_names(
    player_match_log: pd.DataFrame, fpl_data: pd.DataFrame, overrides: dict[str, str]
) -> None:
    player_match_log = (
        player_match_log[["season", "player"]]
        .drop_duplicates()
        .rename({"player": "fbref_name"}, axis=1)
    )
    fpl_data = (
        fpl_data[["season", "full_name", "total_points"]]
        .groupby(["season", "full_name"], as_index=False)
        .sum()
        .rename(columns={"full_name": "fpl_name"})
    )
    matched_df = []
    for season in player_match_log["season"].unique():
        player_match_log_season = player_match_log[
            player_match_log["season"] == season
        ].copy()
        fpl_data_season = fpl_data[fpl_data["season"] == season].copy()

        tqdm.pandas(desc=f"Fuzzy matching {season}")
        player_match_log_season["fpl_name"] = player_match_log_season[
            "fbref_name"
        ].progress_apply(
            lambda fbref_name: process.extract(
                fbref_name, fpl_data_season["fpl_name"].tolist(), limit=1
            )
        )
        player_match_log_season = player_match_log_season.explode("fpl_name")
        player_match_log_season["fuzzy_score"] = player_match_log_season[
            "fpl_name"
        ].str[1]
        player_match_log_season["fpl_name"] = player_match_log_season["fpl_name"].str[0]
        matched_df.append(player_match_log_season)
    matched_df = pd.concat(matched_df, ignore_index=True)
    for fbref_name, fpl_name in overrides.items():
        matched_df.loc[
            matched_df["fbref_name"] == fbref_name, ["fpl_name", "fuzzy_score"]
        ] = (fpl_name, 100)
    matched_df = pd.merge(matched_df, fpl_data, on=["season", "fpl_name"], how="outer")
    matched_df.loc[
        ((matched_df["fuzzy_score"] < 90) | (matched_df["fuzzy_score"].isna()))
        & (matched_df["total_points"] > 0),
        "review",
    ] = True
    matched_df.loc[
        matched_df["fpl_name"].notnull(), "duplicated"
    ] = matched_df.duplicated(subset=["season", "fpl_name"], keep=False).replace(
        False, np.nan
    )
    logger.warning(
        f"{matched_df['review'].sum()}/{len(matched_df)} records in player name mapping needs review."
    )
    logger.warning(
        f"{matched_df['duplicated'].sum()}/{len(matched_df)} duplicated mappings."
    )
    return matched_df


def filter_data(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
):
    team_match_log["days_till_next"] = (
        (team_match_log.groupby("team")["date"].shift(-1) - team_match_log["date"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    team_match_log["days_since_last"] = (
        (team_match_log["date"] - team_match_log.groupby("team")["date"].shift(1))
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    player_match_log = player_match_log.loc[
        (player_match_log["comp"] == "Premier League")
        & (player_match_log["season"] >= parameters["start_year"]),
        [
            "season",
            "player",
            "date",
            "round",
            "venue",
            "squad",
            "opponent",
            "start",
            "pos",
            "min",
            "gls",
            "ast",
            "pk",
            "pkatt",
            "sh",
            "sot",
            "touches",
            "xg",
            "npxg",
            "xag",
            "sca",
            "gca",
            "sota",
            "ga",
            "saves",
            "savepct",
            "cs",
            "psxg",
        ],
    ].reset_index(drop=True)
    team_match_log = team_match_log.loc[
        (team_match_log["comp"] == "Premier League")
        & (team_match_log["season"] >= parameters["start_year"]),
        [
            "team",
            "date",
            "opponent",
            "poss",
            "gf",
            "ga",
            "xg",
            "xga",
        ],
    ].reset_index(drop=True)
    fpl_data = fpl_data[["season", "date", "full_name", "fpl_points", "value"]]

    return player_match_log, team_match_log, fpl_data


def combine_data(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    combined_data = pd.merge(
        player_match_log,
        team_match_log,
        on=["team", "opponent", "date"],
        how="inner",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["team", "date"],
        right_on=["team", "next_match"],
        how="left",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["opponent", "date"],
        right_on=["team", "next_match"],
        how="left",
        suffixes=("", "_opp"),
    )
    combined_data = pd.merge(
        combined_data,
        fpl_data,
        on=["season", "date", "fpl_name"],
        how="right",
    )
    combined_data = combined_data.drop(["next_match"], axis=1)
    return combined_data


def align_data_structure(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    player_name_mapping: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_name_mapping = player_name_mapping.set_index(["fbref_name", "season"])[
        "fpl_name"
    ].to_dict()
    player_match_log["player_season"] = list(
        zip(player_match_log["player"], player_match_log["season"])
    )
    player_match_log["fpl_name"] = (
        player_match_log["player_season"]
        .map(player_name_mapping)
        .fillna(player_match_log["player_season"].str[0])
    )
    player_match_log = player_match_log.sort_values(
        ["date", "squad", "fpl_name"]
    ).reset_index(drop=True)
    player_match_log = player_match_log.rename(columns={"squad": "team"})
    player_match_log = player_match_log.drop(["player_season"], axis=1)

    team_match_log = team_match_log.sort_values(["date", "team"]).reset_index(drop=True)
    team_match_log = team_match_log.rename(
        columns={
            "poss": "team_poss",
            "gf": "team_gf",
            "ga": "team_ga",
            "xg": "team_xg",
            "xga": "team_xga",
        },
        errors="raise",
    )

    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    elo_data = elo_data.drop(["date", "season"], axis=1)

    fpl_data = fpl_data.rename(columns={"full_name": "fpl_name"})

    return player_match_log, team_match_log, elo_data, fpl_data


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def ensure_proper_dtypes(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log["date"] = pd.to_datetime(player_match_log["date"])
    player_match_log["round"] = player_match_log["round"].apply(get_week_number)
    player_match_log["start"] = player_match_log["start"].replace({"Y": 1, "N": 0})

    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
    team_match_log["round"] = team_match_log["round"].apply(get_week_number)

    elo_data["date"] = pd.to_datetime(elo_data["date"])

    fpl_data["date"] = pd.to_datetime(fpl_data["kickoff_time"].str[:10])
    fpl_data["fpl_points"] = fpl_data["total_points"]

    return player_match_log, team_match_log, elo_data, fpl_data


def add_unplayed_matches(player_match_log: pd.DataFrame):
    output_data = pd.DataFrame()
    for season, season_data in player_match_log.groupby("season"):
        player_round = pd.DataFrame(
            list(
                itertools.product(
                    [season],
                    season_data["fpl_name"].unique(),
                    season_data["round"].dropna().unique(),
                )
            ),
            columns=["season", "fpl_name", "round"],
        )
        fill_dates = player_round.merge(
            season_data, on=["season", "fpl_name", "round"], how="left"
        )
        fill_dates = fill_dates.sort_values("date")
        output_data = pd.concat([output_data, fill_dates])
    output_data = output_data.reset_index(drop=True)
    return output_data


def impute_missing_values(player_match_log: pd.DataFrame) -> pd.DataFrame:
    player_match_log = add_unplayed_matches(player_match_log)
    return player_match_log


def clean_data(
    player_match_log,
    team_match_log,
    elo_data,
    fpl_data,
    player_name_mapping,
    parameters,
):
    player_match_log, team_match_log, elo_data, fpl_data = ensure_proper_dtypes(
        player_match_log, team_match_log, elo_data, fpl_data
    )
    player_match_log, team_match_log, fpl_data = filter_data(
        player_match_log, team_match_log, fpl_data, parameters
    )
    player_match_log, team_match_log, elo_data, fpl_data = align_data_structure(
        player_match_log,
        team_match_log,
        elo_data,
        player_name_mapping,
        fpl_data,
        parameters,
    )
    player_match_log = impute_missing_values(player_match_log)
    combined_data = combine_data(
        player_match_log,
        team_match_log,
        elo_data,
        fpl_data,
        parameters,
    )
    return combined_data


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


def calculate_pts_data(data: pd.DataFrame) -> pd.DataFrame:
    pts_data = data.copy().drop_duplicates(["season", "team", "date"])

    pts_data = pts_data[["season", "team", "date", "team_gf", "team_ga"]]

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
    pts_data = pts_data.drop(["match_points", "league_points"], axis=1)

    return pts_data


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
    df = df.sort_values(by=["fpl_name", "date"])

    # Create moving average
    ma_df = df.groupby("fpl_name").apply(
        lambda x: calculate_multi_lag_ma(x, match_stat_col, lag)
    )
    return pd.merge(df, ma_df, left_index=True, right_index=True)


from pandas.core.groupby.generic import DataFrameGroupBy


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
    data: pd.DataFrame, ma_features: [str], ma_lag: int
) -> pd.DataFrame:
    for feature in tqdm(ma_features, desc="Creating MA features"):
        data = single_ma_feature(data, feature, ma_lag)

    data = data.drop(
        [
            col
            for col in ma_features
            if col
            not in [
                "value",
                "att_total",
                "home_att_total",
                "away_att_total",
                "def_total",
                "home_def_total",
                "away_def_total",
                "fpl_points",
            ]
        ],
        axis=1,
    )
    return data


# Define a function to select the position with the highest count
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


def extract_mode_pos(df: pd.DataFrame) -> pd.DataFrame:
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


def feature_engineering(data: pd.DataFrame, parameters) -> pd.DataFrame:
    data = agg_home_away_elo(data)

    pts_data = calculate_pts_data(data)
    data = data.merge(pts_data, on=["season", "team", "date"])
    data = extract_mode_pos(data)

    # data = create_ma_features(data, parameters["ma_features"], parameters["ma_lag"])

    return data


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]
    train_val_data = processed_data[processed_data["season"] < holdout_year]
    holdout_data = processed_data[processed_data["season"] >= holdout_year]

    return train_val_data, holdout_data
