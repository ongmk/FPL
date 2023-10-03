import itertools
import logging
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from src.fpl.pipelines.model_pipeline.preprocessing.feature_engineering import (
    feature_engineering,
)
from src.fpl.pipelines.model_pipeline.preprocessing.imputer import impute_missing_values
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
            "round",
            "date",
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
    fpl_data = fpl_data[
        [
            "season",
            "date",
            "round",
            "full_name",
            "total_points",
            "value",
            "team",
            "opponent_team_name",
            "was_home",
            "position",
            "team_h_score",
            "team_a_score",
        ]
    ]

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
        fpl_data,
        on=["season", "date", "fpl_name"],
        how="right",
        suffixes=("", "_fpl"),
    )

    combined_data["round"] = combined_data["round_fpl"].fillna(combined_data["round"])
    combined_data["venue"] = combined_data["venue_fpl"].fillna(combined_data["venue"])
    combined_data["team_ga"] = combined_data["team_ga_fpl"].fillna(
        combined_data["team_ga"]
    )
    combined_data["team_gf"] = combined_data["team_gf_fpl"].fillna(
        combined_data["team_gf"]
    )
    combined_data["team"] = combined_data["team_fpl"].fillna(combined_data["team"])
    combined_data["opponent"] = combined_data["opponent_fpl"].fillna(
        combined_data["opponent"]
    )
    combined_data["pos"] = combined_data["pos"].fillna(combined_data["pos_fpl"])
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
    combined_data = combined_data.drop(
        [
            "next_match",
            "pos_fpl",
            "venue_fpl",
            "round_fpl",
            "team_fpl",
            "opponent_fpl",
            "team_ga_fpl",
            "team_gf_fpl",
        ],
        axis=1,
    )
    return combined_data


def align_data_structure(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    player_name_mapping: pd.DataFrame,
    fpl_2_fbref_team_mapping: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log["round"] = player_match_log["round"].apply(get_week_number)
    player_match_log["start"] = player_match_log["start"].replace({"Y": 1, "N": 0})
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
    player_match_log = player_match_log.rename(columns={"squad": "team"})
    player_match_log = player_match_log.drop(["player_season"], axis=1)

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

    fpl_data["team"] = fpl_data["team"].map(fpl_2_fbref_team_mapping)
    fpl_data["opponent"] = fpl_data["opponent_team_name"].map(fpl_2_fbref_team_mapping)
    fpl_data["team_gf"] = np.where(
        fpl_data["was_home"], fpl_data["team_h_score"], fpl_data["team_a_score"]
    )
    fpl_data["team_ga"] = np.where(
        fpl_data["was_home"], fpl_data["team_a_score"], fpl_data["team_h_score"]
    )
    fpl_data["venue"] = fpl_data["was_home"].map({True: "Home", False: "Away"})
    fpl_data["pos"] = fpl_data["position"].map(
        {
            "DEF": "CB,DF,WB,RB,LB",
            "FWD": "LW,RW,FW",
            "GK": "GK",
            "GKP": "GK",
            "MID": "DM,LM,CM,RM,MF,AM",
        }
    )
    fpl_data = fpl_data.rename(
        columns={"full_name": "fpl_name", "total_points": "fpl_points"}
    )
    fpl_data = fpl_data.drop(
        ["opponent_team_name", "was_home", "position", "team_a_score", "team_h_score"],
        axis=1,
    )
    fpl_data = fpl_data.sort_values(["date", "fpl_name"]).reset_index(drop=True)

    return player_match_log, team_match_log, elo_data, fpl_data


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def convert_to_datetime(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log["date"] = pd.to_datetime(player_match_log["date"])
    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
    elo_data["date"] = pd.to_datetime(elo_data["date"])
    fpl_data["date"] = pd.to_datetime(fpl_data["kickoff_time"].str[:10])
    return player_match_log, team_match_log, elo_data, fpl_data


def add_unplayed_matches(fpl_data: pd.DataFrame):
    output_data = pd.DataFrame()
    for season, season_data in fpl_data.groupby("season"):
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
        fill_dates = fill_dates.sort_values(["date", "fpl_name"])
        output_data = pd.concat([output_data, fill_dates])
    output_data = output_data.reset_index(drop=True)
    return output_data


def sample_players(data: pd.DataFrame) -> pd.DataFrame:
    seasons = data["season"].unique()
    sampled_data = []

    for season in seasons:
        season_data = data[data["season"] == season]
        teams = season_data["team"].unique()

        for team in teams:
            if pd.notna(team):
                team_data = season_data[season_data["team"] == team]
            else:
                team_data = season_data[season_data["team"].isna()]
            players = team_data["fpl_name"].unique()
            sampled_player = pd.Series(players).sample(n=1, random_state=42)
            sampled_season_data = season_data[
                season_data["fpl_name"].isin(sampled_player)
            ]
            sampled_data.append(sampled_season_data)

    sampled_data = pd.concat(sampled_data)

    return sampled_data


def clean_data(
    player_match_log,
    team_match_log,
    elo_data,
    fpl_data,
    player_name_mapping,
    fpl_2_fbref_team_mapping,
    read_processed_data,
    parameters,
):
    player_match_log, team_match_log, elo_data, fpl_data = convert_to_datetime(
        player_match_log, team_match_log, elo_data, fpl_data
    )
    if parameters["use_cache"]:
        cached_date = read_processed_data.loc[
            read_processed_data["cached"] == True, "date"
        ].max()
        player_match_log = player_match_log[player_match_log["date"] > cached_date]
        team_match_log = team_match_log[team_match_log["date"] > cached_date]
        elo_data = elo_data[elo_data["date"] > cached_date]
        fpl_data = fpl_data[fpl_data["date"] > cached_date]
        raise Exception("TODO")
    player_match_log, team_match_log, fpl_data = filter_data(
        player_match_log, team_match_log, fpl_data, parameters
    )
    player_match_log, team_match_log, elo_data, fpl_data = align_data_structure(
        player_match_log,
        team_match_log,
        elo_data,
        player_name_mapping,
        fpl_2_fbref_team_mapping,
        fpl_data,
        parameters,
    )
    fpl_data = add_unplayed_matches(fpl_data)
    combined_data = combine_data(
        player_match_log,
        team_match_log,
        elo_data,
        fpl_data,
        parameters,
    )
    if parameters["test_sampling"]:
        combined_data = sample_players(combined_data)

    return combined_data


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]

    ma_cols = processed_data.filter(regex=r"\w+_ma\d+").columns
    ma_cols = set([re.sub(r"_ma\d+", "", col) for col in ma_cols])
    excluded = [
        "value",
        "att_total",
        "home_att_total",
        "away_att_total",
        "def_total",
        "home_def_total",
        "away_def_total",
        "fpl_points",
    ]
    ma_cols = [col for col in ma_cols if col not in excluded]
    non_features = ["cached", "start", "match_points", "league_points"]
    processed_data = processed_data.drop(ma_cols + non_features, axis=1)

    train_val_data = processed_data[processed_data["season"] < holdout_year]
    holdout_data = processed_data[processed_data["season"] >= holdout_year]

    return train_val_data, holdout_data
