import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_data(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
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
        player_match_log["comp"] == "Premier League",
        [
            "season",
            "fpl_name",
            "player",
            "round",
            "date",
            "venue",
            "team",
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
        team_match_log["comp"] == "Premier League",
        [
            "team",
            "date",
            "opponent",
            "team_poss",
            "team_gf",
            "team_ga",
            "team_xg",
            "team_xga",
        ],
    ].reset_index(drop=True)

    fpl_data = fpl_data[
        [
            "season",
            "round",
            "date",
            "element",
            "fpl_name",
            "team",
            "opponent",
            "fpl_points",
            "value",
            "kickoff_time",
            "venue",
            "team_ga",
            "team_gf",
            "pos",
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
        how="left",
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
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    elo_data = elo_data.drop(["date", "season"], axis=1)
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
    if parameters["debug_run"]:
        combined_data = sample_players(combined_data)
    return combined_data


def align_data_structure(
    data_check_complete,
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
    player_name_mapping: pd.DataFrame,
    fpl_2_fbref_team_mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log, team_match_log, fpl_data = convert_to_datetime(
        player_match_log, team_match_log, fpl_data
    )
    player_match_log["round"] = player_match_log["round"].apply(get_week_number)
    player_match_log["start"] = player_match_log["start"].replace({"Y": 1, "N": 0})
    player_name_mapping = player_name_mapping.set_index("fbref_name")[
        "fpl_name"
    ].to_dict()
    player_match_log["fpl_name"] = player_match_log["player"].map(player_name_mapping)
    player_match_log = player_match_log.rename(columns={"squad": "team"})

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

    fpl_data["team"] = fpl_data["team_name"].map(fpl_2_fbref_team_mapping)
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
        [
            "team_name",
            "opponent_team_name",
            "was_home",
            "position",
            "team_a_score",
            "team_h_score",
        ],
        axis=1,
    )
    fpl_data = fpl_data.sort_values(["date", "fpl_name"]).reset_index(drop=True)

    return filter_data(player_match_log, team_match_log, fpl_data)


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def convert_to_datetime(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log["date"] = pd.to_datetime(player_match_log["date"])
    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
    fpl_data["date"] = pd.to_datetime(fpl_data["kickoff_time"].str[:10])
    return player_match_log, team_match_log, fpl_data


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


def split_data(processed_data, data_params, model_params):
    train_start_year = data_params["train_start_year"]
    holdout_year = data_params["holdout_year"]
    group_by = model_params["group_by"]
    target = model_params["target"]
    categorical_features = model_params["categorical_features"]
    numerical_features = model_params["numerical_features"]
    info_columns = model_params["info_columns"]

    features = categorical_features + numerical_features
    features_labels = features + [target]
    useful_cols = [group_by] + info_columns + features_labels

    train_val_data = processed_data.loc[
        (processed_data["season"] >= train_start_year)
        & (processed_data["season"] < holdout_year),
        useful_cols,
    ]
    train_val_data = drop_incomplete_data(
        train_val_data,
        features_labels,
        "train_val",
    )

    holdout_data = processed_data.loc[
        processed_data["season"] >= holdout_year, useful_cols
    ]
    holdout_data = drop_incomplete_data(
        holdout_data,
        features,
        "holdout",
    )

    return train_val_data, holdout_data


def drop_incomplete_data(
    data: pd.DataFrame, feature_columns: list[str], split: str
) -> pd.DataFrame:
    original_len = len(data)
    data = data.dropna(subset=feature_columns)

    filtered_len = original_len - len(data)
    filtered_percentage = filtered_len / original_len * 100

    if filtered_percentage > 30:
        raise RuntimeError(
            f"[{split}] Too many rows {filtered_len}/{original_len}({filtered_percentage:.2f}%) have been filtered out"
        )
    logger.info(
        f"[{split}] {filtered_len}/{original_len}({filtered_percentage:.2f}%) rows are filtered out because they contain NaN."
    )
    return data


if __name__ == "__main__":
    import sqlite3

    import yaml

    conn = sqlite3.connect("./data/fpl.db")
    cur = conn.cursor()

    fpl_name = "Vitalii Mykolenko"
    fbref_name = "Vitaliy Mykolenko"
    fpl_data = pd.read_sql(
        f"select * from raw_fpl_data where full_name = '{fpl_name}'", conn
    )
    player_match_log = pd.read_sql(
        f"select * from raw_player_match_log where player = '{fbref_name}'", conn
    )
    teams = (
        player_match_log.loc[player_match_log["comp"] == "Premier League", "squad"]
        .unique()
        .tolist()
    )
    teams_string = [f"'{t}'" for t in teams]
    teams_string = "(" + ",".join(teams_string) + ")"
    elo_data = pd.read_sql(f"select * from elo_data where team in {teams_string}", conn)
    team_match_log = pd.read_sql(
        f"select * from raw_team_match_log where team in {teams_string}", conn
    )
    player_name_mapping = pd.read_csv("data/preprocess/player_name_mapping.csv")
    with open("data/preprocess/fpl2fbref_team_mapping.yml", "r") as file:
        fpl_2_fbref_team_mapping = yaml.safe_load(file)

    parameters = {"debug_run": False}
