import itertools
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from thefuzz import process
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fuzzy_match_player_names(
    player_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
    fbref2fpl_player_overrides: dict[str, str],
) -> None:
    fpl_data = (
        fpl_data[["full_name", "total_points"]]
        .groupby(["full_name"], as_index=False)
        .sum()
        .rename(columns={"full_name": "fpl_name"})
    )

    player_match_log = player_match_log[["link", "player"]].rename(
        columns={"player": "fbref_name", "link": "fbref_id"}
    )
    player_match_log["fbref_id"] = player_match_log["fbref_id"].str.extract(
        r"/players/([a-f0-9]+)/"
    )
    matched_df = player_match_log.drop_duplicates()
    tqdm.pandas(desc=f"Fuzzy matching player names")
    matched_df["fpl_name"] = (
        matched_df["fbref_name"]
        .progress_apply(
            lambda fbref_name: process.extract(
                fbref_name, fpl_data["fpl_name"].tolist(), limit=1
            )
        )
        .explode()
    )
    name_only_mapping = {
        fbref_name: (fpl_name, 100) for fbref_name, fpl_name in fbref2fpl_player_overrides["name_only_mapping"].items()
    }
    id_name_mapping = {
        (fbref_id, fbref_name): (fpl_name, 100)
        for fbref_id, fbref_name, fpl_name in fbref2fpl_player_overrides[
            "id_name_mapping"
        ]
    }
    matched_df["fpl_name"] = (
        matched_df["fbref_name"].map(name_only_mapping).fillna(matched_df["fpl_name"])
    )

    matched_df["fpl_name"] = matched_df.apply(
        lambda row: id_name_mapping.get((row["fbref_id"], row["fbref_name"]), None),
        axis=1,
    ).fillna(matched_df["fpl_name"])

    matched_df["fuzzy_score"] = matched_df["fpl_name"].str[1]
    matched_df["fpl_name"] = matched_df["fpl_name"].str[0]

    matched_df = pd.merge(matched_df, fpl_data, on=["fpl_name"], how="outer")
    matched_df.loc[
        ((matched_df["fuzzy_score"] < 90) | (matched_df["fuzzy_score"].isna()))
        & (matched_df["total_points"] > 0),
        "review",
    ] = 1
    matched_df["duplicated"] = (
        matched_df["fpl_name"].duplicated(keep=False).map({True: 1, False: pd.NA})
    )
    matched_df.loc[
        (matched_df["fbref_name"].isna()) & (matched_df["total_points"] > 0),
        "missing_matchlogs",
    ] = 1
    if matched_df["review"].sum() > 0:
        logger.warning(
            f"{int(matched_df['review'].sum())}/{len(matched_df)} records in player name mappings needs review."
        )
    if matched_df["duplicated"].sum() > 0:
        logger.warning(
            f"{int(matched_df['duplicated'].sum())} duplicated player name mappings."
        )
    if matched_df["missing_matchlogs"].sum() > 0:
        logger.warning(
            f"There are missing FBRef matchlogs for {int(matched_df['missing_matchlogs'].sum())} players."
        )
    return matched_df


def data_checks(player_name_mapping):
    duplicated_mappings = int(player_name_mapping["duplicated"].sum())
    if duplicated_mappings > 0:
        raise ValueError(
            f"There are {duplicated_mappings} duplicated player name mappings."
        )
    n_missing_matchlog = int(player_name_mapping["missing_matchlogs"].sum())
    if n_missing_matchlog > 0:
        raise ValueError(
            f"There are missing FBRef matchlogs for {n_missing_matchlog} players."
        )
    logger.info("Data checks completed.")
    return True


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
    player_name_mapping = player_name_mapping.set_index("fbref_name")[
        "fpl_name"
    ].to_dict()
    player_match_log["fpl_name"] = (
        player_match_log["player"]
        .map(player_name_mapping)
        .fillna(player_match_log["player"])
    )
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

    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    elo_data = elo_data.drop(["date", "season"], axis=1)

    fpl_data = add_unplayed_matches(fpl_data)
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
                    season_data["full_name"].unique(),
                    season_data["round"].dropna().unique(),
                )
            ),
            columns=["season", "full_name", "round"],
        )
        fill_dates = player_round.merge(
            season_data, on=["season", "full_name", "round"], how="left"
        )
        fill_dates = fill_dates.sort_values(["date", "full_name"])
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
    data_check_complete,
    player_match_log,
    team_match_log,
    elo_data,
    fpl_data,
    player_name_mapping,
    fpl_2_fbref_team_mapping,
    parameters,
):
    player_match_log, team_match_log, elo_data, fpl_data = convert_to_datetime(
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
        fpl_2_fbref_team_mapping,
        fpl_data,
        parameters,
    )
    combined_data = combine_data(
        player_match_log,
        team_match_log,
        elo_data,
        fpl_data,
        parameters,
    )
    if parameters["debug_run"]:
        combined_data = sample_players(combined_data)

    return combined_data


def split_data(processed_data, data_params, model_params):
    holdout_year = data_params["holdout_year"]
    group_by = model_params["group_by"]
    target = model_params["target"]
    categorical_features = model_params["categorical_features"]
    numerical_features = model_params["numerical_features"]

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
    non_features += processed_data.filter(regex="_elo$").columns.tolist()
    non_features += processed_data.filter(regex="_opp$").columns.tolist()
    processed_data = processed_data.drop(ma_cols + non_features, axis=1)
    original_len = len(processed_data)
    contains_na = processed_data.isna().any(axis=1)
    filtered_rows = contains_na.sum()
    filtered_data = processed_data[~contains_na]
    if filtered_rows / original_len > 0.3:
        raise RuntimeError(
            f"Too many rows ({filtered_rows}/{original_len}) have been filtered out"
        )
    logger.info(
        f"{filtered_rows}/{original_len} rows are filtered out because they contain NaN."
    )
    filtered_data = filtered_data[
        [group_by] + categorical_features + numerical_features + [target]
    ]

    train_val_data = filtered_data[filtered_data["season"] < holdout_year]
    holdout_data = filtered_data[filtered_data["season"] >= holdout_year]

    return train_val_data, holdout_data


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

    parameters = {"start_year": "2016-2017", "debug_run": False}

    df = clean_data(
        player_match_log,
        team_match_log,
        elo_data,
        fpl_data,
        player_name_mapping,
        fpl_2_fbref_team_mapping,
        parameters,
    )
    assert len(df) > 0
    pass
