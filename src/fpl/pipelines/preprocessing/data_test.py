import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def data_tests(
    player_name_mapping: pd.DataFrame,
    fpl_data: pd.DataFrame,
    fpl_2_fbref_team_mapping: dict[str, str],
    team_match_log: pd.DataFrame,
) -> bool:
    check_player_name_mapping(player_name_mapping)

    check_team_name_mapping(fpl_data, fpl_2_fbref_team_mapping)

    fpl_data, team_match_log = align_format(
        fpl_data, fpl_2_fbref_team_mapping, team_match_log
    )

    check_fpl_teams_uniqueness(fpl_data)

    return True


def align_format(fpl_data, fpl_2_fbref_team_mapping, team_match_log):
    fpl_data["date"] = pd.to_datetime(fpl_data["kickoff_time"].str[:10])
    fpl_data["team"] = fpl_data["team_name"].map(fpl_2_fbref_team_mapping)
    fpl_data["opponent"] = fpl_data["opponent_team_name"].map(fpl_2_fbref_team_mapping)
    team_match_log = team_match_log.loc[team_match_log["comp"] == "Premier League"]
    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
    team_match_log["round"] = team_match_log["round"].apply(get_week_number)
    return fpl_data, team_match_log


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def check_fpl_teams_uniqueness(fpl_data):
    counts = fpl_data.groupby(
        ["season", "kickoff_time", "team", "opponent_team"]
    ).size()
    assert counts.min() >= 10, "Some team-opponent pairs have less than 10 matches."


def check_team_name_mapping(fpl_data, fpl_2_fbref_team_mapping):
    teams = fpl_data[["season", "team_name"]].drop_duplicates()
    teams["fbref_team"] = teams["team_name"].map(fpl_2_fbref_team_mapping)
    assert (
        teams["fbref_team"].isna().sum() == 0
    ), "Some FPL teams are not mapped to FBRef."


def check_player_name_mapping(player_name_mapping):
    duplicated_mappings = player_name_mapping["duplicated"].sum()
    assert (
        duplicated_mappings == 0
    ), f"There are {duplicated_mappings} duplicated player name mappings."

    n_missing_matchlog = int(player_name_mapping["missing_matchlogs"].sum())
    assert (
        n_missing_matchlog == 0
    ), f"There are missing FBRef matchlogs for {n_missing_matchlog} players."
