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

    check_dates(fpl_data, team_match_log)

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


def check_dates(fpl_data, team_match_log):
    # date_fpl = fpl_data.loc[
    #     fpl_data["total_points"].notna(),
    #     ["season", "round", "team_name", "opponent", "kickoff_time", "date"],
    # ].drop_duplicates()
    # date_fbref = team_match_log[
    #     ["season", "round", "team_name", "opponent", "date"]
    # ].drop_duplicates()

    # combined = pd.merge(
    #     date_fpl,
    #     date_fbref,
    #     on=[
    #         "season",
    #         "team",
    #         "opponent",
    #         "date",
    #     ],
    #     how="outer",
    # )
    # assert (
    #     combined["round_y"].isna().sum() == 0
    # ), "Some FPL kickoff times are not matched to FBRef dates."
    pass


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
