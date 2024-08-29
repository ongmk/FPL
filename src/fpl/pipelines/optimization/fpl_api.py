import logging
from collections import defaultdict
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_current_season_str(events_data: list[dict]) -> str:
    year_fixture_count = defaultdict(int)

    for fixture in events_data:
        year = fixture["deadline_time"].split("-")[0]
        year_fixture_count[year] += 1

    top_years = sorted(year_fixture_count, key=year_fixture_count.get, reverse=True)[:2]
    top_years_sorted = sorted(top_years)

    return f"{top_years_sorted[0]}-{top_years_sorted[1]}"


def get_fpl_base_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], str]
):
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    fpl_data = r.json()

    gameweeks = {"current": 0, "finished": [], "future": []}
    for w in fpl_data["events"]:
        if w["is_current"] == True:
            gameweeks["current"] = w["id"]
        elif w["finished"] == True:
            gameweeks["finished"].append(w["id"])
        else:
            gameweeks["future"].append(w["id"])

    element_data = pd.DataFrame(fpl_data["elements"]).set_index("id")

    team_data = pd.DataFrame(fpl_data["teams"]).set_index("id")
    element_data["team"] = element_data["team"].map(team_data["name"].to_dict())

    type_data = pd.DataFrame(fpl_data["element_types"]).set_index(["id"])
    element_data["position"] = element_data["element_type"].map(
        type_data["singular_name_short"].to_dict()
    )

    element_data["full_name"] = element_data["first_name"].str.cat(
        element_data["second_name"], sep=" "
    )
    element_data = element_data[
        [
            "web_name",
            "full_name",
            "team",
            "element_type",
            "position",
            "now_cost",
            "chance_of_playing_next_round",
        ]
    ]
    current_season = get_current_season_str(fpl_data["events"])

    return element_data, team_data, type_data, gameweeks, current_season


def fetch_player_fixtures(
    player_id: int, current_season: str
) -> list[pd.DataFrame, pd.DataFrame]:
    r = requests.get(
        f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    )
    data = r.json()
    history = pd.DataFrame(
        data["history"],
        columns=[
            "element",
            "round",
            "fixture",
            "opponent_team",
            "kickoff_time",
            "total_points",
            "was_home",
            "team_h_score",
            "team_a_score",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "value",
            "transfers_balance",
            "selected",
            "transfers_in",
            "transfers_out",
            "starts",
            "expected_goals",
            "expected_goal_involvements",
            "expected_assists",
            "expected_goals_conceded",
        ],
    )
    fixtures = pd.DataFrame(
        data["fixtures"],
        columns=[
            "element",
            "kickoff_time",
            "opponent_team",
            "is_home",
            "id",
            "event",
            "starts",
            "team_a",
            "team_h",
        ],
    )
    if len(history) > 0:
        history = history[
            ~history["fixture"].isin(fixtures["id"])
        ]  # sometimes the same fixture appears in history and fixtures
    fixtures["element"] = player_id
    fixtures["team"] = fixtures.apply(
        lambda row: row["team_h"] if row["is_home"] else row["team_a"], axis=1
    )
    fixtures["opponent_team"] = fixtures.apply(
        lambda row: row["team_a"] if row["is_home"] else row["team_h"], axis=1
    )
    fixtures = fixtures.drop(columns=["team_a", "team_h"])
    fixtures = fixtures.rename(
        {"is_home": "was_home", "id": "fixture", "event": "round"}, axis=1
    )
    fixtures = pd.concat([history, fixtures])
    fixtures["season"] = current_season
    return fixtures


def get_most_recent_fpl_game() -> dict:
    r = requests.get(f"https://fantasy.premierleague.com/api/fixtures/")
    data = r.json()
    data = sorted(data, key=lambda f: f["kickoff_time"], reverse=True)
    most_recent_fixture = next((d for d in data if d["finished"] == True), None)
    return most_recent_fixture


def get_current_season_fpl_data() -> pd.DataFrame:
    element_data, team_data, _, _, current_season = get_fpl_base_data()

    current_season_data = []
    for idx, row in tqdm(
        element_data.iterrows(), desc="Fetching player history", total=len(element_data)
    ):
        player_fixtures = fetch_player_fixtures(idx, current_season)
        player_fixtures["value"] = player_fixtures["value"].fillna(row["now_cost"])
        current_season_data.append(player_fixtures)
    current_season_data = pd.concat(current_season_data, ignore_index=True)

    opponents = (
        current_season_data.groupby(["was_home", "fixture"])["opponent_team"]
        .apply(lambda x: x.unique()[0])
        .to_dict()
    )
    missing_team = current_season_data["team"].isna()
    current_season_data.loc[missing_team, "team"] = current_season_data.loc[
        missing_team
    ].apply(lambda x: opponents[(not x["was_home"], x["fixture"])], axis=1)

    current_season_data = pd.merge(
        element_data.drop(columns=["team"]),
        current_season_data,
        left_on="id",
        right_on="element",
    )
    current_season_data["opponent_team_name"] = current_season_data[
        "opponent_team"
    ].map(team_data["name"].to_dict())
    current_season_data["team_name"] = current_season_data["team"].map(
        team_data["name"].to_dict()
    )
    current_season_data = current_season_data[
        [
            "season",
            "round",
            "element",
            "full_name",
            "team",
            "team_name",
            "position",
            "fixture",
            "starts",
            "opponent_team",
            "opponent_team_name",
            "total_points",
            "was_home",
            "kickoff_time",
            "team_h_score",
            "team_a_score",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "value",
            "transfers_balance",
            "selected",
            "transfers_in",
            "transfers_out",
            "expected_goals",
            "expected_goal_involvements",
            "expected_assists",
            "expected_goals_conceded",
        ]
    ]
    return current_season_data


def get_fpl_team_data(team_id: int, gw: int) -> list[dict]:
    general_request = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    )
    general_data = general_request.json()
    transfer_request = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
    )
    transfer_data = reversed(transfer_request.json())
    history_request = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/history/"
    )
    history_data = history_request.json()
    picks_request = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    )
    picks_data = picks_request.json()
    if picks_data.get("detail") == "Not found.":
        logger.warn(
            f"Team {team_id} has not made picks for GW {gw}. Setting intial squad to empty."
        )
        initial_squad = []
    else:
        initial_squad = picks_data["picks"]
        initial_squad = [p["element"] for p in initial_squad]
    return general_data, transfer_data, initial_squad, history_data


if __name__ == "__main__":
    fetch_player_fixtures(30, "2023-2024")
