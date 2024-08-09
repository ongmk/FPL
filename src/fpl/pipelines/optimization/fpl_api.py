import itertools
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel
from tqdm import tqdm

from fpl.utils import PydanticDataFrame

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

    all_gws = {"current": 0, "finished": [], "future": []}
    for w in fpl_data["events"]:
        if w["is_current"] == True:
            all_gws["current"] = w["id"]
        elif w["finished"] == True:
            all_gws["finished"].append(w["id"])
        else:
            all_gws["future"].append(w["id"])

    element_data = pd.DataFrame(fpl_data["elements"])

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
            "id",
            "web_name",
            "full_name",
            "team",
            "element_type",
            "position",
            "now_cost",
        ]
    ]
    current_season = get_current_season_str(fpl_data["events"])

    return element_data, team_data, type_data, all_gws, current_season


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
            "total_points",
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
        player_fixtures = fetch_player_fixtures(row["id"], current_season)
        player_fixtures["value"] = player_fixtures["value"].fillna(row["now_cost"])
        current_season_data.append(player_fixtures)
    current_season_data = pd.concat(current_season_data, ignore_index=True)
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


class FplData(BaseModel):
    merged_data: PydanticDataFrame
    team_data: PydanticDataFrame
    type_data: PydanticDataFrame
    gameweeks: list[int]
    initial_squad: list[int]
    team_name: str
    in_the_bank: float
    free_transfers: int
    current_season: str

    class Config:
        arbitrary_types_allowed = True


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
    return general_data, transfer_data, initial_squad, history_data


def get_free_transfers(
    transfer_data: dict, history_data: dict, current_gameweek: int
) -> int:
    free_transfers = 1
    if current_gameweek <= 2:
        return free_transfers

    transfer_count = defaultdict(int)

    for transfer in transfer_data:
        transfer_count[transfer["event"]] += 1
    for chip in history_data["chips"]:
        if chip["name"] in ["wildcard", "bboost"]:
            transfer_count[chip["event"]] = 1

    # F2 = min(max(F1-T+1, 1), 5)
    for week in range(2, current_gameweek):
        free_transfers = min(max(free_transfers - transfer_count[week] + 1, 1), 5)
    return free_transfers


def get_live_data(
    inference_results: pd.DataFrame, parameters: dict[str, Any]
) -> FplData:
    team_id = parameters["team_id"]
    horizon = parameters["horizon"]

    elements_team, team_data, type_data, all_gws, current_season = get_fpl_base_data()
    gameweeks = all_gws["future"][:horizon]
    current_gw = all_gws["current"]

    pred_pts_data = get_pred_pts_data(inference_results, current_season, gameweeks)
    merged_data = merge_data(elements_team, pred_pts_data)

    general_data, transfer_data, initial_squad, history_data = get_fpl_team_data(
        team_id, current_gw
    )
    in_the_bank = (general_data["last_deadline_bank"] or 1000) / 10
    free_transfers = get_free_transfers(transfer_data, history_data, current_gw)

    initial_squad = [p["element"] for p in initial_squad]
    merged_data["sell_price"] = merged_data["now_cost"]
    for t in transfer_data:
        if t["element_in"] not in initial_squad:
            continue
        bought_price = t["element_in_cost"]
        merged_data.loc[t["element_in"], "bought_price"] = bought_price
        current_price = merged_data.loc[t["element_in"], "now_cost"]
        if current_price <= bought_price:
            continue
        sell_price = np.floor(np.mean([current_price, bought_price]))
        merged_data.loc[t["element_in"], "sell_price"] = sell_price

    keys = [k for k in merged_data.columns.to_list() if "xPts_" in k]
    merged_data["total_ev"] = merged_data[keys].sum(axis=1)
    merged_data = merged_data.sort_values(by=["total_ev"], ascending=[False])

    logger.info(f"Team: {general_data['name']}.")
    logger.info(f"Current week: {current_gw}")
    logger.info(f"Optimizing for {horizon} weeks. {gameweeks}")

    return FplData(
        merged_data=merged_data,
        team_data=team_data,
        type_data=type_data,
        gameweeks=gameweeks,
        initial_squad=initial_squad,
        team_name=general_data["name"],
        in_the_bank=in_the_bank,
        free_transfers=free_transfers,
        current_season=current_season,
    )


def merge_data(elements_team, pred_pts_data):
    merged_data = elements_team.merge(
        pred_pts_data,
        left_on="full_name",
        right_index=True,
        how="left",
    ).set_index("id")

    initial_num_rows = len(merged_data)
    merged_data = merged_data.dropna(subset=pred_pts_data.columns)
    final_num_rows = len(merged_data)
    percentage_dropped = (initial_num_rows - final_num_rows) / initial_num_rows * 100
    if percentage_dropped > 20:
        logger.warn(
            f"More than 20% of the rows were dropped. ({percentage_dropped:.2f}%)"
        )

    return merged_data


def get_pred_pts_data(inference_results, current_season, gameweeks):
    pred_pts_data = inference_results.loc[
        (inference_results["season"] == current_season)
        & inference_results["round"].isin(gameweeks)
    ]
    pred_pts_data = pred_pts_data.pivot_table(
        index="fpl_name", columns="round", values="prediction", aggfunc="first"
    ).fillna(0)
    pred_pts_data.columns = [f"xPts_{int(col)}" for col in pred_pts_data.columns]
    return pred_pts_data


def get_backtest_data(latest_elements_team, gw):
    backtest_data = pd.read_csv("data/raw/backtest_data/merged_gw.csv")[
        ["name", "position", "team", "xP", "GW", "value"]
    ]
    backtest_data = backtest_data[~backtest_data["GW"].isna()]
    backtest_data = backtest_data.sort_values(by="xP")
    backtest_data = backtest_data.drop_duplicates(subset=["name", "GW"])

    player_gw = pd.DataFrame(
        list(
            itertools.product(
                backtest_data["name"].unique(),
                backtest_data["GW"].unique(),
            )
        ),
        columns=["name", "GW"],
    )
    backtest_data = player_gw.merge(backtest_data, how="left", on=["name", "GW"])
    backtest_data["value"] = backtest_data.groupby("name")["value"].ffill()
    backtest_data["position"] = backtest_data.groupby("name")["position"].ffill()
    backtest_data["team"] = backtest_data.groupby("name")["team"].ffill()
    backtest_data["xP"] = backtest_data["xP"].fillna(0)
    gw_data = backtest_data.loc[backtest_data["GW"] == gw, :]
    elements_team = pd.merge(
        latest_elements_team,
        gw_data,
        left_on=["full_name", "name"],
        right_on=["name", "team"],
        suffixes=("", "_y"),
    )
    elements_team = (
        elements_team.drop(elements_team.filter(regex="_y$").columns, axis=1)
        .drop("GW", axis=1)
        .rename({"value": "now_cost"}, axis=1)
    )
    return elements_team


if __name__ == "__main__":
    fetch_player_fixtures(30, "2023-2024")
