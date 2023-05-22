import logging
import requests
import pandas as pd
import itertools
from src.fpl.pipelines.optimization_pipeline.fetch_predictions import (
    get_pred_pts_data,
    resolve_fpl_names,
)
import numpy as np
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def get_fpl_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
    elements_team = pd.merge(
        element_data, team_data, left_on="team", right_index=True, suffixes=("", "_y")
    )
    elements_team = elements_team.drop(
        elements_team.filter(regex="_y$").columns, axis=1
    )
    elements_team["full_name"] = elements_team["first_name"].str.cat(
        elements_team["second_name"], sep=" "
    )
    elements_team = elements_team[
        [
            "web_name",
            "team",
            "element_type",
            "name",
            "full_name",
            "short_name",
            "now_cost",
            "id",
        ]
    ]

    type_data = pd.DataFrame(fpl_data["element_types"]).set_index(["id"])

    return elements_team, team_data, type_data, all_gws


@dataclass
class FplData:
    merged_data: pd.DataFrame
    team_data: pd.DataFrame
    type_data: pd.DataFrame
    gameweeks: list[int]
    initial_squad: pd.DataFrame
    itb: float


def get_live_data(team_id: int, horizon: int) -> FplData:
    elements_team, team_data, type_data, all_gws = get_fpl_base_data()
    gameweeks = all_gws["future"][:horizon]
    current_gw = all_gws["current"]

    pred_pts_data = get_pred_pts_data(gameweeks)
    pred_pts_data = resolve_fpl_names(
        pred_pts_data, elements_team[["web_name", "short_name"]]
    )

    merged_data = elements_team.merge(
        pred_pts_data,
        left_on=["web_name", "short_name"],
        right_on=["FPL name", "Team"],
        how="left",
    ).set_index("id")

    r = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/")
    general_data = r.json()
    itb = general_data["last_deadline_bank"] / 10
    initial_squad = get_initial_squad(team_id, current_gw)
    initial_squad = [p["element"] for p in initial_squad]
    r = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
    )
    transfer_data = reversed(r.json())
    merged_data["sell_price"] = merged_data["now_cost"]
    for t in transfer_data:
        if t["element_in"] in initial_squad:
            bought_price = t["element_in_cost"]
            merged_data.loc[t["element_in"], "bought_price"] = bought_price
            current_price = merged_data.loc[t["element_in"], "now_cost"]
            if current_price > bought_price:
                sell_price = np.ceil(np.mean([current_price, bought_price]))
                merged_data.loc[t["element_in"], "sell_price"] = sell_price

    merged_data = merged_data.dropna(
        subset=["FPL name", "Team", "Pos", "Price", "bought_price"], how="all"
    )
    xPts_cols = [
        column for column in merged_data.columns if re.match(r"xPts_\d+", column)
    ]
    merged_data[xPts_cols] = merged_data[xPts_cols].fillna(0)

    logger.info("=" * 50)
    logger.info(f"Team: {general_data['name']}. Current week: {current_gw}")
    logger.info(f"Optimizing for {horizon} weeks. {gameweeks}")
    logger.info("=" * 50 + "\n")

    return FplData(
        merged_data=merged_data,
        team_data=team_data,
        type_data=type_data,
        gameweeks=gameweeks,
        initial_squad=initial_squad,
        itb=itb,
    )


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
    elements_team = latest_elements_team.merge(
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


def get_initial_squad(team_id: int, gw: int) -> list[dict]:
    r = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    )
    picks_data = r.json()
    return picks_data["picks"]
