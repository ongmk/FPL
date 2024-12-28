import ast
import logging
import re
import time
from collections import defaultdict
from functools import reduce
from subprocess import Popen
from typing import Any

import numpy as np
import pandas as pd
from thefuzz import process

from fpl.pipelines.optimization.chips_suggestion import get_chips_suggestions
from fpl.pipelines.optimization.data_classes import LpData, LpVariables, VariableSums
from fpl.pipelines.optimization.fpl_api import get_fpl_base_data, get_fpl_team_data
from fpl.utils import backup_latest_n

logger = logging.getLogger(__name__)


def split_name_key(text):
    pattern = r"_[1234567890(]"
    match = re.search(pattern, text)
    key_idx = match.start()
    name = text[:key_idx]
    key_part = text[key_idx:].replace("_", "")
    key = ast.literal_eval(key_part)
    return name, key


def raise_exceptions(line):
    if "Infeasible" in line:
        logger.error(line)
        # Troubleshooting with lp_data
        raise ValueError("No solution found.")
    elif "no integer solution" in line:
        logger.error(line)
        raise ValueError("No integer solution found.")
    elif "Stopped on time" in line:
        logger.warning(line)
    elif "Optimal" in line:
        logger.info(line)
    else:
        return None


def solve_lp(
    lp_data: LpData,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
    parameters: dict,
) -> float:
    model_name = parameters["model_name"]
    mps_dir = parameters["mps_dir"]
    mps_path = f"{mps_dir}/{model_name}.mps"
    solution_dir = parameters["solution_dir"]
    solution_path = f"{solution_dir}/{model_name}.sol"
    init_feasible_solution_path = f"{solution_path}.init"
    gap = 0.01
    timeout = 60

    start = time.time()
    command = f"cbc/bin/cbc.exe {mps_path} cost column ratio 1 solve solu {init_feasible_solution_path}"
    process = Popen(command, shell=False)
    process.wait()

    command = f"cbc/bin/cbc.exe {mps_path} mips {init_feasible_solution_path} cost column ratio {gap} sec {timeout} solve solu {solution_path}"
    process = Popen(command, shell=False)
    process.wait()
    solution_time = time.time() - start
    logger.info(f"Solved in {solution_time:.1f} seconds.")

    backup_latest_n(solution_path, 5)

    for name, variable in lp_variables.__dict__.items():
        for key in variable:
            variable[key].varValue = 0

    with open(solution_path, "r") as f:
        for line in f:
            if "objective value" in line:
                raise_exceptions(line)
                continue
            words = line.split()
            variable, value = words[1], float(words[2])
            name, key = split_name_key(variable)
            variable_dict = getattr(lp_variables, name, None)
            if variable_dict is None:
                continue
            variable_dict[key].varValue = value
    return lp_variables, variable_sums, solution_time


def get_free_transfers(
    transfer_data: dict, chips_history: list[dict[str, Any]], current_gameweek: int
) -> int:
    free_transfers = 1
    if current_gameweek <= 2:
        return free_transfers

    transfer_count = defaultdict(int)

    for transfer in transfer_data:
        transfer_count[transfer["event"]] += 1
    for chip in chips_history:
        if chip["name"] in ["wildcard", "freehit"]:
            transfer_count[chip["event"]] = 1

    # F2 = min(max(F1-T+1, 1), 5)
    for week in range(2, current_gameweek):
        free_transfers = min(max(free_transfers - transfer_count[week] + 1, 1), 5)
    return free_transfers


def pivot_column(source_data, column_name, prefix):
    output_data = source_data.pivot_table(
        index="fpl_name", columns="round", values=column_name, aggfunc="first"
    ).fillna(0)
    output_data.columns = [f"{prefix}{int(col)}" for col in output_data.columns]
    return output_data


def aggregate_points_data(
    inference_results: pd.DataFrame, current_season: str, gameweeks: list[int]
):
    in_scope_data = inference_results.loc[
        (inference_results["season"] == current_season)
        & inference_results["round"].isin(gameweeks)
    ]

    pred_pts_data = pivot_column(in_scope_data, "predicted_points", "pred_pts_")
    act_pts_data = pivot_column(in_scope_data, "fpl_points", "act_pts_")
    mins_data = pivot_column(in_scope_data, "minutes", "mins_")

    merged_data = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="left"
        ),
        [pred_pts_data, act_pts_data, mins_data],
    )
    return merged_data


def combine_inference_results(
    elements_data,
    inference_results,
    dnp_inference_results,
    current_gameweek,
):
    dnp_inference_results = process_dnp_data(dnp_inference_results, current_gameweek)
    merged_data = pd.merge(
        elements_data,
        dnp_inference_results,
        left_index=True,
        right_on="element",
        how="left",
        suffixes=("", "_dnp"),
    )
    merged_data["chance_of_playing_next_round"] = (
        merged_data["chance_of_playing_next_round"].fillna(100) / 100
    )
    merged_data["prediction"] = 1 - merged_data["prediction"].fillna(1)

    merged_data = pd.merge(
        merged_data, inference_results, on="element", how="left", suffixes=("_dnp", "")
    )
    merged_data.loc[
        (merged_data["status"] == "s") & (merged_data["round"] > current_gameweek),
        "chance_of_playing_next_round",
    ] = 1.0

    merged_data["chance_of_playing_next_round"] = merged_data[
        ["prediction_dnp", "chance_of_playing_next_round"]
    ].min(axis=1)

    merged_data["predicted_points"] = (
        merged_data["prediction"].fillna(0)
        * merged_data["chance_of_playing_next_round"]
    )
    new_players = merged_data.loc[merged_data["round"].isna(), "web_name"].tolist()
    merged_data = merged_data.dropna(subset=["round"])
    logger.info(f"{new_players} are dropped from optimization as they are new players.")
    merged_data["round"] = merged_data["round"].astype(int)
    return merged_data


def fuzzy_match_elements(input_name, merged_data):
    webname, _ = process.extractOne(input_name, merged_data["web_name"].tolist())
    return merged_data.index[merged_data["web_name"] == webname].values[0]


def get_forced_transfers(merged_data, parameters):
    parameters = parameters["force_transfers"]

    in_elements = []
    if parameters["buy"] is not None:
        for input_name, buy_price in parameters["buy"].items():
            element = fuzzy_match_elements(input_name, merged_data)
            in_elements.append(element)
            if buy_price is not None:
                merged_data.loc[element, "bought_price"] = buy_price

    out_elements = []
    if parameters["sell"] is not None:
        for input_name, sell_price in parameters["sell"].items():
            element = fuzzy_match_elements(input_name, merged_data)
            out_elements.append(element)
            if sell_price is not None:
                merged_data.loc[element, "bought_price"] = sell_price

    keep_elements = []
    if parameters["keep"] is not None:
        for input_name in parameters["keep"]:
            element = fuzzy_match_elements(input_name, merged_data)
            keep_elements.append(element)

    forced_transfers = {
        "in": in_elements,
        "out": out_elements,
        "keep": keep_elements,
    }

    return forced_transfers, merged_data


def get_live_data(
    inference_results: pd.DataFrame,
    dnp_inference_results: pd.DataFrame,
    fpl_2_fbref_team_mapping: dict[str, str],
    parameters: dict[str, Any],
) -> LpData:
    team_id = parameters["team_id"]
    horizon = parameters["horizon"]
    transfer_horizon = parameters["transfer_horizon"]

    elements_data, team_data, type_data, gameweeks_data, current_season = (
        get_fpl_base_data()
    )
    elements_data["team"] = elements_data["team"].map(fpl_2_fbref_team_mapping)
    team_data["name"] = team_data["name"].map(fpl_2_fbref_team_mapping)
    gameweeks = gameweeks_data["future"][:horizon]
    current_gw = gameweeks_data["current"]

    general_data, transfer_data, initial_squad, history_data = get_fpl_team_data(
        team_id, current_gw
    )
    in_the_bank = general_data["last_deadline_bank"] / 10
    in_the_bank = 1000 if in_the_bank is None else in_the_bank
    free_transfers = get_free_transfers(
        transfer_data, history_data["chips"], gameweeks[0]
    )

    inference_results = inference_results.loc[
        inference_results["season"] == current_season
    ]
    dnp_inference_results = dnp_inference_results.loc[
        dnp_inference_results["season"] == current_season
    ]

    if current_gw > 1:
        assert (
            inference_results.loc[
                (inference_results["round"] == current_gw), "fpl_points"
            ]
            .isna()
            .sum()
            == 0,
            "Missing data for previous gameweek.",
        )

    inference_results = combine_inference_results(
        elements_data,
        inference_results,
        dnp_inference_results,
        gameweeks[0],
    )

    merged_data = prepare_data(
        inference_results,
        elements_data,
        current_season,
        gameweeks,
        transfer_data,
        initial_squad,
    )

    forced_transfers, merged_data = get_forced_transfers(merged_data, parameters)

    chips_usage = get_chips_suggestions(
        inference_results,
        initial_squad,
        history_data["chips"],
        gameweeks[0],
        parameters,
    )

    logger.info("=" * 50)
    logger.info(f"Team: {general_data['name']}.")
    logger.info(f"Current week: {current_gw}")
    logger.info(f"In the bank: {in_the_bank}\tFree transfers: {free_transfers}")
    logger.info("=" * 50)

    logger.info(
        f"Optimizing for {horizon} weeks. {gameweeks}. Making transfers for {transfer_horizon} weeks."
    )

    return LpData(
        merged_data=merged_data,
        team_data=team_data,
        type_data=type_data,
        gameweeks=gameweeks,
        initial_squad=initial_squad,
        team_name=general_data["name"],
        in_the_bank=in_the_bank,
        free_transfers=free_transfers,
        current_season=current_season,
        chips_usage=chips_usage,
        forced_transfers=forced_transfers,
    )


def process_dnp_data(data: pd.DataFrame, current_round: int) -> pd.DataFrame:
    data = data.loc[(data["round"] <= current_round)]
    data = data.loc[
        data["round"] == data["round"].max(),
        [
            "team",
            "element",
            "fpl_name",
            "prediction",
        ],
    ]

    return data


def prepare_data(
    inference_results,
    elements_data,
    current_season,
    gameweeks,
    transfer_data,
    initial_squad,
):
    agg_data = aggregate_points_data(inference_results, current_season, gameweeks)
    merged_data = elements_data.merge(
        agg_data,
        left_on="full_name",
        right_index=True,
        how="left",
    )
    pred_pts_cols = [f"pred_pts_{gw}" for gw in gameweeks]
    merged_data[pred_pts_cols] = merged_data[pred_pts_cols].fillna(0)
    calculate_buy_sell_price(merged_data, transfer_data, initial_squad, elements_data)
    merged_data = sort_by_expected_value(merged_data)
    return merged_data


def sort_by_expected_value(merged_data):
    keys = [k for k in merged_data.columns.to_list() if "pred_pts_" in k]
    merged_data["total_ev"] = merged_data[keys].sum(axis=1)
    merged_data = merged_data.sort_values(by=["total_ev"], ascending=[False])
    return merged_data


def get_sell_price(row):
    if row["now_cost"] <= row["bought_price"]:
        return row["now_cost"]
    else:
        return int(np.floor(np.mean([row["now_cost"], row["bought_price"]])))


def calculate_buy_sell_price(merged_data, transfer_data, initial_squad, elements_data):
    initial_costs = (
        elements_data.loc[initial_squad, "now_cost"]
        - elements_data.loc[initial_squad, "cost_change_start"]
    )
    merged_data["sell_price"] = merged_data["now_cost"]
    merged_data["bought_price"] = np.nan
    for t in transfer_data:
        if t["element_in"] not in initial_squad:
            continue
        bought_price = t["element_in_cost"]
        if t["element_in"] not in merged_data.index:
            merged_data.loc[t["element_in"]] = elements_data.loc[
                t["element_in"], ["full_name", "team_name", "position", "value"]
            ]
            merged_data.loc[t["element_in"]] = merged_data.loc[t["element_in"]].fillna(
                0
            )
        merged_data.loc[t["element_in"], "bought_price"] = bought_price
    pre_season_squad = (
        merged_data.index.isin(initial_squad) & merged_data["bought_price"].isna()
    )
    merged_data.loc[pre_season_squad, "bought_price"] = merged_data.loc[
        pre_season_squad
    ].index.map(initial_costs)

    merged_data.loc[initial_squad, "sell_price"] = merged_data.loc[initial_squad].apply(
        get_sell_price, axis=1
    )

    return None


if __name__ == "__main__":
    transfer_data = [
        {
            "element_in": 277,
            "element_in_cost": 45,
            "element_out": 103,
            "element_out_cost": 45,
            "entry": 1792016,
            "event": 3,
            "time": "2024-08-31T08:25:19.568041Z",
        },
        {
            "element_in": 347,
            "element_in_cost": 55,
            "element_out": 554,
            "element_out_cost": 45,
            "entry": 1792016,
            "event": 4,
            "time": "2024-09-05T01:22:37.163320Z",
        },
        {
            "element_in": 342,
            "element_in_cost": 66,
            "element_out": 348,
            "element_out_cost": 94,
            "entry": 1792016,
            "event": 4,
            "time": "2024-09-05T01:22:37.168334Z",
        },
        {
            "element_in": 339,
            "element_in_cost": 60,
            "element_out": 333,
            "element_out_cost": 43,
            "entry": 1792016,
            "event": 4,
            "time": "2024-09-13T02:01:38.746909Z",
        },
        {
            "element_in": 148,
            "element_in_cost": 57,
            "element_out": 401,
            "element_out_cost": 84,
            "entry": 1792016,
            "event": 5,
            "time": "2024-09-17T16:16:46.250218Z",
        },
        {
            "element_in": 91,
            "element_in_cost": 45,
            "element_out": 414,
            "element_out_cost": 40,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.251677Z",
        },
        {
            "element_in": 18,
            "element_in_cost": 60,
            "element_out": 211,
            "element_out_cost": 49,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.252019Z",
        },
        {
            "element_in": 3,
            "element_in_cost": 61,
            "element_out": 335,
            "element_out_cost": 60,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.252462Z",
        },
        {
            "element_in": 594,
            "element_in_cost": 46,
            "element_out": 408,
            "element_out_cost": 40,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.252809Z",
        },
        {
            "element_in": 327,
            "element_in_cost": 79,
            "element_out": 13,
            "element_out_cost": 82,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.253087Z",
        },
        {
            "element_in": 99,
            "element_in_cost": 72,
            "element_out": 199,
            "element_out_cost": 69,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.253332Z",
        },
        {
            "element_in": 432,
            "element_in_cost": 53,
            "element_out": 277,
            "element_out_cost": 45,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.253568Z",
        },
        {
            "element_in": 434,
            "element_in_cost": 54,
            "element_out": 328,
            "element_out_cost": 126,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.253839Z",
        },
        {
            "element_in": 94,
            "element_in_cost": 49,
            "element_out": 342,
            "element_out_cost": 65,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.254060Z",
        },
        {
            "element_in": 351,
            "element_in_cost": 153,
            "element_out": 148,
            "element_out_cost": 57,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.254303Z",
        },
        {
            "element_in": 232,
            "element_in_cost": 54,
            "element_out": 207,
            "element_out_cost": 74,
            "entry": 1792016,
            "event": 6,
            "time": "2024-09-26T12:09:33.254538Z",
        },
        {
            "element_in": 498,
            "element_in_cost": 51,
            "element_out": 594,
            "element_out_cost": 46,
            "entry": 1792016,
            "event": 8,
            "time": "2024-10-18T06:39:41.336418Z",
        },
        {
            "element_in": 447,
            "element_in_cost": 62,
            "element_out": 4,
            "element_out_cost": 81,
            "entry": 1792016,
            "event": 8,
            "time": "2024-10-18T06:39:41.342586Z",
        },
        {
            "element_in": 326,
            "element_in_cost": 53,
            "element_out": 3,
            "element_out_cost": 62,
            "entry": 1792016,
            "event": 10,
            "time": "2024-11-01T00:53:40.046509Z",
        },
        {
            "element_in": 180,
            "element_in_cost": 79,
            "element_out": 447,
            "element_out_cost": 63,
            "entry": 1792016,
            "event": 10,
            "time": "2024-11-01T00:53:40.052804Z",
        },
        {
            "element_in": 328,
            "element_in_cost": 131,
            "element_out": 327,
            "element_out_cost": 76,
            "entry": 1792016,
            "event": 13,
            "time": "2024-11-29T02:01:54.203457Z",
        },
        {
            "element_in": 401,
            "element_in_cost": 85,
            "element_out": 351,
            "element_out_cost": 151,
            "entry": 1792016,
            "event": 13,
            "time": "2024-11-29T02:01:54.209803Z",
        },
        {
            "element_in": 52,
            "element_in_cost": 44,
            "element_out": 326,
            "element_out_cost": 53,
            "entry": 1792016,
            "event": 14,
            "time": "2024-12-02T03:02:04.604030Z",
        },
        {
            "element_in": 17,
            "element_in_cost": 103,
            "element_out": 434,
            "element_out_cost": 54,
            "entry": 1792016,
            "event": 14,
            "time": "2024-12-02T03:02:04.610327Z",
        },
        {
            "element_in": 541,
            "element_in_cost": 71,
            "element_out": 401,
            "element_out_cost": 85,
            "entry": 1792016,
            "event": 14,
            "time": "2024-12-02T03:02:04.610711Z",
        },
    ]
    chips_history = [
        {"name": "wildcard", "time": "2024-09-26T12:09:22.708252Z", "event": 6},
        {"name": "bboost", "time": "2024-10-18T06:40:13.874045Z", "event": 8},
    ]
    gameweeks = [16, 17, 18, 19]
    ft = get_free_transfers(transfer_data, chips_history, gameweeks[0])
    print(ft)
