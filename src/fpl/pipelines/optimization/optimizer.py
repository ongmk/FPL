import ast
import logging
import re
import time
from collections import defaultdict
from subprocess import Popen
from typing import Any

import numpy as np
import pandas as pd

from fpl.pipelines.optimization.data_classes import (
    TYPE_DATA,
    LpData,
    LpVariables,
    VariableSums,
)
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

    with open(f"{solution_path}", "r") as f:
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


def get_historical_picks(team_id, next_gw, merged_data):
    _, _, initial_squad = get_fpl_team_data(team_id, next_gw)
    picks_df = (
        pd.DataFrame(initial_squad)
        .drop("position", axis=1)
        .rename({"element": "id"}, axis=1)
    )
    picks_df["week"] = next_gw
    picks_df = picks_df.merge(
        merged_data[["web_name", f"pred_pts_{next_gw}"]],
        left_on="id",
        right_index=True,
    ).rename({f"pred_pts_{next_gw}": "predicted_xP"}, axis=1)
    summary = [picks_df]
    next_gw_dict = {
        "hits": 0,
        "in_the_bank": 0,
        "free_transfers": 0,
        "solve_time": 0,
        "n_transfers": 0,
        "chip_used": None,
    }
    return picks_df, summary, next_gw_dict


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


def merge_data(elements_team, points_data):
    merged_data = elements_team.merge(
        points_data,
        left_on="full_name",
        right_index=True,
        how="left",
    ).set_index("id")

    prediction_columns = [col for col in points_data if "pred_pts_" in col]
    merged_data[prediction_columns] = (
        merged_data[prediction_columns]
        .fillna(0)
        .mul(merged_data["chance_of_playing_next_round"].fillna(100) / 100, axis=0)
    )

    return merged_data


def aggregate_points_data(
    inference_results: pd.DataFrame, current_season: str, gameweeks: list[int]
):
    in_scope_data = inference_results.loc[
        (inference_results["season"] == current_season)
        & inference_results["round"].isin(gameweeks)
    ]
    pred_pts_data = in_scope_data.pivot_table(
        index="fpl_name", columns="round", values="prediction", aggfunc="first"
    ).fillna(0)
    pred_pts_data.columns = [f"pred_pts_{int(col)}" for col in pred_pts_data.columns]

    act_pts_data = in_scope_data.pivot_table(
        index="fpl_name", columns="round", values="fpl_points", aggfunc="first"
    ).fillna(0)
    act_pts_data.columns = [f"act_pts_{int(col)}" for col in act_pts_data.columns]

    mins_data = in_scope_data.pivot_table(
        index="fpl_name", columns="round", values="minutes", aggfunc="first"
    ).fillna(0)
    mins_data.columns = [f"mins_{int(col)}" for col in mins_data.columns]

    merged_data = pd.merge(
        pred_pts_data, act_pts_data, left_index=True, right_index=True, how="left"
    )
    merged_data = pd.merge(
        merged_data, mins_data, left_index=True, right_index=True, how="left"
    )
    return merged_data


def get_live_data(
    inference_results: pd.DataFrame, parameters: dict[str, Any]
) -> LpData:
    team_id = parameters["team_id"]
    horizon = parameters["horizon"]
    transfer_horizon = parameters["transfer_horizon"]

    elements_data, team_data, type_data, gameweeks_data, current_season = (
        get_fpl_base_data()
    )
    elements_data = elements_data.loc[
        ~elements_data["chance_of_playing_next_round"].isin([0, 25])
    ]
    gameweeks = gameweeks_data["future"][:horizon]
    current_gw = gameweeks_data["current"]

    general_data, transfer_data, initial_squad, history_data = get_fpl_team_data(
        team_id, current_gw
    )
    in_the_bank = (general_data["last_deadline_bank"] or 1000) / 10
    free_transfers = get_free_transfers(transfer_data, history_data, current_gw)

    merged_data = prepare_data(
        inference_results,
        elements_data,
        current_season,
        gameweeks,
        transfer_data,
        initial_squad,
        None,
    )

    logger.info(f"Team: {general_data['name']}.")
    logger.info(f"Current week: {current_gw}")
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
    )


def prepare_data(
    inference_results,
    elements_data,
    current_season,
    gameweeks,
    transfer_data,
    initial_squad,
    backup_fpl_data,
):
    points_data = aggregate_points_data(inference_results, current_season, gameweeks)
    merged_data = merge_data(elements_data, points_data)
    calculate_buy_sell_price(merged_data, transfer_data, initial_squad, backup_fpl_data)
    merged_data = sort_by_expected_value(merged_data)
    return merged_data


def sort_by_expected_value(merged_data):
    keys = [k for k in merged_data.columns.to_list() if "pred_pts_" in k]
    merged_data["total_ev"] = merged_data[keys].sum(axis=1)
    merged_data = merged_data.sort_values(by=["total_ev"], ascending=[False])
    return merged_data


def fpl_data_to_elements_data(fpl_data):
    elements_data = fpl_data[
        ["element", "full_name", "team_name", "position", "value"]
    ].drop_duplicates()
    elements_data["web_name"] = elements_data["full_name"]
    elements_data["element_type"] = elements_data["position"].map(
        {d["singular_name_short"]: d["id"] for d in TYPE_DATA}
    )
    elements_data["value"] = elements_data["value"].astype(int)
    elements_data = elements_data.rename(
        columns={"element": "id", "team_name": "team", "value": "now_cost"}
    )
    elements_data["chance_of_playing_next_round"] = 100
    return elements_data


def calculate_buy_sell_price(
    merged_data, transfer_data, initial_squad, backup_fpl_data
):
    merged_data["sell_price"] = merged_data["now_cost"]
    for t in transfer_data:
        if t["element_in"] not in initial_squad:
            continue
        bought_price = t["element_in_cost"]
        if t["element_in"] not in merged_data.index:
            backup_elements_data = fpl_data_to_elements_data(backup_fpl_data).set_index(
                "id"
            )
            merged_data.loc[t["element_in"]] = backup_elements_data.loc[t["element_in"]]
            merged_data.loc[t["element_in"]] = merged_data.loc[t["element_in"]].fillna(
                0
            )
        merged_data.loc[t["element_in"], "bought_price"] = bought_price
        current_price = merged_data.loc[t["element_in"], "now_cost"]
        if current_price <= bought_price:
            continue
        sell_price = np.floor(np.mean([current_price, bought_price]))
        merged_data.loc[t["element_in"], "sell_price"] = sell_price
    return None


if __name__ == "__main__":
    pass
