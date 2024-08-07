import ast
import logging
import re
import time
from subprocess import Popen

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from fpl.pipelines.optimization.fpl_api import get_backtest_data, get_fpl_team_data
from fpl.pipelines.optimization.lp_constructor import LpVariables, VariableSums
from fpl.utils import backup_latest_n

plt.style.use("ggplot")
matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def split_name_key(text):
    pattern = r"_[1234567890(]"
    match = re.search(pattern, text)
    key_idx = match.start()
    name = text[:key_idx]
    key_part = text[key_idx:].replace("_", "")
    key = ast.literal_eval(key_part)
    return name, key


def solve_lp(
    lp_variables: LpVariables, variable_sums: VariableSums, parameters: dict
) -> float:

    mps_path = parameters["mps_path"]
    solution_path = parameters["solution_path"]
    solution_init_path = f"init_{solution_path}"

    start = time.time()
    command = f"cbc/bin/cbc.exe {mps_path} cost column ratio 1 solve solu {solution_init_path}"
    process = Popen(command, shell=False)
    process.wait()
    command = f"cbc/bin/cbc.exe {mps_path} mips {solution_init_path} cost column solve solu {solution_path}"
    process = Popen(command, shell=False)
    process.wait()
    solution_time = time.time() - start
    print(f"Solved in {solution_time:.1f} seconds.")

    backup_latest_n(solution_path, 5)

    for name, variable in lp_variables.__dict__.items():
        for key in variable:
            variable[key].varValue = 0

    with open(f"{solution_path}", "r") as f:
        for line in f:
            if "objective value" in line:
                if "Infeasible" in line:
                    logger.error(line)
                    raise ValueError("No solution found.")
                else:
                    logger.info(line)
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
        merged_data[["web_name", f"xPts_{next_gw}"]],
        left_on="id",
        right_index=True,
    ).rename({f"xPts_{next_gw}": "predicted_xP"}, axis=1)
    summary = [picks_df]
    next_gw_dict = {
        "hits": 0,
        "itb": 0,
        "remaining_free_transfers": 0,
        "solve_time": 0,
        "n_transfers": 0,
        "chip_used": None,
    }
    return picks_df, summary, next_gw_dict


def backtest_single_player(parameters: dict, title: str = "Backtest Result"):
    # Arguments
    horizon = parameters["horizon"]
    team_id = parameters["team_id"]
    backtest_player_history = parameters["backtest_player_history"]

    # Pre season
    latest_elements_team, team_data, type_data, all_gws = 0  # TODO use historical data
    latest_elements_team = latest_elements_team.drop("now_cost", axis=1)
    itb = 100
    initial_squad = []
    parameters["remaining_free_transfers"] = 1
    total_predicted_xp = 0
    total_xp = 0

    result_gw = []
    result_xp = []
    result_predicted_xp = []
    result_solve_times = []

    for next_gw in tqdm(all_gws["finished"]):
        gameweeks = [i for i in range(next_gw, next_gw + horizon)]
        logger.info(80 * "=")
        logger.info(
            f"Backtesting GW {next_gw}. ITB = {itb:.1f}. remaining_free_transfers = {parameters['remaining_free_transfers']}. {gameweeks}"
        )
        logger.info(80 * "=")
        elements_team = get_backtest_data(latest_elements_team, next_gw)
        if elements_team.empty:
            logger.warning(f"No data from GW {next_gw}")
        else:
            pred_pts_data = 0  # TODO: fix backtest
            for p in initial_squad:
                name = elements_team.loc[elements_team["id"] == p, "web_name"].item()
                team = elements_team.loc[elements_team["id"] == p, "short_name"].item()
                if pred_pts_data.loc[
                    (pred_pts_data["FPL name"] == name)
                    & (pred_pts_data["Team"] == team)
                ].empty:
                    pred_pts_data.loc[len(pred_pts_data)] = [name, team, None, None] + [
                        0 for i in range(len(pred_pts_data.columns) - 4)
                    ]
            merged_data = elements_team.merge(
                pred_pts_data,
                left_on=["web_name", "short_name"],
                right_on=["FPL name", "Team"],
            ).set_index("id")
            merged_data["sell_price"] = merged_data["now_cost"]
            # merged_data[f"xPts_{next_gw}"] = merged_data["xP"] # peek actual xp

            if backtest_player_history:
                picks_df, summary, next_gw_dict = get_historical_picks(
                    team_id, next_gw, merged_data
                )

            else:
                picks_df, summary, next_gw_dict = solve_lp(
                    {
                        "merged_data": merged_data,
                        "team_data": team_data,
                        "type_data": type_data,
                        "gameweeks": gameweeks,
                        "initial_squad": initial_squad,
                        "itb": itb,
                    },
                    parameters,
                )

            logger.info(summary[0])
            # picks_df.to_csv("picks.csv", index=False, encoding="utf-8-sig")

            squad = picks_df.loc[picks_df["week"] == next_gw]
            squad = squad.merge(
                merged_data[["xP"]], left_on="id", right_index=True
            ).set_index("id")
            predicted_xp = (
                squad["predicted_xP"] * squad["multiplier"]
            ).sum() - next_gw_dict["hits"] * 4
            actual_xp = (squad["xP"] * squad["multiplier"]).sum() - next_gw_dict[
                "hits"
            ] * 4
            total_predicted_xp += predicted_xp
            total_xp += actual_xp
            logger.info(
                f"Predicted xP = {predicted_xp:.2f}. ({total_predicted_xp:.2f} overall)"
            )
            logger.info(f"Actual xP = {actual_xp:.2f}. ({total_xp:.2f} overall)")

            if not backtest_player_history:
                if next_gw_dict["chip_used"] not in ("fh", "wc") and next_gw != 1:
                    assert next_gw_dict["remaining_free_transfers"] == min(
                        2,
                        max(
                            1,
                            parameters["remaining_free_transfers"]
                            - next_gw_dict["n_transfers"]
                            + 1,
                        ),
                    )
                else:
                    assert (
                        next_gw_dict["remaining_free_transfers"]
                        == parameters["remaining_free_transfers"]
                    )

            itb = next_gw_dict["itb"]
            parameters["remaining_free_transfers"] = next_gw_dict[
                "remaining_free_transfers"
            ]
            initial_squad = squad.index.to_list()

        result_gw.append(next_gw)
        result_xp.append(total_xp)
        result_predicted_xp.append(total_predicted_xp)
        result_solve_times.append(next_gw_dict["solve_time"])
        logger.info(
            f"Avg solve time: {sum(result_solve_times) / len(result_solve_times):.1f}"
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(result_gw, result_xp, label="Actual xP")
    for index in range(len(result_gw)):
        ax.text(result_gw[index], result_xp[index], f"{result_xp[index]:.1f}", size=12)
    for index in range(len(result_gw)):
        ax.text(
            result_gw[index],
            result_predicted_xp[index],
            f"{result_predicted_xp[index]:.1f}",
            size=12,
        )

    plt.plot(result_gw, result_predicted_xp, linewidth=2.0, label="Predicted xP")
    plt.title(title)
    plt.legend()
    filename = f"[{total_predicted_xp:.1f},{total_xp:.1f}]{title}.png"
    return filename, fig


if __name__ == "__main__":
    pass
