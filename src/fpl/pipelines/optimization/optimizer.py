import ast
import logging
import re
import time
from subprocess import Popen

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pulp import LpProblem, lpSum
from tqdm import tqdm

from fpl.pipelines.optimization.fpl_api import (
    FplData,
    get_backtest_data,
    get_fpl_team_data,
)
from fpl.pipelines.optimization.lp_constructor import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.utils import backup_latest_n

plt.style.use("ggplot")
matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def get_name(row):
    name = ""
    if row["captain"] == 1:
        name += "[c] "
    elif row["vicecaptain"] == 1:
        name += "[v] "
    name += f"{row['name']} {row['predicted_xP']}"
    return name


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

    _, model = LpProblem.fromMPS(mps_path)

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
            getattr(lp_variables, name)[key].varValue = value
    return lp_variables, variable_sums, solution_time


def generate_picks_df(
    fpl_data: FplData,
    lp_params: LpParams,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
) -> pd.DataFrame:
    picks = []
    for w in fpl_data.gameweeks:
        for p in lp_keys.players:
            if (
                lp_variables.squad[p, w].value()
                + lp_variables.squad_free_hit[p, w].value()
                + lp_variables.transfer_out[p, w].value()
                > 0.5
            ):
                player_data = fpl_data.merged_data.loc[p]
                is_captain = 1 if lp_variables.captain[p, w].value() > 0.5 else 0
                is_squad = (
                    1
                    if (
                        lp_variables.use_free_hit[w].value() < 0.5
                        and lp_variables.squad[p, w].value() > 0.5
                    )
                    or (
                        lp_variables.use_free_hit[w].value() > 0.5
                        and lp_variables.squad_free_hit[p, w].value() > 0.5
                    )
                    else 0
                )
                is_lineup = 1 if lp_variables.lineup[p, w].value() > 0.5 else 0
                is_vice = 1 if lp_variables.vicecap[p, w].value() > 0.5 else 0
                is_transfer_in = (
                    1 if lp_variables.transfer_in[p, w].value() > 0.5 else 0
                )
                is_transfer_out = (
                    1 if lp_variables.transfer_out[p, w].value() > 0.5 else 0
                )
                bench_value = -1
                for o in lp_keys.order:
                    if lp_variables.bench[p, w, o].value() > 0.5:
                        bench_value = o
                player_buy_price = 0 if not is_transfer_in else lp_keys.buy_price[p]
                player_sell_price = (
                    0
                    if not is_transfer_out
                    else (
                        lp_keys.sell_price[p]
                        if p in lp_keys.price_modified_players
                        and lp_variables.transfer_out_first[p, w].value() > 0.5
                        else lp_keys.buy_price[p]
                    )
                )
                multiplier = 1 * (is_lineup == 1) + 1 * (is_captain == 1)
                xp_cont = variable_sums.points_player_week[p, w] * multiplier
                position = fpl_data.type_data.loc[
                    player_data["element_type"], "singular_name_short"
                ]
                picks.append(
                    {
                        "week": w,
                        "id": p,
                        "name": player_data["web_name"],
                        "pos": position,
                        "type": player_data["element_type"],
                        "team": player_data["team"],
                        "buy_price": player_buy_price,
                        "sell_price": player_sell_price,
                        "predicted_xP": round(
                            variable_sums.points_player_week[p, w], 2
                        ),
                        "squad": is_squad,
                        "lineup": is_lineup,
                        "bench": bench_value,
                        "captain": is_captain,
                        "vicecaptain": is_vice,
                        "transfer_in": is_transfer_in,
                        "transfer_out": is_transfer_out,
                        "multiplier": multiplier,
                        "xp_cont": xp_cont,
                    }
                )
    picks_df = pd.DataFrame(
        picks,
    ).sort_values(
        by=["week", "lineup", "type", "predicted_xP"],
        ascending=[True, False, True, False],
    )
    return picks_df


def generate_summary(
    fpl_data: FplData,
    lp_params: LpParams,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
    picks_df: pd.DataFrame,
    parameters: dict,
    solution_time: float,
) -> tuple[list[str], dict]:
    summary = []
    total_xp = 0
    for w in fpl_data.gameweeks:
        header = f" GW {w} "
        transfer_summary = get_transfer_summary(
            fpl_data, lp_keys, lp_variables, variable_sums, w
        )
        chip_summary = get_chip_summary(lp_variables, w)
        lineup = get_lineup(picks_df, w)

        gw_xp = (
            lpSum(
                [
                    (lp_variables.lineup[p, w] + lp_variables.captain[p, w])
                    * variable_sums.points_player_week[p, w]
                    for p in lp_keys.players
                ]
            ).value()
            - lp_variables.penalized_transfers[w].value() * 4
        )
        total_xp += gw_xp
        hits = int(lp_variables.penalized_transfers[w].value())
        hit_str = f"({hits} hits)" if hits > 0 else ""
        gw_summary = (
            f"{header:{'*'}^80}\n\n"
            f"{chip_summary}"
            f"{transfer_summary}"
            f"{lineup}\n\n"
            f"Gameweek xP = {gw_xp:.2f} {hit_str}\n"
        )
        summary.append(gw_summary)
        if w == lp_params.next_gw:
            if lp_variables.use_wildcard[w].value() > 0.5:
                chip_used = "wildcard"
            elif lp_variables.use_free_hit[w].value() > 0.5:
                chip_used = "free hit"
            elif lp_variables.use_bench_boost[w].value() > 0.5:
                chip_used = "bench boost"
            else:
                chip_used = None
            next_gw_dict = {
                "in_the_bank": round(lp_variables.in_the_bank[w].value(), 1),
                "remaining_free_transfers": (
                    round(lp_variables.free_transfers[w + 1].value())
                    if w + 1 <= 38
                    else 0
                ),
                "hits": round(lp_variables.penalized_transfers[w].value()),
                "solve_time": solution_time,
                "n_transfers": round(
                    lpSum(
                        [lp_variables.transfer_out[p, w] for p in lp_keys.players]
                    ).value()
                ),
                "chip_used": chip_used,
            }
    overall_summary = (
        "=" * 80 + "\n" + f"{parameters['horizon']} weeks total xP = {total_xp:.2f}"
    )
    summary.append(overall_summary)
    summary = "\n".join(summary)
    logger.info(summary)
    return summary, next_gw_dict


def get_lineup(picks_df, w):
    gw_squad = picks_df.loc[picks_df["week"] == w, :].copy()
    gw_squad.loc[:, "name"] = gw_squad.agg(lambda x: get_name(x), axis=1)
    gw_lineup = gw_squad.loc[gw_squad["lineup"] == 1]
    lineup_str = []
    for p_type in [1, 2, 3, 4]:
        lineup_str.append(
            (gw_lineup.loc[gw_lineup["type"] == p_type, "name"]).str.cat(sep="    ")
        )
    lineup_str.append("")
    gw_bench = gw_squad.loc[gw_squad["bench"] != -1].sort_values("bench")
    lineup_str.append(f'Bench: {gw_bench["name"].str.cat(sep="    ")}')
    length = max([len(s) for s in lineup_str])
    lineup_str = [f"{s:^{length}}" for s in lineup_str]
    lineup_str = "\n".join(lineup_str)
    return lineup_str


def get_transfer_summary(fpl_data, lp_keys, lp_variables, variable_sums, w):
    gw_in = pd.DataFrame([], columns=["", "In", "xP", "Pos"])
    gw_out = pd.DataFrame([], columns=["Out", "xP", "Pos"])
    net_cost = 0
    net_xp = 0
    for p in lp_keys.players:
        if lp_variables.transfer_in[p, w].value() > 0.5:
            price = fpl_data.merged_data["now_cost"][p] / 10
            name = f'{fpl_data.merged_data["web_name"][p]} ({price})'
            pos = fpl_data.merged_data["element_type"][p]
            xp = round(variable_sums.points_player_week[p, w], 2)
            net_cost += price
            net_xp += xp
            gw_in.loc[len(gw_in)] = ["👉", name, xp, pos]
        if lp_variables.transfer_out[p, w].value() > 0.5:
            price = fpl_data.merged_data["sell_price"][p] / 10
            name = f'{fpl_data.merged_data["web_name"][p]} ({price})'
            pos = fpl_data.merged_data["element_type"][p]
            xp = round(variable_sums.points_player_week[p, w], 2)
            net_cost -= price
            net_xp -= xp
            gw_out.loc[len(gw_out)] = [name, xp, pos]
    gw_in = gw_in.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
    gw_out = gw_out.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
    if gw_in.empty:
        transfer_summary = "No transfer made."
    elif gw_out.empty:
        transfer_summary = str(gw_in)
    else:
        transfer_summary = str(pd.concat([gw_out, gw_in], axis=1, join="inner"))
    transfer_summary = (
        f"xP Gain = {net_xp:.2f}    Transfers made = {variable_sums.number_of_transfers[w].value()}\n"
        f"Hits = {lp_variables.penalized_transfers[w].value()}    Total Cost = {net_cost:.1f}\n"
        f"Free Transfers = {lp_variables.free_transfers[w].value()}    In the bank = {lp_variables.in_the_bank[w].value():.2f}\n\n"
        f"{transfer_summary}\n\n"
    )

    return transfer_summary


def get_chip_summary(lp_variables, w):
    if lp_variables.use_wildcard[w].value() > 0.5:
        chip_summary = "[🃏 Wildcard Active]\n"
    elif lp_variables.use_free_hit[w].value() > 0.5:
        chip_summary = "[🆓 Free Hit Active]\n"
    elif lp_variables.use_bench_boost[w].value() > 0.5:
        chip_summary = "[🚀 Bench Boost Active]\n"
    else:
        chip_summary = ""
    return chip_summary


def generate_outputs(
    fpl_data: FplData,
    lp_params: LpParams,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
    solution_time: float,
    parameters: dict,
) -> tuple[pd.DataFrame, list[str], dict]:
    picks_df = generate_picks_df(
        fpl_data, lp_params, lp_keys, lp_variables, variable_sums
    )
    summary, next_gw_dict = generate_summary(
        fpl_data,
        lp_params,
        lp_keys,
        lp_variables,
        variable_sums,
        picks_df,
        parameters,
        solution_time,
    )
    return summary, picks_df, next_gw_dict


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
