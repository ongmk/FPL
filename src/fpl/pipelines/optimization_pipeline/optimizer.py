import logging
import time
from subprocess import Popen

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pulp import LpProblem, lpSum
from tqdm import tqdm

from fpl.pipelines.optimization_pipeline.fpl_api import get_backtest_data
from fpl.pipelines.optimization_pipeline.lp_constructor import LpVariables
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


def generate_picks_df(lp_variables: LpVariables) -> pd.DataFrame:
    picks = []
    for w in lp_variables.gameweeks:
        for p in lp_variables.players:
            if (
                lp_variables.squad[p, w].value()
                + lp_variables.squad_fh[p, w].value()
                + lp_variables.transfer_out[p, w].value()
                > 0.5
            ):
                lp = lp_variables.merged_data.loc[p]
                is_captain = 1 if lp_variables.captain[p, w].value() > 0.5 else 0
                is_squad = (
                    1
                    if (
                        lp_variables.use_fh[w].value() < 0.5
                        and lp_variables.squad[p, w].value() > 0.5
                    )
                    or (
                        lp_variables.use_fh[w].value() > 0.5
                        and lp_variables.squad_fh[p, w].value() > 0.5
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
                for o in lp_variables.order:
                    if lp_variables.bench[p, w, o].value() > 0.5:
                        bench_value = o
                player_buy_price = (
                    0 if not is_transfer_in else lp_variables.buy_price[p]
                )
                player_sell_price = (
                    0
                    if not is_transfer_out
                    else (
                        lp_variables.sell_price[p]
                        if p in lp_variables.price_modified_players
                        and lp_variables.transfer_out_first[p, w].value() > 0.5
                        else lp_variables.buy_price[p]
                    )
                )
                multiplier = 1 * (is_lineup == 1) + 1 * (is_captain == 1)
                xp_cont = lp_variables.points_player_week[p, w] * multiplier
                position = lp_variables.type_data.loc[
                    lp["element_type"], "singular_name_short"
                ]
                picks.append(
                    [
                        w,
                        p,
                        lp["web_name"],
                        position,
                        lp["element_type"],
                        lp["name"],
                        player_buy_price,
                        player_sell_price,
                        round(lp_variables.points_player_week[p, w], 2),
                        is_squad,
                        is_lineup,
                        bench_value,
                        is_captain,
                        is_vice,
                        is_transfer_in,
                        is_transfer_out,
                        multiplier,
                        xp_cont,
                    ]
                )
    picks_df = pd.DataFrame(
        picks,
        columns=[
            "week",
            "id",
            "name",
            "pos",
            "type",
            "team",
            "buy_price",
            "sell_price",
            "predicted_xP",
            "squad",
            "lineup",
            "bench",
            "captain",
            "vicecaptain",
            "transfer_in",
            "transfer_out",
            "multiplier",
            "xp_cont",
        ],
    ).sort_values(
        by=["week", "lineup", "type", "predicted_xP"],
        ascending=[True, False, True, False],
    )
    return picks_df


def generate_summary(
    lp_variables: LpVariables,
    picks_df: pd.DataFrame,
    parameters: dict,
    solution_time: float,
) -> tuple[list[str], dict]:
    summary_of_actions = []
    total_xp = 0
    for w in lp_variables.gameweeks:
        header = f" GW {w} "
        gw_in = pd.DataFrame([], columns=["", "In", "xP", "Pos"])
        gw_out = pd.DataFrame([], columns=["Out", "xP", "Pos"])
        net_cost = 0
        net_xp = 0
        for p in lp_variables.players:
            if lp_variables.transfer_in[p, w].value() > 0.5:
                price = lp_variables.merged_data["now_cost"][p] / 10
                name = f'{lp_variables.merged_data["web_name"][p]} ({price})'
                pos = lp_variables.merged_data["element_type"][p]
                xp = round(lp_variables.points_player_week[p, w], 2)
                net_cost += price
                net_xp += xp
                gw_in.loc[len(gw_in)] = ["👉", name, xp, pos]
            if lp_variables.transfer_out[p, w].value() > 0.5:
                price = lp_variables.merged_data["sell_price"][p] / 10
                name = f'{lp_variables.merged_data["web_name"][p]} ({price})'
                pos = lp_variables.merged_data["element_type"][p]
                xp = round(lp_variables.points_player_week[p, w], 2)
                net_cost -= price
                net_xp -= xp
                gw_out.loc[len(gw_out)] = [name, xp, pos]
        gw_in = gw_in.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
        gw_out = gw_out.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
        if lp_variables.use_wc[w].value() > 0.5:
            chip_summary = "[Wildcard Active]\n"
        elif lp_variables.use_fh[w].value() > 0.5:
            chip_summary = "[Free Hit Active]\n"
        elif lp_variables.use_bb[w].value() > 0.5:
            chip_summary = "[Bench Boost Active]\n"
        else:
            chip_summary = ""
        if w in lp_variables.transfer_gws:
            transfer_summary = (
                f"Free Transfers = {lp_variables.free_transfers[w].value()}    Hits = {lp_variables.penalized_transfers[w].value()}\n"
                f"Cost = {net_cost:.1f}    ITB = {lp_variables.in_the_bank[w].value():.2f}   xP Gain = {net_xp:.2f}.\n\n"
                f"{str(gw_in) if gw_out.empty else str(pd.concat([gw_out, gw_in], axis=1, join='inner'))}\n\n"
            )
        else:
            transfer_summary = "\n"

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
        gw_xp = (
            lpSum(
                [
                    (lp_variables.lineup[p, w] + lp_variables.captain[p, w])
                    * lp_variables.points_player_week[p, w]
                    for p in lp_variables.players
                ]
            ).value()
            - lp_variables.penalized_transfers[w].value() * 4
        )
        total_xp += gw_xp
        hits = int(lp_variables.penalized_transfers[w].value())
        hit_str = f"({hits} hits)" if hits > 0 else ""
        gw_summary = (
            f"\n"
            f"{header:{'*'}^80}\n\n"
            f"{chip_summary}"
            f"{transfer_summary}"
            f"{lineup_str}\n\n"
            f"Gameweek xP = {gw_xp:.2f} {hit_str}\n"
        )
        summary_of_actions.append(gw_summary)
        if w == lp_variables.next_gw:
            if lp_variables.use_wc[w].value() > 0.5:
                chip_used = "wc"
            elif lp_variables.use_fh[w].value() > 0.5:
                chip_used = "fh"
            elif lp_variables.use_bb[w].value() > 0.5:
                chip_used = "bb"
            else:
                chip_used = None
            next_gw_dict = {
                "itb": round(lp_variables.in_the_bank[w].value(), 1),
                "ft": (
                    round(lp_variables.free_transfers[w + 1].value())
                    if w + 1 <= 38
                    else 0
                ),
                "hits": round(lp_variables.penalized_transfers[w].value()),
                "solve_time": solution_time,
                "n_transfers": round(
                    lpSum(
                        [lp_variables.transfer_out[p, w] for p in lp_variables.players]
                    ).value()
                ),
                "chip_used": chip_used,
            }
    overall_summary = (
        f"\n"
        f"{'':{'='}^80}\n"
        f"{parameters['horizon']} weeks total xP = {total_xp:.2f}"
    )
    summary_of_actions.append(overall_summary)
    return summary_of_actions, next_gw_dict


def solve_lp(
    lp_constructed: bool, parameters: dict
) -> tuple[pd.DataFrame, list[str], dict]:

    mps_path = parameters["mps_path"]
    solution_path = parameters["solution_path"]
    solution_init_path = f"init_{solution_path}"

    t0 = time.time()
    command = f"cbc/bin/cbc.exe {mps_path} cost column ratio 1 solve solu {solution_init_path}"
    process = Popen(command, shell=False)
    process.wait()
    command = f"cbc/bin/cbc.exe {mps_path} mips {solution_init_path} cost column sec 20 solve solu {solution_path}"
    process = Popen(command, shell=False)
    process.wait()
    t1 = time.time()
    print(t1 - t0, "seconds passed")

    _, model = LpProblem.fromMPS(mps_path)

    backup_latest_n(solution_path, 5)

    for variable in model.variables():
        variable.varValue = 0
    vars = model.variablesDict()

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
            # words = [word for word in words if word != "**"]

            var_name, var_value = words[1], float(words[2])
            vars[var_name].varValue = var_value
    return None


def generate_outputs(
    lp_variables: LpVariables, parameters: dict, solution_time: float
) -> tuple[pd.DataFrame, list[str], dict]:
    picks_df = generate_picks_df(lp_variables)
    summary_of_actions, next_gw_dict = generate_summary(
        lp_variables, picks_df, parameters, solution_time
    )
    return picks_df, summary_of_actions, next_gw_dict


def get_historical_picks(team_id, next_gw, merged_data):
    initial_squad = get_initial_squad(team_id, next_gw)
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
        "ft": 0,
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
    parameters["ft"] = 1
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
            f"Backtesting GW {next_gw}. ITB = {itb:.1f}. FT = {parameters['ft']}. {gameweeks}"
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
                    assert next_gw_dict["ft"] == min(
                        2, max(1, parameters["ft"] - next_gw_dict["n_transfers"] + 1)
                    )
                else:
                    assert next_gw_dict["ft"] == parameters["ft"]

            itb = next_gw_dict["itb"]
            parameters["ft"] = next_gw_dict["ft"]
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
