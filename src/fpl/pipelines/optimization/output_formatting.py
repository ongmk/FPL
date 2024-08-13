import logging
from typing import Literal, Optional, Union

import pandas as pd

from fpl.pipelines.optimization.data_classes import (
    TYPE_DATA,
    GwResults,
    LpData,
    StartingParams,
)
from fpl.pipelines.optimization.lp_constructor import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
    prepare_lp_keys,
    prepare_lp_params,
    sum_lp_variables,
)

logger = logging.getLogger(__name__)


def get_chip_used(
    gameweek: int, lp_variables: LpVariables, variable_sums: VariableSums
) -> str:
    if lp_variables.use_wildcard[gameweek].value() > 0.5:
        return "wildcard"
    elif lp_variables.use_free_hit[gameweek].value() > 0.5:
        return "free hit"
    elif lp_variables.use_bench_boost[gameweek].value() > 0.5:
        return "bench boost"
    elif variable_sums.use_triple_captain_week[gameweek].value() > 0.5:
        return "triple captain"
    else:
        return None


def get_lineup_bench(
    gameweek: int, lp_keys: LpKeys, lp_variables: LpVariables
) -> tuple[list[int], dict[int, int]]:
    lineup = []
    bench = {}
    for p in lp_keys.players:
        if lp_variables.lineup[p, gameweek].value() > 0.5:
            lineup.append(p)
        for o in lp_keys.order:
            if lp_variables.bench[p, gameweek, o].value() > 0.5:
                bench[o] = p
    return lineup, bench


def get_transfer_data(
    gameweek: int, squad: list[int], lp_data: LpData
) -> list[dict[str, Union[int, float]]]:
    in_players = [player for player in squad if player not in lp_data.initial_squad]
    out_players = [player for player in lp_data.initial_squad if player not in squad]
    out_players = out_players + (len(in_players) - len(out_players)) * [None]

    transfer_data = []
    for in_player, out_player in zip(in_players, out_players):
        transfer_data.append(
            {
                "element_in": in_player,
                "element_in_cost": lp_data.merged_data.loc[in_player, "now_cost"],
                "element_out": out_player,
                "entry": gameweek,
            }
        )
    return transfer_data


def get_captain_vicecap(
    gameweek: int, lp_keys: LpKeys, lp_variables: LpVariables
) -> tuple[int, int]:
    for p in lp_keys.players:
        if lp_variables.captain[p, gameweek].value() > 0.5:
            captain = p
        elif lp_variables.vicecap[p, gameweek].value() > 0.5:
            vicecap = p
    return captain, vicecap


def calculate_gw_points(
    points: pd.Series,
    minutes: pd.Series,
    lineup: list[int],
    captain: int,
    vicecap: int,
    bench: dict[int, int],
    chip_used: Optional[
        Literal["wildcard", "bench_boost", "free_hit", "triple_captain"]
    ],
    hits: int,
    lp_keys: LpKeys,
) -> float:
    total_points = points.loc[lineup].sum()
    captain_mins = minutes.loc[captain]
    captain_multiplier = 2 if chip_used == "triple_captain" else 1
    if captain_mins > 0:
        total_points += points.loc[captain] * captain_multiplier
    else:
        total_points += points.loc[vicecap] * captain_multiplier
    if chip_used == "bench_boost":
        for o in lp_keys.order:
            total_points += points.loc[bench[o]]
    total_points -= hits * 4
    return total_points


def get_gw_results(
    gameweek: int,
    lp_data: LpData,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
    solution_time: int,
) -> GwResults:
    chip_used = get_chip_used(gameweek, lp_variables, variable_sums)
    lineup, bench = get_lineup_bench(gameweek, lp_keys, lp_variables)
    transfer_data = get_transfer_data(gameweek, lineup + list(bench.values()), lp_data)
    captain, vicecap = get_captain_vicecap(gameweek, lp_keys, lp_variables)
    hits = round(lp_variables.penalized_transfers[gameweek].value())
    total_actual_points = None
    if f"act_pts_{gameweek}" in lp_data.merged_data.columns:
        total_actual_points = calculate_gw_points(
            lp_data.merged_data[f"act_pts_{gameweek}"],
            lp_data.merged_data[f"mins_{gameweek}"],
            lineup,
            captain,
            vicecap,
            bench,
            chip_used,
            hits,
            lp_keys,
        )
    predicted_points = lp_data.merged_data[f"pred_pts_{gameweek}"]
    dummy_minutes = pd.Series(90, index=lp_data.merged_data.index)
    total_predicted_points = calculate_gw_points(
        predicted_points,
        dummy_minutes,
        lineup,
        captain,
        vicecap,
        bench,
        chip_used,
        hits,
        lp_keys,
    )

    return GwResults(
        gameweek=gameweek,
        transfer_data=transfer_data,
        captain=captain,
        vicecap=vicecap,
        lineup=lineup,
        bench=bench,
        chip_used=chip_used,
        hits=hits,
        total_predicted_points=total_predicted_points,
        total_actual_points=total_actual_points,
        free_transfers=round(lp_variables.free_transfers[gameweek].value()),
        in_the_bank=round(lp_variables.in_the_bank[gameweek].value(), 1),
        starting_params=StartingParams(
            gameweek=gameweek - 1,
            free_transfers=round(lp_variables.free_transfers[gameweek - 1].value()),
            in_the_bank=round(lp_variables.in_the_bank[gameweek - 1].value(), 1),
        ),
        player_details=lp_data.merged_data.loc[lineup + list(bench.values())],
    )


def get_gw_summary(
    gw_results: GwResults,
) -> tuple[list[str], dict]:
    gw_summary = []
    header = f" GW {gw_results.gameweek} "
    transfer_summary = get_transfer_summary(gw_results)
    chip_summary = get_chip_summary(gw_results)
    lineup = get_lineup(gw_results)
    hit_str = f"({gw_results.hits} hits)" if gw_results.hits > 0 else ""

    gw_summary.append(f"\n{header:{'*'}^80}\n")
    if chip_summary is not None:
        gw_summary.append(chip_summary)
    if gw_results.total_actual_points is not None:
        gw_summary.append(
            f"Gameweek Actual Points    = {gw_results.total_actual_points:.2f} {hit_str}"
        )
    gw_summary.append(
        f"Gameweek Predicted Points = {gw_results.total_predicted_points:.2f} {hit_str}"
    )
    gw_summary.append(transfer_summary)
    gw_summary.append(lineup)
    return "\n".join(gw_summary)


def get_name(
    row: pd.Series,
    captain: int,
    vicecap: int,
    gameweek: int,
    actual_points_available: bool,
) -> str:
    name = ""
    if row.name == captain:
        name += "[c] "
    elif row.name == vicecap:
        name += "[v] "
    points = (
        row[f"act_pts_{gameweek}"]
        if actual_points_available
        else row[f"pred_pts_{gameweek}"]
    )
    name += f"{row['web_name']} {points:.1f}"
    return name


def get_lineup(gw_results: GwResults):

    gw_lineup = gw_results.player_details.loc[gw_results.lineup]
    gw_lineup["name"] = gw_lineup.apply(
        lambda row: get_name(
            row,
            gw_results.captain,
            gw_results.vicecap,
            gw_results.gameweek,
            gw_results.total_actual_points is not None,
        ),
        axis=1,
    )

    lineup_str = []
    for p_type in TYPE_DATA["id"]:
        lineup_str.append(
            (gw_lineup.loc[gw_lineup["element_type"] == p_type, "name"]).str.cat(
                sep="    "
            )
        )
    lineup_str.append("")
    lineup_str.append("Bench:")
    for k, v in gw_results.bench.items():
        name = get_name(
            gw_results.player_details.loc[v],
            gw_results.captain,
            gw_results.vicecap,
            gw_results.gameweek,
            gw_results.total_actual_points is not None,
        )
        lineup_str.append(f"{k}: {name}")
    length = max([len(s) for s in lineup_str])
    lineup_str = [f"{s:^{length}}" for s in lineup_str]
    lineup_str = "\n".join(lineup_str)
    return lineup_str


def get_transfer_summary(
    gw_results: GwResults,
):
    actual_points_available = gw_results.total_actual_points is not None

    gw_in = pd.DataFrame([], columns=["", "In", "Points", "Pos"])
    gw_out = pd.DataFrame([], columns=["Out", "Points", "Pos"])
    net_cost = 0
    net_points = 0

    in_players = [transfer["element_in"] for transfer in gw_results.transfer_data]
    for player in in_players:
        details = gw_results.player_details.loc[player]
        price = details["now_cost"] / 10
        name = f'{details["web_name"]} ({price})'
        pos = details["element_type"]
        points = (
            details[f"act_pts_{gw_results.gameweek}"]
            if actual_points_available
            else details[f"pred_pts_{gw_results.gameweek}"]
        )
        net_cost += price
        net_points += points
        gw_in.loc[len(gw_in)] = ["👉", name, points, pos]

    out_players = [
        transfer["element_out"]
        for transfer in gw_results.transfer_data
        if transfer["element_out"] is not None
    ]
    for player in out_players:
        price = details["sell_price"] / 10
        name = f'{details["web_name"]} ({price})'
        pos = details["element_type"]
        points = (
            details[f"act_pts_{gw_results.gameweek}"]
            if actual_points_available
            else details[f"pred_pts_{gw_results.gameweek}"]
        )
        net_cost -= price
        net_points -= points
        gw_out.loc[len(gw_out)] = [name, points, pos]
    gw_in = gw_in.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
    gw_out = gw_out.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
    if gw_in.empty:
        transfer_summary = "No transfer made."
    elif gw_out.empty:
        transfer_summary = str(gw_in)
    else:
        transfer_summary = str(pd.concat([gw_out, gw_in], axis=1, join="inner"))
    transfer_summary = (
        f"Points Gain = {net_points:.2f}    Transfers made = {len(gw_results.transfer_data)}\n"
        f"Hits = {gw_results.hits}    Total Cost = {net_cost:.1f}\n"
        f"Rem. Free Transfers = {gw_results.free_transfers}    Rem. In the bank = {gw_results.free_transfers}\n\n"
        f"{transfer_summary}\n"
    )

    return transfer_summary


def get_chip_summary(gw_results: GwResults):
    if gw_results.chip_used == "wildcard":
        return "🃏🃏🃏 WILDCARD ACTIVE 🃏🃏🃏"  # fmt: skip
    elif gw_results.chip_used == "free_hit":
        return "🆓🆓🆓 FREE HIT ACTIVE 🆓🆓🆓"  # fmt: skip
    elif gw_results.chip_used == "bench_boost":
        return "🚀🚀🚀 BENCH BOOST ACTIVE 🚀🚀🚀"  # fmt: skip
    elif gw_results.chip_used == "triple_captain":
        return "👑👑👑 TRIPLE CAPTAIN ACTIVE 👑👑👑"  # fmt: skip
    else:
        return None


def generate_outputs(
    lp_data: LpData,
    lp_variables: LpVariables,
    solution_time: float,
    parameters: dict,
) -> tuple[pd.DataFrame, list[str], dict]:
    lp_params = prepare_lp_params(lp_data, parameters)
    lp_keys = prepare_lp_keys(lp_data, lp_params)
    variable_sums = sum_lp_variables(lp_data, lp_params, lp_keys, lp_variables)

    summary = [
        "=" * 80,
        (
            f"Team: {lp_data.team_name}.\n"
            f"In the bank = {lp_data.in_the_bank:.1f}    Free transfers = {lp_data.free_transfers}\n"
            f"Optimizing for gameweeks : {', '.join([str(w) for w in lp_data.gameweeks])}"
        ),
        "=" * 80,
    ]
    total_predicted_points = 0
    total_actual_points = 0
    actual_points_available = False
    for w in lp_data.gameweeks:
        gw_results = get_gw_results(
            w, lp_data, lp_keys, lp_variables, variable_sums, solution_time
        )
        summary.append(get_gw_summary(gw_results))
        total_predicted_points += gw_results.total_predicted_points
        if gw_results.total_actual_points is not None:
            actual_points_available = True
            total_actual_points += gw_results.total_actual_points

    summary.append(
        f"{len(lp_data.gameweeks):>2} weeks total predicted points = {total_predicted_points:.2f}"
    )
    if actual_points_available:
        summary.append(f"         total actual points    = {total_actual_points:.2f}")
    summary.append(f"Optimization time = {int(solution_time)} seconds\n")

    summary = "\n".join(summary)

    return summary
