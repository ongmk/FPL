import logging
from typing import Literal, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

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

plt.style.use("ggplot")
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def get_chip_used(
    gameweek: int, lp_variables: LpVariables, variable_sums: VariableSums
) -> Optional[Literal["Wildcard", "Bench Boost", "Free Hit", "Triple Captain"]]:
    if lp_variables.use_wildcard[gameweek].value() > 0.5:
        return "Wildcard"
    elif lp_variables.use_free_hit[gameweek].value() > 0.5:
        return "Free Hit"
    elif lp_variables.use_bench_boost[gameweek].value() > 0.5:
        return "Bench Boost"
    elif variable_sums.use_triple_captain_week[gameweek].value() > 0.5:
        return "Triple Captain"
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
    gameweek: int, current_squad: list[int], previous_squad: list[int], lp_data: LpData
) -> list[dict[str, Union[int, float]]]:
    in_players = [
        player for player in current_squad if player not in lp_data.initial_squad
    ]
    out_players = [player for player in previous_squad if player not in current_squad]
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
        Literal["Wildcard", "Bench Boost", "Free Hit", "Triple Captain"]
    ],
    hits: int,
    lp_keys: LpKeys,
) -> float:
    total_points = points.loc[lineup].sum()
    captain_mins = minutes.loc[captain]
    captain_multiplier = 2 if chip_used == "Triple Captain" else 1
    if captain_mins > 0:
        total_points += points.loc[captain] * captain_multiplier
    else:
        total_points += points.loc[vicecap] * captain_multiplier
    if chip_used == "Bench Boost":
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
    previous_squad: list[int],
) -> GwResults:
    chip_used = get_chip_used(gameweek, lp_variables, variable_sums)
    lineup, bench = get_lineup_bench(gameweek, lp_keys, lp_variables)
    current_squad = lineup + list(bench.values())
    transfer_data = get_transfer_data(gameweek, current_squad, previous_squad, lp_data)
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
    relevant_players = (
        lineup
        + list(bench.values())
        + [d["element_out"] for d in transfer_data if d["element_out"] is not None]
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
        player_details=lp_data.merged_data.loc[relevant_players],
        solution_time=solution_time,
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


def get_lineup_name(
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
        lambda row: get_lineup_name(
            row,
            gw_results.captain,
            gw_results.vicecap,
            gw_results.gameweek,
            gw_results.total_actual_points is not None,
        ),
        axis=1,
    )

    lineup_str = []
    for details in TYPE_DATA:
        p_type = details["id"]
        lineup_str.append(
            (gw_lineup.loc[gw_lineup["element_type"] == p_type, "name"]).str.cat(
                sep="    "
            )
        )
    lineup_str.append("")
    lineup_str.append("Bench:")
    for order in sorted(gw_results.bench.keys()):
        player_id = gw_results.bench[order]
        name = get_lineup_name(
            gw_results.player_details.loc[player_id],
            gw_results.captain,
            gw_results.vicecap,
            gw_results.gameweek,
            gw_results.total_actual_points is not None,
        )
        lineup_str.append(f"{order}: {name}")
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
        details = gw_results.player_details.loc[player]
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
    if gw_results.chip_used == "Wildcard":
        return "🃏🃏🃏 WILDCARD ACTIVE 🃏🃏🃏"  # fmt: skip
    elif gw_results.chip_used == "Free Hit":
        return "🆓🆓🆓 FREE HIT ACTIVE 🆓🆓🆓"  # fmt: skip
    elif gw_results.chip_used == "Bench Boost":
        return "🚀🚀🚀 BENCH BOOST ACTIVE 🚀🚀🚀"  # fmt: skip
    elif gw_results.chip_used == "Triple Captain":
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
    previous_squad = lp_data.initial_squad
    for w in lp_data.gameweeks:
        gw_results = get_gw_results(
            w,
            lp_data,
            lp_keys,
            lp_variables,
            variable_sums,
            solution_time,
            previous_squad,
        )
        summary.append(get_gw_summary(gw_results))
        total_predicted_points += gw_results.total_predicted_points
        if gw_results.total_actual_points is not None:
            actual_points_available = True
            total_actual_points += gw_results.total_actual_points
        previous_squad = gw_results.lineup + list(gw_results.bench.values())

    summary.append(
        f"{len(lp_data.gameweeks):>2} weeks total predicted points = {total_predicted_points:.2f}"
    )
    if actual_points_available:
        summary.append(f"         total actual points    = {total_actual_points:.2f}")
    summary.append(f"Optimization time = {int(solution_time)} seconds\n")

    for s in summary:
        logger.info(s)
    summary = "\n".join(summary)
    return summary


def shorten_name(name):
    name = name.split(" ")
    for i in range(len(name) - 1):
        name[i] = name[i][0]
    return ".".join(name)


def get_name(row, in_players, out_players, gameweek):
    if row["captain"] == True:
        cap_string = "[c] "
    elif row["vicecap"] == True:
        cap_string = "[v] "
    else:
        cap_string = ""
    if row.name in in_players:
        in_out_string = "+ "
    elif row.name in out_players:
        in_out_string = "- "
    else:
        in_out_string = ""
    shorter_name = shorten_name(row["web_name"])
    return f"{in_out_string}{cap_string}{shorter_name}: {int(row[f'act_pts_{gameweek}'])} ({int(row[f'pred_pts_{gameweek}'])})"


def get_squad_names(gw_results: GwResults, out_players: list[int]):
    player_details = gw_results.player_details.sort_values(
        ["element_type", f"act_pts_{gw_results.gameweek}"], ascending=[True, False]
    )
    player_details.loc[gw_results.captain, "captain"] = True
    player_details.loc[gw_results.vicecap, "vicecap"] = True

    lines = []
    in_players = [gw["element_in"] for gw in gw_results.transfer_data]
    for _, group in player_details.groupby("element_type"):
        for i, row in group.iterrows():
            if i in gw_results.lineup:
                lines.append(
                    get_name(row, in_players, out_players, gw_results.gameweek)
                )
        lines.append("")
    lines.pop()
    lines.append("----------")

    for _, player in gw_results.bench.items():
        lines.append(
            get_name(
                player_details.loc[player], in_players, out_players, gw_results.gameweek
            )
        )

    return lines


def plot_backtest_results(backtest_results: list[GwResults], title: str) -> Figure:
    gameweeks = []
    predicted_pts = []
    actual_pts = []
    free_transfers = []
    in_the_bank = []
    squad = []
    chip_used = []
    hits = []
    cumulative_actual_pts = 0
    cumulative_predicted_pts = 0
    for idx, res in enumerate(backtest_results):
        gameweeks.append(res.gameweek)
        cumulative_predicted_pts += res.total_predicted_points
        predicted_pts.append(cumulative_predicted_pts)
        cumulative_actual_pts += res.total_actual_points
        actual_pts.append(cumulative_actual_pts)
        free_transfers.append(res.free_transfers)
        in_the_bank.append(res.in_the_bank)
        out_players = (
            [
                transfers["element_out"]
                for transfers in backtest_results[idx + 1].transfer_data
            ]
            if idx + 1 < len(backtest_results)
            else []
        )
        squad.append(get_squad_names(res, out_players))
        chip_used.append(res.chip_used)
        hits.append(res.hits)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(len(backtest_results) * 2, 15),
        gridspec_kw={"height_ratios": [4, 2, 1, 1]},
    )
    fig.suptitle(title, fontsize=20)

    ax1.set_xticks(gameweeks)
    ax1.set_xticklabels([f"GW{gw}" for gw in gameweeks])
    ax1.plot(gameweeks, actual_pts, color="blue", marker="o", label="Actual")
    ax1.plot(
        gameweeks,
        predicted_pts,
        color="grey",
        marker="o",
        linestyle=":",
        label="Predicted",
    )
    for gw, pred, act, chip, gw_hits in zip(
        gameweeks, predicted_pts, actual_pts, chip_used, hits
    ):
        tooltip = ""
        if chip is not None:
            tooltip += f"{chip}\n"
        if gw_hits > 0:
            tooltip += f"{gw_hits} hits\n"
        y = max(pred, act)
        ax1.text(gw, y, tooltip, color="red", fontsize=10, ha="center", va="bottom")
    ax1.set_title("Total Points")
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set_ylim(ymin=0)
    ax1.legend()

    for x, lines in enumerate(squad):
        for y, text in enumerate(lines):
            if "+ " in text:
                color = "green"
            elif "- " in text:
                color = "red"
            else:
                color = "black"
            ax2.text(x + 1, y, text, color=color, fontsize=9, ha="center", va="bottom")

    ax2.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_facecolor("white")
    ax2.set_ylim(0, max([len(x) for x in squad]) + 1)
    ax2.invert_yaxis()

    ax3.plot(gameweeks, in_the_bank, color="green", marker="o")
    ax3.set_title("In the Bank")
    ax3.xaxis.set_tick_params(labelbottom=True)

    ax4.plot(gameweeks, free_transfers, color="red", marker="o")
    ax4.set_title("Free Transfers")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
