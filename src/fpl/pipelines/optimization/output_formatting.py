import logging

import pandas as pd
from pulp import lpSum

from fpl.pipelines.optimization.fpl_api import FplData
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


def get_name(row):
    name = ""
    if row["captain"] == 1:
        name += "[c] "
    elif row["vicecaptain"] == 1:
        name += "[v] "
    name += f"{row['name']} {row['predicted_xP']}"
    return name


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
            f"\n{header:{'*'}^80}\n\n"
            f"Gameweek xP = {gw_xp:.2f} {hit_str}\n\n"
            f"{chip_summary}"
            f"{transfer_summary}"
            f"{lineup}\n"
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
    summary = "\n".join(summary) + "\n\n"
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


def get_transfer_summary(
    fpl_data: FplData,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
    w: int,
):
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
        f"Rem. Free Transfers = {lp_variables.free_transfers[w+1].value()}    In the bank = {lp_variables.in_the_bank[w].value():.2f}\n\n"
        f"{transfer_summary}\n\n"
    )

    return transfer_summary


def get_chip_summary(lp_variables, w):
    if lp_variables.use_wildcard[w].value() > 0.5:
        chip_summary = "🃏🃏🃏 WILDCARD ACTIVEive🃏🃏🃏\n"
    elif lp_variables.use_free_hit[w].value() > 0.5:
        chip_summary = "🆓🆓🆓 FREE HIT ACTIVE 🆓🆓🆓\n"
    elif lp_variables.use_bench_boost[w].value() > 0.5:
        chip_summary = "🚀🚀🚀 BENCH BOOST ACTIVE🚀🚀🚀\n"
    else:
        chip_summary = ""
    return chip_summary


def generate_outputs(
    fpl_data: FplData,
    lp_variables: LpVariables,
    solution_time: float,
    parameters: dict,
) -> tuple[pd.DataFrame, list[str], dict]:
    lp_params = prepare_lp_params(fpl_data, parameters)
    lp_keys = prepare_lp_keys(fpl_data, lp_params)
    variable_sums = sum_lp_variables(fpl_data, lp_params, lp_keys, lp_variables)

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
