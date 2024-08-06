import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

from pulp import (
    LpBinary,
    LpContinuous,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
)

from fpl.pipelines.optimization.constraints import (
    ChipConstraints,
    SquadConstraints,
    TransferConstraints,
)
from fpl.pipelines.optimization.data_classes import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.pipelines.optimization.fpl_api import FplData
from fpl.utils import backup_latest_n


def prepare_lp_params(fpl_data: FplData, parameters: dict[str, Any]) -> LpParams:
    wildcard_week = parameters["wildcard_week"]
    bench_boost_week = parameters["bench_boost_week"]
    free_hit_week = parameters["free_hit_week"]
    tr_horizon = parameters["tr_horizon"]
    next_gw = fpl_data.gameweeks[0]
    transfer_gws = fpl_data.gameweeks[:tr_horizon]

    return LpParams(
        next_gw=next_gw,
        transfer_gws=transfer_gws,
        threshold_gw=2 if next_gw == 1 else next_gw,
        ft=parameters["ft"],
        horizon=parameters["horizon"],
        wildcard_week=wildcard_week if wildcard_week in transfer_gws else None,
        bench_boost_week=bench_boost_week if bench_boost_week in transfer_gws else None,
        free_hit_week=free_hit_week if free_hit_week in transfer_gws else None,
        decay=parameters["decay"],
        ft_bonus=parameters["ft_bonus"],
        itb_bonus=parameters["itb_bonus"],
        bench_weights=parameters["bench_weights"],
    )


def prepare_lp_keys(fpl_data: FplData, lp_params: LpParams) -> LpKeys:
    players = fpl_data.merged_data.index.to_list()
    all_gws = [lp_params.next_gw - 1] + fpl_data.gameweeks
    order = [0, 1, 2, 3]
    price_modified_players = fpl_data.merged_data.loc[
        fpl_data.merged_data["sell_price"] != fpl_data.merged_data["now_cost"]
    ].index.to_list()

    return LpKeys(
        element_types=fpl_data.type_data.index.to_list(),
        teams=fpl_data.team_data["name"].to_list(),
        players=players,
        price_modified_players=price_modified_players,
        all_gws=all_gws,
        player_all_gws=list(itertools.product(players, all_gws)),
        player_gameweeks=list(itertools.product(players, fpl_data.gameweeks)),
        order=order,
        player_gameweeks_order=list(
            itertools.product(players, fpl_data.gameweeks, order)
        ),
        price_modified_players_gameweeks=list(
            itertools.product(price_modified_players, fpl_data.gameweeks)
        ),
        player_type=fpl_data.merged_data["element_type"].to_dict(),
        sell_price=(fpl_data.merged_data["sell_price"] / 10).to_dict(),
        buy_price=(fpl_data.merged_data["now_cost"] / 10).to_dict(),
    )


def initialize_variables(fpl_data: FplData, lp_keys: LpKeys) -> LpVariables:
    squad = LpVariable.dicts("squad", lp_keys.player_all_gws, cat=LpBinary)
    squad_free_hit = LpVariable.dicts(
        "squad_free_hit", lp_keys.player_gameweeks, cat=LpBinary
    )
    lineup = LpVariable.dicts("lineup", lp_keys.player_gameweeks, cat=LpBinary)
    captain = LpVariable.dicts("captain", lp_keys.player_gameweeks, cat=LpBinary)
    vicecap = LpVariable.dicts("vicecap", lp_keys.player_gameweeks, cat=LpBinary)
    bench = LpVariable.dicts("bench", lp_keys.player_gameweeks_order, cat=LpBinary)
    transfer_in = LpVariable.dicts(
        "transfer_in", lp_keys.player_gameweeks, cat=LpBinary
    )
    transfer_out_first = LpVariable.dicts(
        "transfer_out_first", lp_keys.price_modified_players_gameweeks, cat=LpBinary
    )
    transfer_out_regular = LpVariable.dicts(
        "transfer_out_regular", lp_keys.player_gameweeks, cat=LpBinary
    )
    transfer_out = {
        (p, w): transfer_out_regular[p, w]
        + (transfer_out_first[p, w] if p in lp_keys.price_modified_players else 0)
        for p in lp_keys.players
        for w in fpl_data.gameweeks
    }
    in_the_bank = LpVariable.dicts(
        "in_the_bank", lp_keys.all_gws, cat=LpContinuous, lowBound=0
    )
    free_transfers = LpVariable.dicts(
        "free_transfers", lp_keys.all_gws, cat=LpInteger, lowBound=0, upBound=2
    )
    penalized_transfers = LpVariable.dicts(
        "penalized_transfers", fpl_data.gameweeks, cat=LpInteger, lowBound=0
    )
    aux = LpVariable.dicts("aux", fpl_data.gameweeks, cat=LpBinary)

    use_wildcard = LpVariable.dicts("use_wildcard", fpl_data.gameweeks, cat=LpBinary)
    use_bench_boost = LpVariable.dicts(
        "use_bench_boost", fpl_data.gameweeks, cat=LpBinary
    )
    use_free_hit = LpVariable.dicts("use_free_hit", fpl_data.gameweeks, cat=LpBinary)
    return LpVariables(
        squad=squad,
        squad_free_hit=squad_free_hit,
        lineup=lineup,
        captain=captain,
        vicecap=vicecap,
        bench=bench,
        transfer_in=transfer_in,
        transfer_out=transfer_out,
        transfer_out_first=transfer_out_first,
        transfer_out_regular=transfer_out_regular,
        in_the_bank=in_the_bank,
        free_transfers=free_transfers,
        penalized_transfers=penalized_transfers,
        aux=aux,
        use_wildcard=use_wildcard,
        use_bench_boost=use_bench_boost,
        use_free_hit=use_free_hit,
    )


def sum_lp_variables(
    fpl_data: FplData, lp_params: LpParams, lp_keys: LpKeys, lp_variables: LpVariables
) -> VariableSums:
    lineup_type_count = {
        (t, w): lpSum(
            lp_variables.lineup[p, w]
            for p in lp_keys.players
            if fpl_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in fpl_data.gameweeks
    }
    squad_type_count = {
        (t, w): lpSum(
            lp_variables.squad[p, w]
            for p in lp_keys.players
            if fpl_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in fpl_data.gameweeks
    }
    squad_free_hit_type_count = {
        (t, w): lpSum(
            lp_variables.squad_free_hit[p, w]
            for p in lp_keys.players
            if fpl_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in fpl_data.gameweeks
    }
    sold_amount = {
        w: lpSum(
            [
                lp_keys.sell_price[p] * lp_variables.transfer_out_first[p, w]
                for p in lp_keys.price_modified_players
            ]
        )
        + lpSum(
            [
                lp_keys.buy_price[p] * lp_variables.transfer_out_regular[p, w]
                for p in lp_keys.players
            ]
        )
        for w in fpl_data.gameweeks
    }
    fh_sell_price = {
        p: (
            lp_keys.sell_price[p]
            if p in lp_keys.price_modified_players
            else lp_keys.buy_price[p]
        )
        for p in lp_keys.players
    }
    bought_amount = {
        w: lpSum(
            [
                lp_keys.buy_price[p] * lp_variables.transfer_in[p, w]
                for p in lp_keys.players
            ]
        )
        for w in fpl_data.gameweeks
    }
    points_player_week = {
        (p, w): fpl_data.merged_data.loc[p, f"xPts_{w}"]
        for p in lp_keys.players
        for w in fpl_data.gameweeks
    }
    squad_count = {
        w: lpSum(lp_variables.squad[p, w] for p in lp_keys.players)
        for w in fpl_data.gameweeks
    }
    squad_free_hit_count = {
        w: lpSum(lp_variables.squad_free_hit[p, w] for p in lp_keys.players)
        for w in fpl_data.gameweeks
    }
    number_of_transfers = {
        w: lpSum([lp_variables.transfer_out[p, w] for p in lp_keys.players])
        for w in fpl_data.gameweeks
    }
    number_of_transfers[lp_params.next_gw - 1] = 1
    transfer_diff = {
        w: number_of_transfers[w]
        - lp_variables.free_transfers[w]
        - 15 * lp_variables.use_wildcard[w]
        for w in fpl_data.gameweeks
    }

    return VariableSums(
        lineup_type_count=lineup_type_count,
        squad_type_count=squad_type_count,
        squad_free_hit_type_count=squad_free_hit_type_count,
        sold_amount=sold_amount,
        fh_sell_price=fh_sell_price,
        bought_amount=bought_amount,
        points_player_week=points_player_week,
        squad_count=squad_count,
        squad_free_hit_count=squad_free_hit_count,
        number_of_transfers=number_of_transfers,
        transfer_diff=transfer_diff,
    )


def add_constraints(fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums):
    ChipConstraints.global_level(
        fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )
    TransferConstraints.global_level(
        fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )
    for p in lp_keys.players:
        SquadConstraints.player_level(
            p, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )
        TransferConstraints.player_level(
            p, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )

    for w in fpl_data.gameweeks:
        SquadConstraints.gameweek_level(
            w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )
        TransferConstraints.gameweek_level(
            w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )

        for t in lp_keys.element_types:
            SquadConstraints.type_gameweek_level(
                t, w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )

        for t in lp_keys.teams:
            SquadConstraints.team_gameweek_level(
                t, w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )

        for p in lp_keys.players:
            SquadConstraints.player_gameweek_level(
                p, w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )
            TransferConstraints.player_gameweek_level(
                p, w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )
            ChipConstraints.player_gameweek_level(
                p, w, fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )


def add_objective_function(
    fpl_data: FplData,
    model: LpProblem,
    lp_params: LpParams,
    lp_keys: LpKeys,
    lp_variables: LpVariables,
    variable_sums: VariableSums,
):
    gw_xp = {
        w: lpSum(
            [
                variable_sums.points_player_week[p, w]
                * (
                    lp_variables.lineup[p, w]
                    + lp_variables.captain[p, w]
                    + 0.1 * lp_variables.vicecap[p, w]
                    + lpSum(
                        lp_params.bench_weights[o] * lp_variables.bench[p, w, o]
                        for o in lp_keys.order
                    )
                )
                for p in lp_keys.players
            ]
        )
        for w in fpl_data.gameweeks
    }
    gw_total = {
        w: gw_xp[w]
        - 4 * lp_variables.penalized_transfers[w]
        + lp_params.ft_bonus * lp_variables.free_transfers[w]
        + lp_params.itb_bonus * lp_variables.in_the_bank[w]
        for w in fpl_data.gameweeks
    }

    decay_objective = lpSum(
        [
            gw_total[w] * pow(lp_params.decay, i)
            for i, w in enumerate(fpl_data.gameweeks)
        ]
    )
    model += -decay_objective, "total_decay_xp"

    return None


def construct_lp(
    fpl_data: FplData, parameters: dict
) -> tuple[LpParams, LpVariables, LpKeys, VariableSums]:
    mps_path = parameters["mps_path"]

    lp_params = prepare_lp_params(fpl_data, parameters)
    lp_keys = prepare_lp_keys(fpl_data, lp_params)
    lp_variables = initialize_variables(fpl_data, lp_keys)
    variable_sums = sum_lp_variables(fpl_data, lp_params, lp_keys, lp_variables)

    model = LpProblem(name="fpl_model", sense=LpMinimize)
    add_constraints(fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums)
    add_objective_function(
        fpl_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )

    model.writeMPS(mps_path)

    backup_latest_n(mps_path, n=5)

    return lp_params, lp_keys, lp_variables, variable_sums
