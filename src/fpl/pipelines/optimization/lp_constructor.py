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
    LpData,
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.utils import backup_latest_n


def prepare_lp_params(lp_data: LpData, parameters: dict[str, Any]) -> LpParams:
    wildcard_week = parameters["wildcard_week"]
    bench_boost_week = parameters["bench_boost_week"]
    free_hit_week = parameters["free_hit_week"]
    triple_captain_week = parameters["triple_captain_week"]
    transfer_horizon = parameters["transfer_horizon"]
    next_gw = lp_data.gameweeks[0]
    transfer_gws = lp_data.gameweeks[:transfer_horizon]

    return LpParams(
        next_gw=next_gw,
        transfer_gws=transfer_gws,
        threshold_gw=2 if next_gw == 1 else next_gw,
        horizon=parameters["horizon"],
        wildcard_week=wildcard_week if wildcard_week in transfer_gws else None,
        bench_boost_week=bench_boost_week if bench_boost_week in transfer_gws else None,
        free_hit_week=free_hit_week if free_hit_week in transfer_gws else None,
        triple_captain_week=(
            triple_captain_week if triple_captain_week in transfer_gws else None
        ),
        decay=parameters["decay"],
        free_transfer_bonus=parameters["free_transfer_bonus"],
        in_the_bank_bonus=parameters["in_the_bank_bonus"],
        bench_weights=parameters["bench_weights"],
    )


def prepare_lp_keys(lp_data: LpData, lp_params: LpParams) -> LpKeys:
    players = lp_data.merged_data.index.to_list()
    all_gws = [lp_params.next_gw - 1] + lp_data.gameweeks
    order = [0, 1, 2, 3]
    price_modified_players = lp_data.merged_data.loc[
        lp_data.merged_data["sell_price"] != lp_data.merged_data["now_cost"]
    ].index.to_list()

    return LpKeys(
        element_types=lp_data.type_data.index.to_list(),
        teams=lp_data.team_data["name"].to_list(),
        players=players,
        price_modified_players=price_modified_players,
        all_gws=all_gws,
        player_all_gws=list(itertools.product(players, all_gws)),
        player_gameweeks=list(itertools.product(players, lp_data.gameweeks)),
        order=order,
        player_gameweeks_order=list(
            itertools.product(players, lp_data.gameweeks, order)
        ),
        price_modified_players_gameweeks=list(
            itertools.product(price_modified_players, lp_data.gameweeks)
        ),
        player_type=lp_data.merged_data["element_type"].to_dict(),
        sell_price=(lp_data.merged_data["sell_price"] / 10).to_dict(),
        buy_price=(lp_data.merged_data["now_cost"] / 10).to_dict(),
    )


def initialize_variables(lp_data: LpData, lp_keys: LpKeys) -> LpVariables:
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
        for w in lp_data.gameweeks
    }
    in_the_bank = LpVariable.dicts(
        "in_the_bank", lp_keys.all_gws, cat=LpContinuous, lowBound=0
    )
    free_transfers = LpVariable.dicts(
        "free_transfers",
        lp_keys.all_gws + [lp_keys.all_gws[-1] + 1],
        cat=LpInteger,
        lowBound=0,
        upBound=2,
    )
    penalized_transfers = LpVariable.dicts(
        "penalized_transfers", lp_data.gameweeks, cat=LpInteger, lowBound=0
    )

    use_wildcard = LpVariable.dicts("use_wildcard", lp_data.gameweeks, cat=LpBinary)
    use_bench_boost = LpVariable.dicts(
        "use_bench_boost", lp_data.gameweeks, cat=LpBinary
    )
    use_triple_captain = LpVariable.dicts(
        "use_triple_captain", lp_keys.player_gameweeks, cat=LpBinary
    )
    use_free_hit = LpVariable.dicts("use_free_hit", lp_data.gameweeks, cat=LpBinary)
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
        use_wildcard=use_wildcard,
        use_bench_boost=use_bench_boost,
        use_free_hit=use_free_hit,
        use_triple_captain=use_triple_captain,
    )


def sum_lp_variables(
    lp_data: LpData, lp_params: LpParams, lp_keys: LpKeys, lp_variables: LpVariables
) -> VariableSums:
    lineup_type_count = {
        (t, w): lpSum(
            lp_variables.lineup[p, w]
            for p in lp_keys.players
            if lp_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in lp_data.gameweeks
    }
    squad_type_count = {
        (t, w): lpSum(
            lp_variables.squad[p, w]
            for p in lp_keys.players
            if lp_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in lp_data.gameweeks
    }
    squad_free_hit_type_count = {
        (t, w): lpSum(
            lp_variables.squad_free_hit[p, w]
            for p in lp_keys.players
            if lp_data.merged_data.loc[p, "element_type"] == t
        )
        for t in lp_keys.element_types
        for w in lp_data.gameweeks
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
        for w in lp_data.gameweeks
    }
    free_hit_sell_price = {
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
        for w in lp_data.gameweeks
    }
    points_player_week = {
        (p, w): lp_data.merged_data.loc[p, f"xPts_{w}"]
        for p in lp_keys.players
        for w in lp_data.gameweeks
    }
    squad_count = {
        w: lpSum(lp_variables.squad[p, w] for p in lp_keys.players)
        for w in lp_data.gameweeks
    }
    squad_free_hit_count = {
        w: lpSum(lp_variables.squad_free_hit[p, w] for p in lp_keys.players)
        for w in lp_data.gameweeks
    }
    number_of_transfers = {
        w: lpSum([lp_variables.transfer_out[p, w] for p in lp_keys.players])
        for w in lp_data.gameweeks
    }
    number_of_transfers[lp_params.next_gw - 1] = 1
    transfer_diff = {
        w: number_of_transfers[w]
        - lp_variables.free_transfers[w]
        - 15 * lp_variables.use_wildcard[w]
        for w in lp_data.gameweeks
    }
    use_triple_captain_week = {
        w: lpSum(lp_variables.use_triple_captain[p, w] for p in lp_keys.players)
        for w in lp_data.gameweeks
    }

    return VariableSums(
        lineup_type_count=lineup_type_count,
        squad_type_count=squad_type_count,
        squad_free_hit_type_count=squad_free_hit_type_count,
        sold_amount=sold_amount,
        free_hit_sell_price=free_hit_sell_price,
        bought_amount=bought_amount,
        points_player_week=points_player_week,
        squad_count=squad_count,
        squad_free_hit_count=squad_free_hit_count,
        number_of_transfers=number_of_transfers,
        transfer_diff=transfer_diff,
        use_triple_captain_week=use_triple_captain_week,
    )


def add_constraints(lp_data, model, lp_params, lp_keys, lp_variables, variable_sums):
    ChipConstraints.global_level(
        lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )
    TransferConstraints.global_level(
        lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )
    for p in lp_keys.players:
        SquadConstraints.player_level(
            p, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )
        TransferConstraints.player_level(
            p, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )

    for w in lp_data.gameweeks:
        SquadConstraints.gameweek_level(
            w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )
        TransferConstraints.gameweek_level(
            w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
        )

        for t in lp_keys.element_types:
            SquadConstraints.type_gameweek_level(
                t, w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )

        for t in lp_keys.teams:
            SquadConstraints.team_gameweek_level(
                t, w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )

        for p in lp_keys.players:
            SquadConstraints.player_gameweek_level(
                p, w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )
            TransferConstraints.player_gameweek_level(
                p, w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )
            ChipConstraints.player_gameweek_level(
                p, w, lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
            )


def add_objective_function(
    lp_data: LpData,
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
                    + lp_variables.use_triple_captain[p, w]
                    + lpSum(
                        lp_params.bench_weights[o] * lp_variables.bench[p, w, o]
                        for o in lp_keys.order
                    )
                )
                for p in lp_keys.players
            ]
        )
        for w in lp_data.gameweeks
    }
    gw_total = {
        w: gw_xp[w]
        - 4 * lp_variables.penalized_transfers[w]
        + lp_params.free_transfer_bonus * lp_variables.free_transfers[w] * int(w != 38)
        + lp_params.in_the_bank_bonus * lp_variables.in_the_bank[w] * int(w != 38)
        for w in lp_data.gameweeks
    }

    decay_objective = lpSum(
        [gw_total[w] * pow(lp_params.decay, i) for i, w in enumerate(lp_data.gameweeks)]
    )
    model += -decay_objective, "total_decay_xp"

    return None


def construct_lp(lp_data: LpData, parameters: dict) -> tuple[LpVariables, VariableSums]:
    model_name = parameters["model_name"]
    mps_dir = parameters["mps_dir"]
    mps_path = f"{mps_dir}/{model_name}.mps"

    lp_params = prepare_lp_params(lp_data, parameters)
    lp_keys = prepare_lp_keys(lp_data, lp_params)
    lp_variables = initialize_variables(lp_data, lp_keys)
    variable_sums = sum_lp_variables(lp_data, lp_params, lp_keys, lp_variables)

    model = LpProblem(name=model_name, sense=LpMinimize)
    add_constraints(lp_data, model, lp_params, lp_keys, lp_variables, variable_sums)
    add_objective_function(
        lp_data, model, lp_params, lp_keys, lp_variables, variable_sums
    )

    model.writeMPS(mps_path)

    backup_latest_n(mps_path, n=5)

    return lp_params, lp_keys, lp_variables, variable_sums
