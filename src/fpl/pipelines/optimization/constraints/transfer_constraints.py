from pulp import LpBinary, LpInteger, LpProblem, LpVariable, lpSum

from fpl.pipelines.optimization.constraints.base_constraints import BaseConstraints
from fpl.pipelines.optimization.data_classes import (
    LpData,
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)


def linearize_min_function(X, x1, x2, M, idx):
    """
    X = min(x1, x2)
    Let M be a constant such that x1,x2 <= M in any "reasonable" solution to the problem.
    """
    aux = LpVariable(f"min_aux_{idx}", cat=LpBinary)

    constraints = []

    constraints.append(X <= x1)
    constraints.append(X <= x2)
    constraints.append(X >= x1 - M * (1 - aux))
    constraints.append(X >= x2 - M * aux)
    return constraints


def linearize_max_function(X, x1, x2, M, idx):
    """
    X = max(x1, x2)
    Let M be a constant such that x1,x2 <= M in any "reasonable" solution to the problem.
    """
    aux = LpVariable(f"max_aux_{idx}", cat=LpBinary)

    constraints = []
    constraints.append(X >= x1)
    constraints.append(X >= x2)
    constraints.append(X <= x1 + M * (1 - aux))
    constraints.append(X <= x2 + M * aux)
    return constraints


def linearize_if_else_function(X, x1, x2, M, flag):
    """
    X = x1 if flag = 1 else x2
    Let M be a constant such that x1,x2 <= M in any "reasonable" solution to the problem.
    """

    constraints = []
    constraints.append(X >= x1 - M * (1 - flag))
    constraints.append(X <= x1 + M * (1 - flag))
    constraints.append(X >= x2 - M * flag)
    constraints.append(X <= x2 + M * flag)
    return constraints


def add_free_transfer_constraints(
    week: int, lp_variables: LpVariables, variable_sums: VariableSums, model: LpProblem
):
    """
    F2 = min(max(F1-T+1, 1), 5)
    """

    non_chip_transfers = LpVariable(
        f"non_chip_transfers_{week}", cat=LpInteger, lowBound=0, upBound=15
    )
    if_else_constraints = linearize_if_else_function(
        X=non_chip_transfers,
        x1=1,
        x2=variable_sums.number_of_transfers[week],
        M=15,
        flag=lp_variables.use_free_hit[week]
        + lp_variables.use_wildcard[week]
        + lp_variables.use_bench_boost[week],
    )
    for idx, constraint in enumerate(if_else_constraints):
        model += constraint, f"if_else_constraint{idx}_{week}"

    tmp = LpVariable(f"tmp_{week}", cat=LpInteger, lowBound=1, upBound=6)
    max_constraints = linearize_max_function(
        X=tmp,
        x1=(lp_variables.free_transfers[week - 1] - non_chip_transfers + 1),
        x2=1,
        M=6,
        idx=week,
    )
    for idx, constraint in enumerate(max_constraints):
        model += constraint, f"max_constraint{idx}_{week}"

    min_constraints = linearize_min_function(
        X=lp_variables.free_transfers[week],
        x1=tmp,
        x2=5,
        M=6,
        idx=week,
    )
    for idx, constraint in enumerate(min_constraints):
        model += constraint, f"min_constraint{idx}_{week}"
    return None


class TransferConstraints(BaseConstraints):

    def global_level(
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lp_variables.in_the_bank[lp_params.next_gw - 1] == lp_data.in_the_bank,
            "initial_in_the_bank",
        )
        model += (
            lp_variables.free_transfers[lp_params.next_gw - 1]
            == lp_data.free_transfers,
            "initial_free_transfers",
        )

        model += (
            lpSum(
                lp_variables.transfer_in[p, w] + lp_variables.transfer_out[p, w]
                for p in lp_keys.players
                for w in lp_data.gameweeks
                if w not in lp_params.transfer_gws
            )
            == 0,
            f"no_transfer",
        )

    def player_level(
        player: int,
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        # Multiple-sell fix
        if player in lp_keys.price_modified_players:
            model += (
                lpSum(
                    lp_variables.transfer_out_first[player, w]
                    for w in lp_data.gameweeks
                )
                <= 1,
                f"multi_sell_3_{player}",
            )

    def gameweek_level(
        gameweek: int,
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        if lp_params.next_gw == 1 and gameweek < lp_params.threshold_gw:
            model += (
                lp_variables.free_transfers[gameweek] == 1,
                f"preseason_initial_free_transfers_{gameweek}",
            )

        if gameweek >= lp_params.threshold_gw:
            add_free_transfer_constraints(gameweek, lp_variables, variable_sums, model)
        model += (
            lp_variables.penalized_transfers[gameweek]
            >= variable_sums.transfer_diff[gameweek],
            f"penalized_transfers_relation_{gameweek}",
        )
        # Transfer constraints
        model += (
            lp_variables.in_the_bank[gameweek]
            == lp_variables.in_the_bank[gameweek - 1]
            + variable_sums.sold_amount[gameweek]
            - variable_sums.bought_amount[gameweek],
            f"continuous_budget_{gameweek}",
        )
        model += (
            lpSum(
                variable_sums.free_hit_sell_price[p]
                * lp_variables.squad[p, gameweek - 1]
                for p in lp_keys.players
            )
            + lp_variables.in_the_bank[gameweek - 1]
            >= lpSum(
                variable_sums.free_hit_sell_price[p]
                * lp_variables.squad_free_hit[p, gameweek]
                for p in lp_keys.players
            ),
            f"free_hit_budget_{gameweek}",
        )

    def player_gameweek_level(
        player: int,
        gameweek: int,
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lp_variables.squad[player, gameweek]
            == lp_variables.squad[player, gameweek - 1]
            + lp_variables.transfer_in[player, gameweek]
            - lp_variables.transfer_out[player, gameweek],
            f"squad_transfer_relation_{player}_{gameweek}",
        )
        model += (
            lp_variables.transfer_in[player, gameweek]
            <= 1 - lp_variables.use_free_hit[gameweek],
            f"no_transfer_in_free_hit_{player}_{gameweek}",
        )
        model += (
            lp_variables.transfer_out[player, gameweek]
            <= 1 - lp_variables.use_free_hit[gameweek],
            f"no_transfer_out_free_hit_{player}_{gameweek}",
        )

        # Multiple-sell fix
        if player in lp_keys.price_modified_players:
            model += (
                lp_variables.transfer_out_first[player, gameweek]
                + lp_variables.transfer_out_regular[player, gameweek]
                <= 1,
                f"multi_sell_1_{player}_{gameweek}",
            )
            model += (
                lp_params.horizon
                * lpSum(
                    lp_variables.transfer_out_first[player, wbar]
                    for wbar in lp_data.gameweeks
                    if wbar <= gameweek
                )
                >= lpSum(
                    lp_variables.transfer_out_regular[player, wbar]
                    for wbar in lp_data.gameweeks
                    if wbar >= gameweek
                ),
                f"multi_sell_2_{player}_{gameweek}",
            )

        # Transfer in/out fix
        model += (
            lp_variables.transfer_in[player, gameweek]
            + lp_variables.transfer_out[player, gameweek]
            <= 1,
            f"transfer_in_out_limit_{player}_{gameweek}",
        )
