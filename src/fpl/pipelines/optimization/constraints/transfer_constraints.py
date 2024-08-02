from pulp import LpProblem, lpSum

from fpl.pipelines.optimization.constraints.base_constraints import BaseConstraints
from fpl.pipelines.optimization.data_classes import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.pipelines.optimization.fpl_api import FplData


class TransferConstraints(BaseConstraints):

    def global_level(
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lp_variables.in_the_bank[lp_params.next_gw - 1] == fpl_data.itb,
            "initial_itb",
        )
        model += (
            lp_variables.free_transfers[lp_params.next_gw] == lp_params.ft,
            "initial_ft",
        )

        if lp_params.next_gw == 1 and lp_params.threshold_gw in fpl_data.gameweeks:
            model += (
                lp_variables.free_transfers[lp_params.threshold_gw] == lp_params.ft,
                "ps_initial_ft",
            )
        model += (
            lpSum(
                lp_variables.transfer_in[p, w] + lp_variables.transfer_out[p, w]
                for p in lp_keys.players
                for w in fpl_data.gameweeks
                if w not in lp_params.transfer_gws
            )
            == 0,
            f"no_transfer",
        )

    def player_level(
        player: int,
        fpl_data: FplData,
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
                    for w in fpl_data.gameweeks
                )
                <= 1,
                f"multi_sell_3_{player}",
            )

    def gameweek_level(
        gameweek: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:

        if gameweek > lp_params.next_gw:
            model += (
                lp_variables.free_transfers[gameweek] >= 1,
                f"future_ft_limit_{gameweek}",
            )
        # Free transfer constraints
        if gameweek > lp_params.threshold_gw:
            model += (
                lp_variables.free_transfers[gameweek] == lp_variables.aux[gameweek] + 1,
                f"aux_ft_rel_{gameweek}",
            )
            model += (
                lp_variables.free_transfers[gameweek - 1]
                - variable_sums.number_of_transfers[gameweek - 1]
                - 2 * lp_variables.use_wc[gameweek - 1]
                - 2 * lp_variables.use_fh[gameweek - 1]
                <= 2 * lp_variables.aux[gameweek],
                f"force_aux_1_{gameweek}",
            )
            model += (
                lp_variables.free_transfers[gameweek - 1]
                - variable_sums.number_of_transfers[gameweek - 1]
                - 2 * lp_variables.use_wc[gameweek - 1]
                - 2 * lp_variables.use_fh[gameweek - 1]
                >= lp_variables.aux[gameweek]
                + (-14) * (1 - lp_variables.aux[gameweek]),
                f"force_aux_2_{gameweek}",
            )
        model += (
            lp_variables.penalized_transfers[gameweek]
            >= variable_sums.transfer_diff[gameweek],
            f"pen_transfer_rel_{gameweek}",
        )
        # Transfer constraints
        model += (
            lp_variables.in_the_bank[gameweek]
            == lp_variables.in_the_bank[gameweek - 1]
            + variable_sums.sold_amount[gameweek]
            - variable_sums.bought_amount[gameweek],
            f"cont_budget_{gameweek}",
        )
        model += (
            lpSum(
                variable_sums.fh_sell_price[p] * lp_variables.squad[p, gameweek - 1]
                for p in lp_keys.players
            )
            + lp_variables.in_the_bank[gameweek - 1]
            >= lpSum(
                variable_sums.fh_sell_price[p] * lp_variables.squad_fh[p, gameweek]
                for p in lp_keys.players
            ),
            f"fh_budget_{gameweek}",
        )

    def player_gameweek_level(
        player: int,
        gameweek: int,
        fpl_data: FplData,
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
            f"squad_transfer_rel_{player}_{gameweek}",
        )
        model += (
            lp_variables.transfer_in[player, gameweek]
            <= 1 - lp_variables.use_fh[gameweek],
            f"no_tr_in_fh_{player}_{gameweek}",
        )
        model += (
            lp_variables.transfer_out[player, gameweek]
            <= 1 - lp_variables.use_fh[gameweek],
            f"no_tr_out_fh_{player}_{gameweek}",
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
                    for wbar in fpl_data.gameweeks
                    if wbar <= gameweek
                )
                >= lpSum(
                    lp_variables.transfer_out_regular[player, wbar]
                    for wbar in fpl_data.gameweeks
                    if wbar >= gameweek
                ),
                f"multi_sell_2_{player}_{gameweek}",
            )

        # Transfer in/out fix
        model += (
            lp_variables.transfer_in[player, gameweek]
            + lp_variables.transfer_out[player, gameweek]
            <= 1,
            f"tr_in_out_limit_{player}_{gameweek}",
        )
