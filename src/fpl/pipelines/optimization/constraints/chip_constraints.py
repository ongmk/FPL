from pulp import LpProblem, lpSum

from fpl.pipelines.optimization.constraints.base_constraints import BaseConstraints
from fpl.pipelines.optimization.data_classes import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.pipelines.optimization.fpl_api import FplData


class ChipConstraints(BaseConstraints):
    def global_level(
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lpSum(lp_variables.use_wc[w] for w in fpl_data.gameweeks)
            <= lp_params.wc_limit,
            "use_wc_limit",
        )
        model += (
            lpSum(lp_variables.use_bb[w] for w in fpl_data.gameweeks)
            <= lp_params.bb_limit,
            "use_bb_limit",
        )
        model += (
            lpSum(lp_variables.use_fh[w] for w in fpl_data.gameweeks)
            <= lp_params.fh_limit,
            "use_fh_limit",
        )
        if lp_params.wc_on is not None:
            model += lp_variables.use_wc[lp_params.wc_on] == 1, "force_wc"
        if lp_params.bb_on is not None:
            model += lp_variables.use_bb[lp_params.bb_on] == 1, "force_bb"
        if lp_params.fh_on is not None:
            model += lp_variables.use_fh[lp_params.fh_on] == 1, "force_fh"

    def gameweek_level(
        gameweek: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lp_variables.use_wc[gameweek]
            + lp_variables.use_fh[gameweek]
            + lp_variables.use_bb[gameweek]
            <= 1,
            f"single_chip_{gameweek}",
        )
        if gameweek > lp_params.next_gw:
            model += (
                lp_variables.aux[gameweek] <= 1 - lp_variables.use_wc[gameweek - 1],
                f"ft_after_wc_{gameweek}",
            )
            model += (
                lp_variables.aux[gameweek] <= 1 - lp_variables.use_fh[gameweek - 1],
                f"ft_after_fh_{gameweek}",
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
            lp_variables.squad_fh[player, gameweek] <= lp_variables.use_fh[gameweek],
            f"fh_squad_logic_{player}_{gameweek}",
        )
