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
        # Do not let LP decide when to use chips
        wildcard_limit = bench_boost_limit = free_hit_limit = triple_captain_limit = 0
        if lp_params.wildcard_week in fpl_data.gameweeks:
            wildcard_limit = 1
            model += (
                lp_variables.use_wildcard[lp_params.wildcard_week] == 1,
                "force_wildcard",
            )
        if lp_params.bench_boost_week in fpl_data.gameweeks:
            bench_boost_limit = 1
            model += (
                lp_variables.use_bench_boost[lp_params.bench_boost_week] == 1,
                "force_bench_boost",
            )
        if lp_params.free_hit_week in fpl_data.gameweeks:
            free_hit_limit = 1
            model += (
                lp_variables.use_free_hit[lp_params.free_hit_week] == 1,
                "force_free_hit",
            )
        if lp_params.triple_captain_week in fpl_data.gameweeks:
            triple_captain_limit = 1
            model += (
                variable_sums.use_triple_captain_week[lp_params.triple_captain_week]
                == 1,
                "force_triple_captain",
            )

        model += (
            lpSum(lp_variables.use_wildcard[w] for w in fpl_data.gameweeks)
            <= wildcard_limit,
            "wildcard_limit",
        )
        model += (
            lpSum(lp_variables.use_bench_boost[w] for w in fpl_data.gameweeks)
            <= bench_boost_limit,
            "bench_boost_limit",
        )
        model += (
            lpSum(lp_variables.use_free_hit[w] for w in fpl_data.gameweeks)
            <= free_hit_limit,
            "free_hit_limit",
        )
        model += (
            lpSum(variable_sums.use_triple_captain_week[w] for w in fpl_data.gameweeks)
            <= triple_captain_limit,
            "triple_captain_limit",
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
        model += (
            lp_variables.use_wildcard[gameweek]
            + lp_variables.use_free_hit[gameweek]
            + lp_variables.use_bench_boost[gameweek]
            + lp_variables.use_triple_captain[gameweek]
            <= 1,
            f"single_chip_{gameweek}",
        )
        if gameweek > lp_params.next_gw:
            model += (
                lp_variables.aux[gameweek]
                <= 1 - lp_variables.use_wildcard[gameweek - 1],
                f"ft_after_wc_{gameweek}",
            )
            model += (
                lp_variables.aux[gameweek]
                <= 1 - lp_variables.use_free_hit[gameweek - 1],
                f"ft_after_free_hit_{gameweek}",
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
            lp_variables.squad_free_hit[player, gameweek]
            <= lp_variables.use_free_hit[gameweek],
            f"free_hit_squad_logic_{player}_{gameweek}",
        )
