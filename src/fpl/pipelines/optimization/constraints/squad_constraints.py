from pulp import LpProblem, lpSum

from fpl.pipelines.optimization.constraints.base_constraints import BaseConstraints
from fpl.pipelines.optimization.data_classes import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.pipelines.optimization.fpl_api import FplData


class SquadConstraints(BaseConstraints):
    def player_level(
        player: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        if player in fpl_data.initial_squad:
            model += (
                lp_variables.squad[player, lp_params.next_gw - 1] == 1,
                f"initial_squad_players_{player}",
            )
        else:
            model += (
                lp_variables.squad[player, lp_params.next_gw - 1] == 0,
                f"initial_squad_others_{player}",
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
        model += variable_sums.squad_count[gameweek] == 15, f"squad_count_{gameweek}"
        model += (
            variable_sums.squad_free_hit_count[gameweek]
            == 15 * lp_variables.use_free_hit[gameweek],
            f"squad_free_hit_count_{gameweek}",
        )
        model += (
            lpSum([lp_variables.lineup[p, gameweek] for p in lp_keys.players])
            == 11 + 4 * lp_variables.use_bench_boost[gameweek],
            f"lineup_count_{gameweek}",
        )
        model += (
            lpSum(
                lp_variables.bench[p, gameweek, 0]
                for p in lp_keys.players
                if lp_keys.player_type[p] == 1
            )
            == 1 - lp_variables.use_bench_boost[gameweek],
            f"bench_gk_{gameweek}",
        )
        for o in [1, 2, 3]:
            model += (
                lpSum(lp_variables.bench[p, gameweek, o] for p in lp_keys.players)
                == 1 - lp_variables.use_bench_boost[gameweek],
                f"bench_count_{gameweek}_{o}",
            )
        model += (
            lpSum([lp_variables.captain[p, gameweek] for p in lp_keys.players]) == 1,
            f"captain_count_{gameweek}",
        )
        model += (
            lpSum([lp_variables.vicecap[p, gameweek] for p in lp_keys.players]) == 1,
            f"vicecap_count_{gameweek}",
        )

    def type_gameweek_level(
        _type: int,
        gameweek: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            variable_sums.lineup_type_count[_type, gameweek]
            >= fpl_data.type_data.loc[_type, "squad_min_play"],
            f"valid_formation_lb_{_type}_{gameweek}",
        )
        model += (
            variable_sums.lineup_type_count[_type, gameweek]
            <= fpl_data.type_data.loc[_type, "squad_max_play"]
            + lp_variables.use_bench_boost[gameweek],
            f"valid_formation_ub_{_type}_{gameweek}",
        )
        model += (
            variable_sums.squad_type_count[_type, gameweek]
            == fpl_data.type_data.loc[_type, "squad_select"],
            f"valid_squad_{_type}_{gameweek}",
        )
        model += (
            variable_sums.squad_free_hit_type_count[_type, gameweek]
            == fpl_data.type_data.loc[_type, "squad_select"]
            * lp_variables.use_free_hit[gameweek],
            f"valid_squad_free_hit_{_type}_{gameweek}",
        )

    def team_gameweek_level(
        team: int,
        gameweek: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        model += (
            lpSum(
                lp_variables.squad[p, gameweek]
                for p in lp_keys.players
                if fpl_data.merged_data.loc[p, "team"] == team
            )
            <= 3,
            f"team_limit_{team}_{gameweek}",
        )
        model += (
            lpSum(
                lp_variables.squad_free_hit[p, gameweek]
                for p in lp_keys.players
                if fpl_data.merged_data.loc[p, "team"] == team
            )
            <= 3,
            f"team_limit_fh_{team}_{gameweek}",
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
            lp_variables.lineup[player, gameweek]
            <= lp_variables.squad[player, gameweek]
            + lp_variables.use_free_hit[gameweek],
            f"lineup_squad_rel_{player}_{gameweek}",
        )
        model += (
            lp_variables.lineup[player, gameweek]
            <= lp_variables.squad_free_hit[player, gameweek]
            + 1
            - lp_variables.use_free_hit[gameweek],
            f"lineup_squad_free_hit_rel_{player}_{gameweek}",
        )
        for o in lp_keys.order:
            model += (
                lp_variables.bench[player, gameweek, o]
                <= lp_variables.squad[player, gameweek]
                + lp_variables.use_free_hit[gameweek],
                f"bench_squad_rel_{player}_{gameweek}_{o}",
            )
            model += (
                lp_variables.bench[player, gameweek, o]
                <= lp_variables.squad_free_hit[player, gameweek]
                + 1
                - lp_variables.use_free_hit[gameweek],
                f"bench_squad_free_hit_rel_{player}_{gameweek}_{o}",
            )
        model += (
            lp_variables.captain[player, gameweek]
            <= lp_variables.lineup[player, gameweek],
            f"captain_lineup_rel_{player}_{gameweek}",
        )
        model += (
            lp_variables.vicecap[player, gameweek]
            <= lp_variables.lineup[player, gameweek],
            f"vicecap_lineup_rel_{player}_{gameweek}",
        )
        model += (
            lp_variables.captain[player, gameweek]
            + lp_variables.vicecap[player, gameweek]
            <= 1,
            f"cap_vc_rel_{player}_{gameweek}",
        )
        model += (
            lp_variables.lineup[player, gameweek]
            + lpSum(lp_variables.bench[player, gameweek, o] for o in lp_keys.order)
            <= 1,
            f"lineup_bench_rel_{player}_{gameweek}_{o}",
        )
