from pulp import LpProblem

from fpl.pipelines.optimization.data_classes import (
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)
from fpl.pipelines.optimization.fpl_api import FplData


class BaseConstraints:

    def global_level(
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass

    def gameweek_level(
        gameweek: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass

    def player_level(
        player: int,
        fpl_data: FplData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass

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
        pass

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
        pass

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
        pass
