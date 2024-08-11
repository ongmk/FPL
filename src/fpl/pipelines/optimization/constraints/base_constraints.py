from pulp import LpProblem

from fpl.pipelines.optimization.data_classes import (
    LpData,
    LpKeys,
    LpParams,
    LpVariables,
    VariableSums,
)


class BaseConstraints:

    def global_level(
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass

    def gameweek_level(
        gameweek: int,
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass

    def player_level(
        player: int,
        lp_data: LpData,
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
        lp_data: LpData,
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
        lp_data: LpData,
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
        lp_data: LpData,
        model: LpProblem,
        lp_params: LpParams,
        lp_keys: LpKeys,
        lp_variables: LpVariables,
        variable_sums: VariableSums,
    ) -> None:
        pass
