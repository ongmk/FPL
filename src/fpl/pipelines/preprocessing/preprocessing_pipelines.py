from kedro.pipeline import Pipeline, node

from fpl.pipelines.preprocessing.data_test import data_tests
from fpl.pipelines.preprocessing.elo_calculation import calculate_elo_score
from fpl.pipelines.preprocessing.feature_engineering import feature_engineering
from fpl.pipelines.preprocessing.imputer import impute_missing_values
from fpl.pipelines.preprocessing.preprocessor import (
    align_data_structure,
    combine_data,
    split_data,
)

preprocessing_pipeline = Pipeline(
    [
        # node(
        #     func=data_tests,
        #     inputs=[
        #         "PLAYER_NAME_MAPPING",
        #         "FPL_DATA",
        #         "FPL2FBREF_TEAM_MAPPING",
        #         "TEAM_MATCH_LOG",
        #     ],
        #     outputs="check_complete",
        # ),
        # node(
        #     func=align_data_structure,
        #     inputs=[
        #         "check_complete",
        #         "PLAYER_MATCH_LOG",
        #         "TEAM_MATCH_LOG",
        #         "FPL_DATA",
        #         "PLAYER_NAME_MAPPING",
        #         "FPL2FBREF_TEAM_MAPPING",
        #     ],
        #     outputs=[
        #         "aligned_player_match_log",
        #         "aligned_team_match_log",
        #         "aligned_fpl_data",
        #     ],
        # ),
        # node(
        #     func=calculate_elo_score,
        #     inputs=[
        #         "aligned_team_match_log",
        #         "aligned_fpl_data",
        #         "READ_ELO_DATA",
        #         "params:preprocessing",
        #     ],
        #     outputs="ELO_DATA",
        # ),
        # node(
        #     func=combine_data,
        #     inputs=[
        #         "aligned_player_match_log",
        #         "aligned_team_match_log",
        #         "ELO_DATA",
        #         "aligned_fpl_data",
        #         "params:preprocessing",
        #     ],
        #     outputs="combined_data",
        # ),
        # node(
        #     func=feature_engineering,
        #     inputs=[
        #         "combined_data",
        #         "READ_PROCESSED_DATA",
        #         "params:preprocessing",
        #     ],
        #     outputs="INTERMEDIATE_DATA",
        # ),
        # node(
        #     func=impute_missing_values,
        #     inputs=[
        #         "INTERMEDIATE_DATA",
        #         "params:preprocessing",
        #         "params:points_prediction",
        #     ],
        #     outputs="PROCESSED_DATA",
        # ),
        # node(
        #     func=split_data,
        #     inputs=[
        #         "PROCESSED_DATA",
        #         "params:preprocessing",
        #         "params:points_prediction",
        #     ],
        #     outputs=[
        #         "TRAIN_VAL_DATA",
        #         "HOLDOUT_DATA",
        #     ],
        # ),
    ]
)
