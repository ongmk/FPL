from kedro.pipeline import Pipeline, node, pipeline

from fpl.pipelines.preprocess_pipeline.elo_calculation import calculate_elo_score
from fpl.pipelines.preprocess_pipeline.feature_engineering import feature_engineering
from fpl.pipelines.preprocess_pipeline.imputer import impute_missing_values
from fpl.pipelines.preprocess_pipeline.preprocessor import (
    align_data_structure,
    combine_data,
    data_checks,
    split_data,
)


def create_preprocess_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=data_checks,
                inputs=["PLAYER_NAME_MAPPING", "FPL_DATA", "FPL2FBREF_TEAM_MAPPING"],
                outputs="check_complete",
            ),
            node(
                func=align_data_structure,
                inputs=[
                    "check_complete",
                    "PLAYER_MATCH_LOG",
                    "TEAM_MATCH_LOG",
                    "FPL_DATA",
                    "PLAYER_NAME_MAPPING",
                    "FPL2FBREF_TEAM_MAPPING",
                ],
                outputs=[
                    "aligned_player_match_log",
                    "aligned_team_match_log",
                    "aligned_fpl_data",
                ],
            ),
            node(
                func=calculate_elo_score,
                inputs=[
                    "check_complete",
                    "aligned_team_match_log",
                    "aligned_fpl_data",
                    "READ_ELO_DATA",
                    "params:preprocess",
                ],
                outputs="ELO_DATA",
            ),
            node(
                func=combine_data,
                inputs=[
                    "aligned_player_match_log",
                    "aligned_team_match_log",
                    "ELO_DATA",
                    "aligned_fpl_data",
                    "params:preprocess",
                ],
                outputs="combined_data",
            ),
            node(
                func=feature_engineering,
                inputs=[
                    "combined_data",
                    "READ_PROCESSED_DATA",
                    "params:preprocess",
                ],
                outputs="INTERMEDIATE_DATA",
            ),
            node(
                func=impute_missing_values,
                inputs=["INTERMEDIATE_DATA", "params:preprocess"],
                outputs="PROCESSED_DATA",
            ),
            node(
                func=split_data,
                inputs=["PROCESSED_DATA", "params:preprocess", "params:model"],
                outputs=[
                    "TRAIN_VAL_DATA",
                    "HOLDOUT_DATA",
                ],
            ),
        ]
    )


# if __name__ == "__main__":
#     import sqlite3
#     import pandas as pd
#     import yaml

#     # Connect to the SQLite database
#     connection = sqlite3.connect("./data/fpl.db")
#     train_val_data = pd.read_sql_query("SELECT * FROM train_val_data", connection)
#     with open("./conf/base/parameters.yml", "r") as file:
#         parameters = yaml.safe_load(file)
#         parameters = parameters["model"]

#     sklearn_pipeline = create_sklearn_pipeline(
#         train_val_data=train_val_data, parameters=parameters
#     )
#     models = model_selection(parameters)
#     scores = cross_validation(train_val_data, models, sklearn_pipeline, parameters)
#     trained_models = train_model(
#         train_val_data,
#         models,
#         sklearn_pipeline,
#         parameters,
#     )

#     holdout_data = pd.read_sql_query("SELECT * FROM holdout_data", connection)
#     outputs = evaluate_model(
#         holdout_data=holdout_data,
#         models=trained_models,
#         sklearn_pipeline=sklearn_pipeline,
#         experiment_id=1,
#         start_time="now",
#         parameters=parameters,
#     )
#     print(outputs)
