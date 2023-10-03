from kedro.pipeline import Pipeline, node, pipeline
from src.fpl.pipelines.model_pipeline.experiment_helpers import (
    init_experiment,
    run_housekeeping,
)
from src.fpl.pipelines.model_pipeline.modelling.ensemble import model_selection
from src.fpl.pipelines.model_pipeline.modelling.evaluation import evaluate_model_holdout
from src.fpl.pipelines.model_pipeline.modelling.training import (
    create_sklearn_pipeline,
    cross_validation,
    pycaret_compare_models,
    train_model,
)
from src.fpl.pipelines.model_pipeline.preprocessing.elo_calculation import (
    calculate_elo_score,
)
from src.fpl.pipelines.model_pipeline.preprocessing.preprocessor import (
    clean_data,
    feature_engineering,
    fuzzy_match_player_names,
    impute_missing_values,
    split_data,
)


def create_preprocess_pipeline() -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=calculate_elo_score,
            #     inputs=["TEAM_MATCH_LOG", "params:data"],
            #     outputs="ELO_DATA",
            # ),
            # node(
            #     func=fuzzy_match_player_names,
            #     inputs=["PLAYER_MATCH_LOG", "FPL_DATA", "FUZZY_MATCH_OVERRIDES"],
            #     outputs="PLAYER_NAME_MAPPING",
            # ),
            node(
                func=clean_data,
                inputs=[
                    "PLAYER_MATCH_LOG",
                    "TEAM_MATCH_LOG",
                    "ELO_DATA",
                    "FPL_DATA",
                    "PLAYER_NAME_MAPPING",
                    "FPL_2_FBREF_TEAM_MAPPING",
                    "READ_PROCESSED_DATA",
                    "params:data",
                ],
                outputs="cleaned_data",
            ),
            node(
                func=feature_engineering,
                inputs=["cleaned_data", "READ_PROCESSED_DATA", "params:data"],
                outputs="intermediate_data",
            ),
            node(
                func=impute_missing_values,
                inputs=["intermediate_data", "READ_PROCESSED_DATA", "params:data"],
                outputs="PROCESSED_DATA",
            ),
            node(
                func=split_data,
                inputs=["PROCESSED_DATA", "params:data"],
                outputs=[
                    "TRAIN_VAL_DATA",
                    "HOLDOUT_DATA",
                ],
            ),
        ]
    )


def create_model_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=run_housekeeping,
                inputs="params:housekeeping",
                outputs=None,
                name="run_housekeeping",
            ),
            node(
                func=init_experiment,
                inputs="params:model",
                outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
                name="init_experiment",
            ),
            node(
                func=create_sklearn_pipeline,
                inputs=["TRAIN_VAL_DATA", "params:model"],
                outputs="sklearn_pipeline",
                name="create_sklearn_pipeline",
            ),
            node(
                func=pycaret_compare_models,
                inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:model"],
                outputs="PYCARET_RESULT",
                name="pycaret_compare_models",
            ),
            # node(
            #     func=model_selection,
            #     inputs="params:model",
            #     outputs="model",
            #     name="model_selection",
            # ),
            # node(
            #     func=cross_validation,
            #     inputs=[
            #         "TRAIN_VAL_DATA",
            #         "model",
            #         "sklearn_pipeline",
            #         "experiment_id",
            #         "start_time",
            #         "params:model",
            #     ],
            #     outputs=[
            #         "val_score",
            #         "TRAIN_METRICS",
            #         "LAST_FOLD_EVALUATION_PLOTS",
            #     ],
            #     name="cross_validation",
            # ),
            # node(
            #     func=train_model,
            #     inputs=["TRAIN_VAL_DATA", "model", "sklearn_pipeline", "params:model"],
            #     outputs=["FITTED_MODEL", "FITTED_SKLEARN_PIPELINE"],
            #     name="train_model",
            # ),
            # node(
            #     func=evaluate_model_holdout,
            #     inputs=[
            #         "TRAIN_VAL_DATA",
            #         "HOLDOUT_DATA",
            #         "FITTED_MODEL",
            #         "FITTED_SKLEARN_PIPELINE",
            #         "experiment_id",
            #         "start_time",
            #         "params:model",
            #     ],
            #     outputs=[
            #         "HOLDOUT_EVALUATION_RESULT",
            #         "HOLDOUT_EVALUATION_PLOTS",
            #         "HOLDOUT_METRICS",
            #     ],
            #     name="evaluate_model_holdout",
            # ),
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
