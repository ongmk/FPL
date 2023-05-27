from kedro.pipeline import Pipeline, node, pipeline

from src.fpl.pipelines.model_pipeline.elo_calculation import (
    calculate_elo_score,
    xg_elo_correlation,
)
from src.fpl.pipelines.model_pipeline.preprocessor import preprocess_data
from src.fpl.pipelines.model_pipeline.training import (
    split_data,
    create_sklearn_pipeline,
    pycaret_compare_models,
    model_selection,
    cross_validation,
    train_model,
)
from src.fpl.pipelines.model_pipeline.evaluation import evaluate_model
from src.fpl.pipelines.model_pipeline.experiment_helpers import (
    run_housekeeping,
    init_experiment,
)
import pandas as pd
import sqlite3


def create_preprocess_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_elo_score,
                inputs=["TEAM_MATCH_LOG", "params:data"],
                outputs="ELO_DATA",
                name="elo_score_node",
            ),
            node(
                func=preprocess_data,
                inputs=["TEAM_MATCH_LOG", "ELO_DATA", "ODDS_DATA", "params:data"],
                outputs="PROCESSED_DATA",
                name="preprocess_node",
            ),
            node(
                func=xg_elo_correlation,
                inputs=["PROCESSED_DATA", "params:data"],
                outputs="correlation",
            ),
            node(
                func=split_data,
                inputs=["PROCESSED_DATA", "params:data"],
                outputs=[
                    "TRAIN_VAL_DATA",
                    "HOLDOUT_DATA",
                ],
                name="split_data_node",
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
                name="housekeeping_node",
            ),
            node(
                func=init_experiment,
                inputs="params:model",
                outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
                name="init_experiment_node",
            ),
            node(
                func=create_sklearn_pipeline,
                inputs=["TRAIN_VAL_DATA", "params:model"],
                outputs="sklearn_pipeline",
                name="create_sklearn_pipeline_node",
            ),
            # node(
            #     func=pycaret_compare_models,
            #     inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:model"],
            #     outputs="PYCARET_RESULT",
            #     name="pycaret_compare_models_node",
            # ),
            node(
                func=model_selection,
                inputs="params:model",
                outputs="models",
                name="model_selection_node",
            ),
            node(
                func=cross_validation,
                inputs=["TRAIN_VAL_DATA", "models", "sklearn_pipeline", "params:model"],
                outputs="validation_loss",
                name="cross_validation_node",
            ),
            node(
                func=train_model,
                inputs=["TRAIN_VAL_DATA", "models", "sklearn_pipeline", "params:model"],
                outputs=["FITTED_MODELS", "fitted_sklearn_pipeline"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "TRAIN_VAL_DATA",
                    "HOLDOUT_DATA",
                    "FITTED_MODELS",
                    "fitted_sklearn_pipeline",
                    "experiment_id",
                    "start_time",
                    "params:model",
                ],
                outputs=[
                    "holdout_loss",
                    "EVALUATION_RESULT",
                    "EVALUATION_PLOTS",
                    "EVALUATION_METRICS",
                ],
                name="evaluation_node",
            ),
        ]
    )


if __name__ == "__main__":
    import sqlite3
    import pandas as pd
    import yaml

    # Connect to the SQLite database
    connection = sqlite3.connect("./data/fpl.db")
    train_val_data = pd.read_sql_query("SELECT * FROM train_val_data", connection)
    with open("./conf/base/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)
        parameters = parameters["model"]

    sklearn_pipeline = create_sklearn_pipeline(
        train_val_data=train_val_data, parameters=parameters
    )
    models = model_selection(parameters)
    scores = cross_validation(train_val_data, models, sklearn_pipeline, parameters)
    trained_models = train_model(
        train_val_data,
        models,
        sklearn_pipeline,
        parameters,
    )

    holdout_data = pd.read_sql_query("SELECT * FROM holdout_data", connection)
    outputs = evaluate_model(
        holdout_data=holdout_data,
        models=trained_models,
        sklearn_pipeline=sklearn_pipeline,
        experiment_id=1,
        start_time="now",
        parameters=parameters,
    )
    print(outputs)
