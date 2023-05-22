from kedro.pipeline import Pipeline, node, pipeline
from datetime import datetime

from src.fpl.pipelines.model_pipeline.elo_calculation import (
    calculate_elo_score,
    xg_elo_correlation,
)
from src.fpl.pipelines.model_pipeline.preprocessor import preprocess_data
from src.fpl.pipelines.model_pipeline.training import train_model, split_data
from src.fpl.pipelines.model_pipeline.evaluation import evaluate_model
from src.fpl.pipelines.model_pipeline.housekeeping import run_housekeeping
import pandas as pd
import sqlite3
from flatten_dict import flatten


def snake_to_camel(snake_str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_reducer(k1, k2):
    if k1 is None:
        return snake_to_camel(k2)
    else:
        k1 = snake_to_camel(k1)
        k2 = snake_to_camel(k2)
        return f"{k1}_{k2}"


def init_experiment(parameters):
    conn = sqlite3.connect("./data/fpl.db")
    query = "select COALESCE(max(id)  + 1 , 0) from experiment;"
    cursor = conn.execute(query)
    id = cursor.fetchone()[0]
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cursor.close()
    conn.close()

    record = flatten(parameters, reducer=camel_reducer)
    record = {
        key: ", ".join(sorted(value)) if isinstance(value, list) else value
        for key, value in record.items()
    }

    record["id"] = id
    record["start_time"] = start_time
    experiment_record = pd.DataFrame.from_records([record])

    return id, start_time, experiment_record


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=init_experiment,
                inputs="params:model",
                outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
                name="init_experiment_node",
            ),
            # node(
            #     func=calculate_elo_score,
            #     inputs=["TEAM_MATCH_LOG", "params:data"],
            #     outputs="ELO_DATA",
            #     name="elo_score_node",
            # ),
            # node(
            #     func=preprocess_data,
            #     inputs=["TEAM_MATCH_LOG", "ELO_DATA", "ODDS_DATA", "params:data"],
            #     outputs="PROCESSED_DATA",
            #     name="preprocess_node",
            # ),
            # node(
            #     func=xg_elo_correlation,
            #     inputs=["PROCESSED_DATA", "params:data"],
            #     outputs="correlation",
            # ),
            node(
                func=split_data,
                inputs=["PROCESSED_DATA", "params:data"],
                outputs=[
                    "train_val_data",
                    "holdout_data",
                ],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["train_val_data", "params:model"],
                outputs=["model", "encoder"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "holdout_data",
                    "model",
                    "encoder",
                    "experiment_id",
                    "start_time",
                    "params:model",
                ],
                outputs=["EVALUATION_RESULT", "EVALUATION_PLOTS", "loss"],
                name="evaluation_node",
            ),
            node(
                func=run_housekeeping,
                inputs="params:housekeeping",
                outputs=None,
                name="housekeeping_node",
            ),
        ]
    )
