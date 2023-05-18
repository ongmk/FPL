from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    get_start_time,
    calculate_elo_score,
    preprocess_data,
    split_data,
    train_model,
    evaluate_model,
    xg_elo_correlation,
    run_housekeeping,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_start_time,
                inputs=None,
                outputs="start_time",
                name="get_start_time_node",
            ),
            # node(
            #     func=calculate_elo_score,
            #     inputs=["MATCH_DATA", "params:data"],
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
            # )
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
                    "start_time",
                    "params:model",
                ],
                outputs=["EVALUATION_RESULT", "EVALUATION_PLOTS", "loss"],
                name="evaluation_node",
            ),
            node(
                func=run_housekeeping,
                inputs=[
                    "loss",
                    "params:housekeeping",
                ],
                outputs="pipeline_output",
                name="housekeeping_node",
            ),
        ]
    )
