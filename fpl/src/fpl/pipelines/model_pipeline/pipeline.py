from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_elo_score,
    preprocess_data,
    split_data,
    train_model,
    evaluate_model,
    xg_elo_correlation,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_elo_score,
                inputs=["MATCH_DATA", "params:data"],
                outputs="ELO_DATA",
                name="elo_score_node",
            ),
            node(
                func=preprocess_data,
                inputs=["TEAM_MATCH_LOG", "ELO_DATA", "ODDS_DATA", "params:data"],
                outputs="PROCESSED_DATA",
                name="preprocess_node",
            ),
            # node(
            #     func=xg_elo_correlation,
            #     inputs=["PROCESSED_DATA", "params:data"],
            #     outputs="correlation",
            # )
            # node(
            #     func=split_data,
            #     inputs="train_data",
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split_data_node",
            # ),
            # node(
            #     func=train_model,
            #     inputs=["X_train", "y_train"],
            #     outputs="model",
            #     name="train_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["model", "X_test", "y_test", "X_train", "y_train"],
            #     outputs="score",
            #     name="evaluation_node",
            # ),
        ]
    )
