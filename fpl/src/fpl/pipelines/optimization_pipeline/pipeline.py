from kedro.pipeline import Pipeline, node, pipeline
from .optimizer import live_run


def create_pipeline():
    return Pipeline(
        [
            node(
                func=live_run,
                inputs="params:optimization",
                outputs=["PICKS_SUMMARY", "PICKS_CSV"],
                name="live_run_node",
            ),
        ]
    )
