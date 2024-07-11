"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from fpl.pipelines.init_db_pipeline import create_pipeline as create_init_db_pipeline
from fpl.pipelines.model_pipeline import (
    create_compare_model_pipeline,
    create_feature_selection_pipeline,
    create_hypertuning_pipeline,
    create_inference_evaluation_pipeline,
    create_training_pipeline,
)
from fpl.pipelines.optimization_pipeline import (
    create_backtest_pipeline,
    create_live_pipeline,
)
from fpl.pipelines.preprocess_pipeline import create_preprocess_pipeline
from fpl.pipelines.scraping_pipeline import create_pipeline as create_scraping_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "scraping_pipeline": create_scraping_pipeline(),
        "preprocess_pipeline": create_preprocess_pipeline(),
        "feature_selection_pipeline": create_feature_selection_pipeline(),
        "compare_model_pipeline": create_compare_model_pipeline(),
        "hypertuning_pipeline": create_hypertuning_pipeline(),
        "training_pipeline": create_training_pipeline(),
        "inference_evaluation_pipeline": create_inference_evaluation_pipeline(),
        "init_db_pipeline": create_init_db_pipeline(),
        "live_optimization_pipeline": create_live_pipeline(),
        "backtest_optimization_pipeline": create_backtest_pipeline(),
    }
