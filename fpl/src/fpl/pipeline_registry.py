"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from src.fpl.pipelines.init_db_pipeline import (
    create_pipeline as create_init_db_pipeline,
)
from src.fpl.pipelines.model_pipeline import (
    create_compare_model_pipeline,
    create_feature_selection_pipeline,
    create_model_pipeline,
    create_preprocess_pipeline,
)
from src.fpl.pipelines.optimization_pipeline import (
    create_backtest_pipeline,
    create_live_pipeline,
)
from src.fpl.pipelines.scraping_pipeline import (
    create_pipeline as create_scraping_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "live_optimization_pipeline": create_live_pipeline(),
        "backtest_optimization_pipeline": create_backtest_pipeline(),
        "preprocess_pipeline": create_preprocess_pipeline(),
        "feature_selection_pipeline": create_feature_selection_pipeline(),
        "compare_model_pipeline": create_compare_model_pipeline(),
        "model_pipeline": create_model_pipeline(),
        "init_db_pipeline": create_init_db_pipeline(),
        "scraping_pipeline": create_scraping_pipeline(),
    }
