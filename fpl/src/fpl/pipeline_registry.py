"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from src.fpl.pipelines.optimization_pipeline import (
    create_live_pipeline,
    create_backtest_pipeline,
)
from src.fpl.pipelines.model_pipeline import (
    create_preprocess_pipeline,
    create_model_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    live_optimization_pipeline = create_live_pipeline()
    backtest_optimization_pipeline = create_backtest_pipeline()
    preprocess_pipeline = create_preprocess_pipeline()
    model_pipeline = create_model_pipeline()

    pipelines["live_optimization_pipeline"] = live_optimization_pipeline
    pipelines["backtest_optimization_pipeline"] = backtest_optimization_pipeline
    pipelines["preprocess_pipeline"] = preprocess_pipeline
    pipelines["model_pipeline"] = model_pipeline

    return pipelines
