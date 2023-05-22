"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from src.fpl.pipelines.optimization_pipeline import (
    create_live_pipeline,
    create_backtest_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    live_optimization_pipeline = create_live_pipeline()
    backtest_optimization_pipeline = create_backtest_pipeline()

    pipelines["live_optimization_pipeline"] = live_optimization_pipeline
    pipelines["backtest_optimization_pipeline"] = backtest_optimization_pipeline

    return pipelines
