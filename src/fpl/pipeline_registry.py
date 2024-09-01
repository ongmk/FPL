"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from fpl.pipelines.init_db.init_db_pipelines import *
from fpl.pipelines.modelling.dnp_prediction_pipelines import *
from fpl.pipelines.modelling.modelling_pipelines import *
from fpl.pipelines.optimization.optimization_pipelines import *
from fpl.pipelines.preprocessing.preprocessing_pipelines import *
from fpl.pipelines.scraping.scraping_pipelines import *


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    return {
        "scraping_pipeline": scraping_pipeline,
        "preprocessing_pipeline": preprocessing_pipeline + dnp_preprocessing_pipeline,
        "feature_selection_pipeline": feature_selection_pipeline,
        "compare_model_pipeline": compare_model_pipeline,
        "hypertuning_pipeline": hypertuning_pipeline,
        "training_pipeline": training_pipeline + dnp_training_pipeline,
        "inference_evaluation_pipeline": inference_evaluation_pipeline
        + dnp_inference_evaluation_pipeline,
        "init_db_pipeline": init_db_pipeline,
        "backtest_optimization_pipeline": backtest_pipeline,
        "live_optimization_pipeline": live_optimization_pipeline,
        "live_complete_pipeline": scraping_pipeline
        + preprocessing_pipeline
        + inference_evaluation_pipeline
        + live_optimization_pipeline,
    }
