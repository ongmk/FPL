"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from fpl.pipelines.init_db.init_db_pipelines import init_db_pipeline
from fpl.pipelines.modelling.dnp_prediction_pipelines import (
    dnp_inference_evaluation_pipeline,
    dnp_preprocessing_pipeline,
    dnp_training_pipeline,
)
from fpl.pipelines.modelling.modelling_pipelines import (
    compare_model_pipeline,
    feature_selection_pipeline,
    hypertuning_pipeline,
    inference_evaluation_pipeline,
    training_pipeline,
)
from fpl.pipelines.optimization.optimization_pipelines import (
    backtest_pipeline,
    live_optimization_pipeline,
)
from fpl.pipelines.preprocessing.preprocessing_pipelines import preprocessing_pipeline
from fpl.pipelines.scraping.scraping_pipelines import scraping_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    preprocessing_both = preprocessing_pipeline + dnp_preprocessing_pipeline
    training_both = training_pipeline + dnp_training_pipeline
    inference_evaluation_both = (
        inference_evaluation_pipeline + dnp_inference_evaluation_pipeline
    )

    return {
        "scraping_pipeline": scraping_pipeline,
        "preprocessing_pipeline": preprocessing_both,
        "feature_selection_pipeline": feature_selection_pipeline,
        "compare_model_pipeline": compare_model_pipeline,
        "hypertuning_pipeline": hypertuning_pipeline,
        "training_pipeline": training_both,
        "inference_evaluation_pipeline": inference_evaluation_both,
        "init_db_pipeline": init_db_pipeline,
        "backtest_optimization_pipeline": backtest_pipeline,
        "live_optimization_pipeline": live_optimization_pipeline,
        "end_to_end_pipeline": scraping_pipeline
        + preprocessing_both
        + inference_evaluation_both
        + live_optimization_pipeline,
    }
