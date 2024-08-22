from kedro.pipeline import Pipeline, node

from fpl.pipelines.modelling.experiment_helpers import init_experiment, run_housekeeping
from fpl.pipelines.modelling.modelling.ensemble import model_selection
from fpl.pipelines.modelling.modelling.evaluation import evaluate_model_holdout
from fpl.pipelines.modelling.modelling.training import (
    create_sklearn_pipeline,
    cross_validation,
    feature_selection,
    pca_elbow_method,
    pycaret_compare_models,
    train_model,
)

feature_selection_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:points_prediction"],
            outputs="sklearn_pipeline",
        ),
        node(
            func=feature_selection,
            inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:points_prediction"],
            outputs=["FEATURE_SELECTION_SUMMARY", "FEATURE_SELECTION_PLOTS"],
        ),
        node(
            func=pca_elbow_method,
            inputs=["TRAIN_VAL_DATA", "params:points_prediction"],
            outputs="PCA_ELBOW_METHOD_OUTPUT",
        ),
    ]
)


compare_model_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:points_prediction"],
            outputs="sklearn_pipeline",
        ),
        node(
            func=pycaret_compare_models,
            inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:points_prediction"],
            outputs="PYCARET_RESULT",
        ),
    ]
)


hypertuning_pipeline = Pipeline(
    [
        node(
            func=run_housekeeping,
            inputs="params:housekeeping",
            outputs=None,
        ),
        node(
            func=init_experiment,
            inputs="params:points_prediction",
            outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
        ),
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:points_prediction"],
            outputs="sklearn_pipeline",
        ),
        node(
            func=model_selection,
            inputs="params:points_prediction",
            outputs="model",
        ),
        node(
            func=cross_validation,
            inputs=[
                "TRAIN_VAL_DATA",
                "model",
                "sklearn_pipeline",
                "experiment_id",
                "start_time",
                "params:points_prediction",
            ],
            outputs=[
                "val_score",
                "EXPERIMENT_METRICS",
                "EVALUATION_PLOTS",
            ],
        ),
    ]
)


training_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:points_prediction"],
            outputs="sklearn_pipeline",
        ),
        node(
            func=model_selection,
            inputs="params:points_prediction",
            outputs="points_model",
        ),
        node(
            func=train_model,
            inputs=[
                "TRAIN_VAL_DATA",
                "points_model",
                "sklearn_pipeline",
                "params:points_prediction",
            ],
            outputs=["FITTED_MODEL", "FITTED_SKLEARN_PIPELINE"],
        ),
    ]
)


inference_evaluation_pipeline = Pipeline(
    [
        node(
            func=run_housekeeping,
            inputs="params:housekeeping",
            outputs=None,
        ),
        node(
            func=init_experiment,
            inputs="params:points_prediction",
            outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
        ),
        node(
            func=evaluate_model_holdout,
            inputs=[
                "TRAIN_VAL_DATA",
                "HOLDOUT_DATA",
                "FITTED_MODEL",
                "FITTED_SKLEARN_PIPELINE",
                "experiment_id",
                "start_time",
                "params:points_prediction",
            ],
            outputs=[
                "INFERENCE_RESULTS",
                "EVALUATION_PLOTS",
                "EXPERIMENT_METRICS",
            ],
        ),
    ]
)
