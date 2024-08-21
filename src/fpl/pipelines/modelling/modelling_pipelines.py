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
            inputs=["TRAIN_VAL_DATA", "params:modelling"],
            outputs="sklearn_pipeline",
            name="create_sklearn_pipeline",
        ),
        node(
            func=feature_selection,
            inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:modelling"],
            outputs=["FEATURE_SELECTION_SUMMARY", "FEATURE_SELECTION_PLOTS"],
            name="feature_selection",
        ),
        node(
            func=pca_elbow_method,
            inputs=["TRAIN_VAL_DATA", "params:modelling"],
            outputs="PCA_ELBOW_METHOD_OUTPUT",
            name="pca_elbow_method",
        ),
    ]
)


compare_model_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:modelling"],
            outputs="sklearn_pipeline",
            name="create_sklearn_pipeline",
        ),
        node(
            func=pycaret_compare_models,
            inputs=["TRAIN_VAL_DATA", "sklearn_pipeline", "params:modelling"],
            outputs="PYCARET_RESULT",
            name="pycaret_compare_models",
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
            inputs="params:modelling",
            outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
        ),
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:modelling"],
            outputs="sklearn_pipeline",
        ),
        node(
            func=model_selection,
            inputs="params:modelling",
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
                "params:modelling",
            ],
            outputs=[
                "val_score",
                "EXPERIMENT_METRICS",
                "LAST_FOLD_EVALUATION_PLOTS",
            ],
        ),
    ]
)


training_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["TRAIN_VAL_DATA", "params:modelling"],
            outputs="sklearn_pipeline",
            name="create_sklearn_pipeline",
        ),
        node(
            func=model_selection,
            inputs="params:modelling",
            outputs="model",
            name="model_selection",
        ),
        node(
            func=train_model,
            inputs=[
                "TRAIN_VAL_DATA",
                "model",
                "sklearn_pipeline",
                "params:modelling",
            ],
            outputs=["FITTED_MODEL", "FITTED_SKLEARN_PIPELINE"],
            name="train_model",
        ),
    ]
)


inference_evaluation_pipeline = Pipeline(
    [
        node(
            func=run_housekeeping,
            inputs="params:housekeeping",
            outputs=None,
            name="run_housekeeping",
        ),
        node(
            func=init_experiment,
            inputs="params:modelling",
            outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
            name="init_experiment",
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
                "params:modelling",
            ],
            outputs=[
                "INFERENCE_RESULTS",
                "HOLDOUT_EVALUATION_PLOTS",
                "HOLDOUT_METRICS",
            ],
            name="evaluate_model_holdout",
        ),
    ]
)
