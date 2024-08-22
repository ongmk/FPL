from kedro.pipeline import Pipeline, node

from fpl.pipelines.modelling.experiment_helpers import init_experiment, run_housekeeping
from fpl.pipelines.modelling.modelling.dnp_prediction import (
    evaluate_dnp_model_holdout,
    process_dnp_data,
)
from fpl.pipelines.modelling.modelling.ensemble import model_selection
from fpl.pipelines.modelling.modelling.training import (
    create_sklearn_pipeline,
    train_model,
)
from fpl.pipelines.preprocessing.preprocessor import split_data

dnp_preprocessing_pipeline = Pipeline(
    [
        node(
            func=process_dnp_data,
            inputs=["PROCESSED_DATA", "params:dnp_prediction"],
            outputs="dnp_data",
        ),
        node(
            func=split_data,
            inputs=["dnp_data", "params:preprocessing", "params:dnp_prediction"],
            outputs=["DNP_TRAIN_VAL_DATA", "DNP_HOLDOUT_DATA"],
        ),
    ]
)

dnp_training_pipeline = Pipeline(
    [
        node(
            func=create_sklearn_pipeline,
            inputs=["DNP_TRAIN_VAL_DATA", "params:dnp_prediction"],
            outputs="dnp_sklearn_pipeline",
        ),
        node(
            func=model_selection,
            inputs="params:dnp_prediction",
            outputs="dnp_model",
        ),
        node(
            func=train_model,
            inputs=[
                "DNP_TRAIN_VAL_DATA",
                "dnp_model",
                "dnp_sklearn_pipeline",
                "params:dnp_prediction",
            ],
            outputs=["DNP_FITTED_MODEL", "DNP_FITTED_SKLEARN_PIPELINE"],
        ),
    ]
)


dnp_inference_evaluation_pipeline = Pipeline(
    [
        node(
            func=run_housekeeping,
            inputs="params:housekeeping",
            outputs=None,
        ),
        node(
            func=init_experiment,
            inputs="params:dnp_prediction",
            outputs=[
                "dnp_experiment_id",
                "dnp_prediction_start_time",
                "DNP_EXPERIMENT_RECORD",
            ],
        ),
        node(
            func=evaluate_dnp_model_holdout,
            inputs=[
                "DNP_TRAIN_VAL_DATA",
                "DNP_HOLDOUT_DATA",
                "DNP_FITTED_MODEL",
                "DNP_FITTED_SKLEARN_PIPELINE",
                "dnp_experiment_id",
                "dnp_prediction_start_time",
                "params:dnp_prediction",
            ],
            outputs=[
                "DNP_INFERENCE_RESULTS",
                "DNP_EVALUATION_PLOTS",
                "DNP_EXPERIMENT_METRICS",
            ],
        ),
    ]
)
