import logging

from kedro.pipeline import Pipeline, node

from fpl.pipelines.modelling.experiment_helpers import init_experiment, run_housekeeping
from fpl.pipelines.optimization.backtest import backtest
from fpl.pipelines.optimization.lp_constructor import construct_lp
from fpl.pipelines.optimization.optimizer import get_live_data, solve_lp
from fpl.pipelines.optimization.output_formatting import generate_outputs

logger = logging.getLogger(__name__)


def create_live_pipeline():
    return Pipeline(
        [
            node(
                func=get_live_data,
                inputs=["INFERENCE_RESULTS", "params:optimization"],
                outputs="LP_DATA",
            ),
            node(
                func=construct_lp,
                inputs=[
                    "LP_DATA",
                    "params:optimization",
                ],
                outputs=["lp_keys", "lp_variables", "variable_sums"],
            ),
            node(
                func=solve_lp,
                inputs=[
                    "LP_DATA",
                    "lp_variables",
                    "variable_sums",
                    "params:optimization",
                ],
                outputs=[
                    "solved_lp_variables",
                    "solved_variable_sums",
                    "solution_time",
                ],
            ),
            node(
                func=generate_outputs,
                inputs=[
                    "LP_DATA",
                    "solved_lp_variables",
                    "solution_time",
                    "params:optimization",
                ],
                outputs="OPTIMIZATION_SUMMARY",
            ),
        ]
    )


def create_backtest_pipeline():
    return Pipeline(
        [
            node(
                func=run_housekeeping,
                inputs="params:housekeeping",
                outputs=None,
            ),
            node(
                func=init_experiment,
                inputs="params:optimization",
                outputs=["experiment_id", "start_time", "EXPERIMENT_RECORD"],
            ),
            node(
                func=backtest,
                inputs=[
                    "experiment_id",
                    "PROCESSED_DATA",
                    "READ_ELO_DATA",
                    "TRAIN_VAL_DATA",
                    "FITTED_MODEL",
                    "FITTED_SKLEARN_PIPELINE",
                    "FPL_DATA",
                    "params:optimization",
                    "params:preprocessing",
                    "params:modelling",
                ],
                outputs=["EXPERIMENT_METRICS", "total_actual_points"],
            ),
        ]
    )


# if __name__ == "__main__":
#     import yaml

#     logging.basicConfig(level=logging.INFO)
#     with open("./conf/base/parameters.yml", "r") as file:
#         parameters = yaml.safe_load(file)
#         parameters = parameters["optimization"]

# live_run(options)
