import logging

from kedro.pipeline import Pipeline, node

from fpl.pipelines.optimization.lp_constructor import construct_lp
from fpl.pipelines.optimization.optimizer import backtest, get_live_data, solve_lp
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
                outputs=["lp_params", "lp_keys", "lp_variables", "variable_sums"],
            ),
            node(
                func=solve_lp,
                inputs=["lp_variables", "variable_sums", "params:optimization"],
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
                func=backtest,
                inputs=["INFERENCE_RESULTS", "FPL_DATA", "params:optimization"],
                outputs="BACKTEST_PLOTS",
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
