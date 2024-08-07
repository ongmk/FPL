import logging
from datetime import datetime

import matplotlib.pyplot as plt
from kedro.pipeline import Pipeline, node
from tqdm import tqdm

from fpl.pipelines.optimization.fpl_api import get_live_data
from fpl.pipelines.optimization.lp_constructor import construct_lp
from fpl.pipelines.optimization.optimizer import backtest_single_player, solve_lp
from fpl.pipelines.optimization.output_formatting import generate_outputs

logger = logging.getLogger(__name__)


def backtest(parameters):
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    players = parameters["backtest_players"]
    plots = {}
    for p, id in tqdm(players.items()):
        parameters["team_id"] = id
        filename, fig = backtest_single_player(parameters, p)
        plots[f"{start_time}__{filename}"] = fig
    plt.close("all")
    return plots


def create_live_pipeline():
    return Pipeline(
        [
            node(
                func=get_live_data,
                inputs=["INFERENCE_RESULTS", "params:optimization"],
                outputs="LP_DATA",
                name="get_live_data",
            ),
            node(
                func=construct_lp,
                inputs=[
                    "LP_DATA",
                    "params:optimization",
                ],
                outputs=["lp_params", "lp_keys", "lp_variables", "variable_sums"],
                name="construct_lp",
            ),
            node(
                func=solve_lp,
                inputs=["lp_variables", "variable_sums", "params:optimization"],
                outputs=[
                    "solved_lp_variables",
                    "solved_variable_sums",
                    "solution_time",
                ],
                name="solve_lp",
            ),
            node(
                func=generate_outputs,
                inputs=[
                    "LP_DATA",
                    "solved_lp_variables",
                    "solution_time",
                    "params:optimization",
                ],
                outputs=["PICKS_SUMMARY", "PICKS_CSV", "next_gw_dict"],
                name="live_optimization_node",
            ),
        ]
    )


def create_backtest_pipeline():
    return Pipeline(
        [
            node(
                func=backtest,
                inputs="params:optimization",
                outputs="BACKTEST_PLOTS",
                name="backtest_optimization_node",
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
