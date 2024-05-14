from kedro.pipeline import Pipeline, node
from fpl.pipelines.optimization_pipeline.fpl_api import get_live_data
from fpl.pipelines.optimization_pipeline.optimizer import (
    solve_multi_period_fpl,
    backtest_single_player,
)
from fpl.pipelines.optimization_pipeline.fetch_predictions import (
    refresh_fpl_names_mapping,
)
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pandas as pd


logger = logging.getLogger(__name__)


def live_run(parameters: dict) -> tuple[str, pd.DataFrame]:
    refresh_fpl_names_mapping()
    fpl_data = get_live_data(parameters["team_id"], parameters["horizon"])
    picks, summary, next_gw_dict = solve_multi_period_fpl(
        fpl_data=fpl_data, parameters=parameters
    )
    logger.info(f"Solved in {next_gw_dict['solve_time']}")
    for s in summary:
        logger.info(s)
    summary = "\n".join(summary)
    return summary, picks


def backtest(parameters):
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    players = parameters["backtest_players"]
    plots = {}
    refresh_fpl_names_mapping()
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
                func=live_run,
                inputs="params:optimization",
                outputs=["PICKS_SUMMARY", "PICKS_CSV"],
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
