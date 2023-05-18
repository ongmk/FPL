from kedro.pipeline import Pipeline, node, pipeline
from .scraper import crawl_team_match_logs, crawl_player_match_logs, crawl_match_odds


def hello():
    return "World"


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=hello, inputs=None, oiutputs="done", name="hello_world_node")
            # node(
            #     func=crawl_team_match_logs,
            #     inputs=None,
            #     outputs="done",
            #     name="crawl_team_match_log_node",
            # ),
            # node(
            #     func=crawl_player_match_logs,
            #     inputs=None,
            #     outputs="done",
            #     name="crawl_player_match_log_node_node",
            # ),
            # node(
            #     func=crawl_match_odds,
            #     inputs=None,
            #     outputs="done",
            #     name="crawl_match_odds_node",
            # ),
        ]
    )
