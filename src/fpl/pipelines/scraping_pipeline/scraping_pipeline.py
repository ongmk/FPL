from kedro.pipeline import Pipeline, node, pipeline

from fpl.pipelines.scraping_pipeline.scraper import (
    align_fpl_player_name,
    crawl_fpl_data,
    crawl_player_match_logs,
    crawl_team_match_logs,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=align_fpl_player_name,
                inputs=["OLD2NEW_FPL_PLAYER_MAPPING", "params:scraper"],
                outputs="fpl_player_names_aligned",
            ),
            # node(
            #     func=crawl_fpl_data,
            #     inputs=["fpl_player_names_aligned", "params:scraper"],
            #     outputs=None,
            # ),
            # node(
            #     func=crawl_team_match_logs,
            #     inputs=["fpl_player_names_aligned", "params:scraper"],
            #     outputs=None,
            # ),
            # node(
            #     func=crawl_player_match_logs,
            #     inputs=["fpl_player_names_aligned", "params:scraper"],
            #     outputs=None,
            # ),
        ]
    )


if __name__ == "__main__":
    # crawl_team_match_logs()
    crawl_player_match_logs()
