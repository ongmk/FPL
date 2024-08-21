from kedro.pipeline import node, pipeline

from fpl.pipelines.scraping.scraper import (
    crawl_fpl_data,
    crawl_player_match_logs,
    crawl_team_match_logs,
)

scraping_pipeline = pipeline(
    [
        node(
            func=crawl_fpl_data,
            inputs=["params:scraping"],
            outputs=None,
        ),
        node(
            func=crawl_team_match_logs,
            inputs=["params:scraping"],
            outputs=None,
        ),
        node(
            func=crawl_player_match_logs,
            inputs=["params:scraping"],
            outputs=None,
        ),
    ]
)


if __name__ == "__main__":
    # crawl_team_match_logs()
    crawl_player_match_logs()
