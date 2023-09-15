from kedro.pipeline import Pipeline, node, pipeline
from src.fpl.pipelines.scraping_pipeline.scraper import (
    crawl_player_match_logs,
    crawl_team_match_logs,
    merge_fpl_data,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_fpl_data,
                inputs=["OLD_FPL_DATA", "params:scraper"],
                outputs=["FPL_HISTORY_BACKUP", "FPL_DATA"],
            ),
            node(
                func=crawl_team_match_logs,
                inputs=["FPL_HISTORY_BACKUP", "params:scraper"],
                outputs=None,
            ),
            node(
                func=crawl_player_match_logs,
                inputs=["FPL_HISTORY_BACKUP", "params:scraper"],
                outputs=None,
            ),
        ]
    )


if __name__ == "__main__":
    # crawl_team_match_logs()
    crawl_player_match_logs()