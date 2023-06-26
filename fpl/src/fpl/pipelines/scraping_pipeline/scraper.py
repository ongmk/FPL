import logging
import sqlite3
from typing import Any

import pandas as pd
from src.fpl.pipelines.scraping_pipeline.FBRefDriver import FBRefDriver
from src.fpl.pipelines.scraping_pipeline.OddsPortalDriver import OddsPortalDriver
from tqdm import tqdm

logger = logging.getLogger(__name__)


def crawl_team_match_logs(parameters: dict[str, Any]):
    latest_season = parameters["latest_season"]
    n_seasons = parameters["n_seasons"]
    seasons = [latest_season - i for i in range(n_seasons)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")
    logger.info(f"Initializing FBRefDriver...")

    with FBRefDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct team, season from raw_team_match_log", conn
        )

        for s in tqdm(seasons, desc="Seasons crawled"):
            team_season_links = d.get_team_season_links(s)

            for team, link in tqdm(
                team_season_links, leave=False, desc="Teams crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["season"] == s) & (crawled_df["team"] == team)
                ].empty:
                    logger.warning(f"{s} {team}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_team_match_log(s, team, link)

                logger.info(f"Saving Match Log:\t{s} {team}")
                match_log_df.to_sql(
                    "raw_team_match_log", conn, if_exists="append", index=False
                )
    conn.close()
    return True


def crawl_player_match_logs(parameters: dict[str, Any]):
    latest_season = parameters["latest_season"]
    n_seasons = parameters["n_seasons"]
    seasons = [latest_season - i for i in range(n_seasons)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")

    logger.info(f"Initializing FBRefDriver...")
    with FBRefDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct player, season from raw_player_match_log", conn
        )

        for s in tqdm(seasons, desc="Seasons crawled"):
            player_season_links = d.get_player_season_links(s)

            for player, pos, link in tqdm(
                player_season_links, leave=False, desc="Players crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["season"] == s) & (crawled_df["player"] == player)
                ].empty:
                    logger.warning(f"{s} {player}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_player_match_log(s, player, pos, link)

                logger.info(f"Saving Match Log:\t{s} {player}")
                match_log_df.to_sql(
                    "raw_player_match_log", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [player, s]

    conn.close()


def crawl_match_odds(parameters: dict[str, Any]):
    latest_season = parameters["latest_season"]
    n_seasons = parameters["n_seasons"]
    seasons = [latest_season - i for i in range(n_seasons)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")

    logger.info(f"Initializing OddsPortalDriver...")
    with OddsPortalDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct h_team, a_team, season from raw_match_odds", conn
        )
        for s in tqdm(seasons, desc="Seasons crawled"):
            match_links = d.get_match_links(s)
            for match, link in tqdm(match_links, leave=False, desc="Matches crawled"):
                h_team, a_team = match.split(" - ")
                if not crawled_df.loc[
                    (crawled_df["season"] == s)
                    & (crawled_df["h_team"] == h_team)
                    & (crawled_df["a_team"] == a_team)
                ].empty:
                    logger.warning(f"{s} {match}\tMatch odds already crawled.\t{link}")
                    continue
                match_odds_df = d.get_match_odds_df(
                    s, h_team, a_team, d.absolute_url(link)
                )

                logger.info(f"Saving Match Odds:\t{s} {match}")
                match_odds_df.to_sql(
                    "raw_match_odds", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [h_team, a_team, s]

    conn.close()


# if __name__ == "__main__":
# crawl_team_match_logs()
# crawl_player_match_logs()
# crawl_match_odds()
