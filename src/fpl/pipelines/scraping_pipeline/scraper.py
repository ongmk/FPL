import logging
import re
import sqlite3
from typing import Any, Tuple

import pandas as pd
from tqdm import tqdm

# from src.fpl.pipelines.scraping_pipeline.OddsPortalDriver import OddsPortalDriver
from src.fpl.pipelines.optimization_pipeline.fpl_api import (
    fetch_most_recent_fixture,
    get_current_season_fpl_data,
)
from src.fpl.pipelines.scraping_pipeline.FBRefDriver import FBRefDriver

logger = logging.getLogger(__name__)


def crawl_team_match_logs(parameters: dict[str, Any]):
    current_season = parameters["current_season"]
    current_year = int(re.findall(r"\d+", current_season)[0])
    if parameters["current_season_only"]:
        seasons = [current_year]
    else:
        seasons = [i for i in range(2016, current_year + 1)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")
    logger.info(f"Initializing FBRefDriver...")

    cur = conn.cursor()
    cur.execute(f"DELETE FROM raw_team_match_log WHERE season = '{current_season}'")
    conn.commit()
    logger.info(f"Deleting team logs from previous weeks.")

    with FBRefDriver(headless=parameters["headless"]) as d:
        crawled_df = pd.read_sql(
            "select distinct team, season from raw_team_match_log", conn
        )

        for season in tqdm(seasons, desc="Seasons crawled"):
            team_season_links = d.get_team_season_links(season)

            for team, link in tqdm(
                team_season_links, leave=False, desc="Teams crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["season"] == season) & (crawled_df["team"] == team)
                ].empty:
                    logger.info(f"{season} {team}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_team_match_log(season, team, link)

                logger.info(f"Saving Match Log:\t{season} {team}")
                match_log_df.to_sql(
                    "raw_team_match_log", conn, if_exists="append", index=False
                )
    conn.close()
    return None


def crawl_player_match_logs(parameters: dict[str, Any]):
    current_season = parameters["current_season"]
    current_year = int(re.findall(r"\d+", current_season)[0])
    if parameters["current_season_only"]:
        seasons = [current_year]
    else:
        seasons = [i for i in range(2016, current_year + 1)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")
    cur = conn.cursor()
    cur.execute(f"DELETE FROM raw_player_match_log WHERE season = '{current_season}'")
    conn.commit()
    logger.info(f"Deleting player logs from previous weeks.")

    logger.info(f"Initializing FBRefDriver...")
    with FBRefDriver(headless=parameters["headless"]) as d:
        crawled_df = pd.read_sql(
            "select distinct player, season from raw_player_match_log", conn
        )

        for s in tqdm(seasons, desc="Seasons crawled"):
            player_season_links = d.get_player_season_links(s, current_season)

            for player, pos, link in tqdm(
                player_season_links, leave=False, desc="Players crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["season"] == s) & (crawled_df["player"] == player)
                ].empty:
                    logger.info(f"{s} {player}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_player_match_log(s, player, pos, link)

                logger.info(f"Saving Match Log:\t{s} {player}")
                match_log_df.to_sql(
                    "raw_player_match_log", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [player, s]

    conn.close()
    return None


def crawl_fpl_data(parameters: dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_season = parameters["current_season"]
    conn = sqlite3.connect("./data/fpl.db")
    cur = conn.cursor()
    most_recent_fixture = fetch_most_recent_fixture()["id"]
    cur.execute(
        f"SELECT sum(total_points) IS NOT null FROM raw_fpl_data WHERE fixture = {most_recent_fixture} and season = '{current_season}'"
    )
    already_crawled = cur.fetchone()[0]

    if already_crawled:
        logger.info(
            f"Most recent match was already crawled. Skipping fetching of FPL data."
        )
        return None

    cur.execute(f"DELETE FROM raw_fpl_data WHERE season = '{current_season}'")
    conn.commit()
    logger.info(f"Deleting fpl data from previous weeks.")

    current_season_data = get_current_season_fpl_data(
        current_season=parameters["current_season"]
    )
    current_season_data.to_sql("raw_fpl_data", conn, if_exists="append", index=False)

    fpl_history = pd.read_sql(
        "select *  from raw_fpl_data where total_points IS NOT NULL order by season DESC, round DESC, element",
        conn,
    ).sort_values(by=["season", "round", "element"], ascending=[False, False, True])
    fpl_history.to_csv("data/fpl_history_backup.csv", index=False)

    return None


if __name__ == "__main__":
    # crawl_team_match_logs()
    # crawl_player_match_logs()
    # crawl_match_odds()
    with FBRefDriver(headless=False) as d:
        match_log_df = d.get_team_match_log("season", "team", "link")
    pass
