import logging
import re
import sqlite3
from typing import Any, Tuple

import pandas as pd
from tqdm import tqdm

# from fpl.pipelines.scraping_pipeline.OddsPortalDriver import OddsPortalDriver
from fpl.pipelines.optimization.fpl_api import (
    get_current_season_fpl_data,
    get_most_recent_fpl_game,
)
from fpl.pipelines.scraping.FBRefDriver import FBRefDriver

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
    cur = conn.cursor()
    logger.info(f"Initializing FBRefDriver...")

    with FBRefDriver(headless=parameters["headless"]) as d:

        date, home, away = d.get_most_recent_game(current_season)
        crawled = pd.read_sql(
            f"""select * from raw_team_match_log 
                where comp = 'Premier League'
                and date = '{date}'
                and team = '{home}'
                and opponent = '{away}'
            """,
            conn,
        )

        if len(crawled) > 0:
            logger.info(
                f"Most recent match was already crawled. Skipping fetching of team match logs."
            )
            return None

        cur.execute(f"DELETE FROM raw_team_match_log WHERE season = '{current_season}'")
        conn.commit()
        logger.info(f"Deleting team logs from previous weeks.")

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

    logger.info(f"Initializing FBRefDriver...")
    with FBRefDriver(headless=parameters["headless"]) as d:

        date, home, away = d.get_most_recent_game(current_season)
        crawled = pd.read_sql(
            f"""select * from raw_player_match_log 
                where comp = 'Premier League'
                and date = '{date}'
                and squad = '{home}'
                and opponent = '{away}'
            """,
            conn,
        )

        if len(crawled) > 5:
            logger.info(
                f"Most recent match was already crawled. Skipping fetching of player match logs."
            )
            return None

        cur.execute(
            f"DELETE FROM raw_player_match_log WHERE season = '{current_season}'"
        )
        conn.commit()
        logger.info(f"Deleting player logs from previous weeks.")

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
    most_recent_fixture = get_most_recent_fpl_game() or {"id": -1}
    crawled = pd.read_sql(
        f"select * from raw_fpl_data where fixture = {most_recent_fixture['id']} and season = '{current_season}'",
        conn,
    )

    if len(crawled) > 0:
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
        match_log_df = d.get_player_match_log(
            "season",
            "player",
            "REF",
            "https://fbref.com/en/players/5f09991f/matchlogs/2017-2018/Patrick-van-Aanholt-Match-Logs",
        )
        print(match_log_df)
    pass
