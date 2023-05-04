import sqlite3
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from tqdm import tqdm
from FBRefDriver import FBRefDriver
from OddsPortalDriver import OddsPortalDriver
from logger import logger


def crawl_team_match_logs():
    seasons = [2021 - i for i in range(10)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    logger.info(f"Initializing SQLite connection...")
    conn = sqlite3.connect("../data/fpl.db")

    logger.info(f"Initializing FBRefDriver...")

    with FBRefDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct TEAM, SEASON from TEAM_MATCH_LOG", conn
        )

        for s in tqdm(seasons, desc="Seasons crawled"):
            team_season_links = d.get_team_season_links(s)

            for team, link in tqdm(
                team_season_links, leave=False, desc="Teams crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["SEASON"] == s) & (crawled_df["TEAM"] == team)
                ].empty:
                    logger.warning(f"{s} {team}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_team_match_log(s, team, link)

                logger.info(f"Saving Match Log:\t{s} {team}")
                match_log_df.to_sql(
                    "TEAM_MATCH_LOG", conn, if_exists="append", index=False
                )
    conn.close()


def crawl_player_match_logs():
    seasons = [2021 - i for i in range(10)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    logger.info(f"Initializing SQLite connection...")
    conn = sqlite3.connect("../data/fpl.db")

    logger.info(f"Initializing FBRefDriver...")
    with FBRefDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct PLAYER, SEASON from PLAYER_MATCH_LOG", conn
        )

        for s in tqdm(seasons, desc="Seasons crawled"):
            player_season_links = d.get_player_season_links(s)

            for player, pos, link in tqdm(
                player_season_links, leave=False, desc="Players crawled"
            ):
                if not crawled_df.loc[
                    (crawled_df["SEASON"] == s) & (crawled_df["PLAYER"] == player)
                ].empty:
                    logger.warning(f"{s} {player}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_player_match_log(s, player, pos, link)

                logger.info(f"Saving Match Log:\t{s} {player}")
                match_log_df.to_sql(
                    "PLAYER_MATCH_LOG", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [player, s]

    conn.close()


def crawl_match_odds():
    seasons = [2021 - i for i in range(10)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    logger.info(f"Initializing SQLite connection...")
    conn = sqlite3.connect("../data/fpl.db")

    logger.info(f"Initializing OddsPortalDriver...")
    with OddsPortalDriver() as d:
        crawled_df = pd.read_sql(
            "select distinct H_TEAM, A_TEAM, SEASON from MATCH_ODDS", conn
        )
        for s in tqdm(seasons, desc="Seasons crawled"):
            match_links = d.get_match_links(s)
            for match, link in tqdm(match_links, leave=False, desc="Matches crawled"):
                h_team, a_team = match.split(" - ")
                if not crawled_df.loc[
                    (crawled_df["SEASON"] == s)
                    & (crawled_df["H_TEAM"] == h_team)
                    & (crawled_df["A_TEAM"] == a_team)
                ].empty:
                    logger.warning(f"{s} {match}\tMatch odds already crawled.\t{link}")
                    continue

                match_odds_df = d.get_match_odds_df(s, h_team, a_team, link)

                logger.info(f"Saving Match Odds:\t{s} {match}")
                match_odds_df.to_sql(
                    "MATCH_ODDS", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [h_team, a_team, s]

    conn.close()


if __name__ == "__main__":
    # crawl_team_match_logs()
    # crawl_player_match_logs()
    crawl_match_odds()
