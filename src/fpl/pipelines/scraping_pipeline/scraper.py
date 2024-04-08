import logging
import re
import sqlite3
from typing import Any, Tuple

import pandas as pd

# from src.fpl.pipelines.scraping_pipeline.OddsPortalDriver import OddsPortalDriver
from src.fpl.pipelines.optimization_pipeline.fpl_api import get_current_season_fpl_data
from src.fpl.pipelines.scraping_pipeline.FBRefDriver import FBRefDriver
from tqdm import tqdm

logger = logging.getLogger(__name__)


def crawl_team_match_logs(parameters: dict[str, Any]):
    current_season = parameters["current_season"]
    current_year = int(re.findall(r"\d+", current_season)[0])
    if parameters["fresh_start"]:
        seasons = [i for i in range(2016, current_year + 1)]
    else:
        seasons = [current_year]
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
    current_season = parameters["current_season"]
    current_year = int(re.findall(r"\d+", current_season)[0])
    if parameters["fresh_start"]:
        seasons = [i for i in range(2016, current_year + 1)]
    else:
        seasons = [current_year]
    seasons = [f"{s}-{s+1}" for s in seasons]

    conn = sqlite3.connect("./data/fpl.db")
    logger.info(f"Initializing FBRefDriver...")

    cur = conn.cursor()
    cur.execute(f"DELETE FROM raw_player_match_log WHERE season = '{current_season}'")
    conn.commit()
    logger.info(f"Deleting player logs from previous weeks.")

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
                    logger.warning(f"{s} {player}\tMatch Log already crawled.\t{link}")
                    continue

                match_log_df = d.get_player_match_log(s, player, pos, link)

                logger.info(f"Saving Match Log:\t{s} {player}")
                match_log_df.to_sql(
                    "raw_player_match_log", conn, if_exists="append", index=False
                )
                crawled_df.loc[len(crawled_df)] = [player, s]

    conn.close()
    return True


def crawl_fpl_data(
    old_data: pd.DataFrame, parameters: dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    override_mapping = {
        "Benjamin Chilwell": "Ben Chilwell",
        "Mateo Kovacic": "Mateo Kovačić",
        "Tomas Soucek": "Tomáš Souček",
    }  # Fix unaligned names between seasons
    old_data["full_name"] = (
        old_data["full_name"].map(override_mapping).fillna(old_data["full_name"])
    )

    fpl_history = old_data[~old_data["total_points"].isna()].sort_values(
        by=["season", "round", "element"], ascending=[False, False, True]
    )
    current_season_data = get_current_season_fpl_data(
        current_season=parameters["current_season"]
    )
    past_season_data = old_data[old_data["season"] != parameters["current_season"]]
    new_data = pd.concat([past_season_data, current_season_data])
    return fpl_history, new_data


# if __name__ == "__main__":
# crawl_team_match_logs()
# crawl_player_match_logs()
# crawl_match_odds()