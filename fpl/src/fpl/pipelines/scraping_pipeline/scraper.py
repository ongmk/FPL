import logging
import sqlite3
from typing import Any

import pandas as pd
from src.fpl.pipelines.scraping_pipeline.backtest_mergers import *
from src.fpl.pipelines.scraping_pipeline.FBRefDriver import FBRefDriver
# from src.fpl.pipelines.scraping_pipeline.OddsPortalDriver import OddsPortalDriver
from src.fpl.pipelines.optimization_pipeline.fpl_api import get_current_season_fpl_data
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
    
    
def get_backtest_data():
    encodings = [
        ("2016-2017", "latin-1"),
        ("2017-2018", "latin-1"),
        ("2018-2019", "latin-1"),
        ("2019-2020", "utf-8"),
        ("2020-2021", "utf-8"),
        ("2021-2022", "utf-8"),
        ("2022-2023", "utf-8"),
    ]

    dfs = []
    path = os.getcwd()
    for season, enc in encodings:
        filename = join(path, "data\\raw\\backtest_data", f"{season}_merged_gw.csv")
        data = pd.read_csv(filename, encoding=enc)
        data["season"] = season
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = df[
        [
            "season",
            "name",
            "position",
            "team",
            "assists",
            "bonus",
            "bps",
            "clean_sheets",
            "creativity",
            "element",
            "fixture",
            "goals_conceded",
            "goals_scored",
            "ict_index",
            "influence",
            "kickoff_time",
            "minutes",
            "opponent_team",
            "own_goals",
            "penalties_missed",
            "penalties_saved",
            "red_cards",
            "round",
            "saves",
            "selected",
            "team_a_score",
            "team_h_score",
            "threat",
            "total_points",
            "transfers_balance",
            "transfers_in",
            "transfers_out",
            "value",
            "was_home",
            "yellow_cards",
            "GW",
        ]
    ]

    df = clean_players_name_string(df)
    df = filter_players_exist_latest(df, col="position")
    df = get_team_name(df, "opponent_team")

    return df[
        [
            "season",
            "round",
            "element",
            "full_name",
            "team",
            "position",
            "fixture",
            "opponent_team",
            "opponent_team_name",
            "total_points",
            "was_home",
            "kickoff_time",
            "team_h_score",
            "team_a_score",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "value",
            "transfers_balance",
            "selected",
            "transfers_in",
            "transfers_out",
        ]
    ]


def merge_fpl_data(parameters: dict[str, Any]) -> pd.DataFrame:
    backtest_data = get_backtest_data()
    current_season_data = get_current_season_fpl_data(current_season = parameters["current_season"])
    return pd.concat([backtest_data, current_season_data])


# def crawl_match_odds(parameters: dict[str, Any]):
#     latest_season = parameters["latest_season"]
#     n_seasons = parameters["n_seasons"]
#     seasons = [latest_season - i for i in range(n_seasons)]
#     seasons = [f"{s}-{s+1}" for s in seasons]

#     conn = sqlite3.connect("./data/fpl.db")

#     logger.info(f"Initializing OddsPortalDriver...")
#     with OddsPortalDriver() as d:
#         crawled_df = pd.read_sql(
#             "select distinct h_team, a_team, season from raw_match_odds", conn
#         )
#         for s in tqdm(seasons, desc="Seasons crawled"):
#             match_links = d.get_match_links(s)
#             for match, link in tqdm(match_links, leave=False, desc="Matches crawled"):
#                 h_team, a_team = match.split(" - ")
#                 if not crawled_df.loc[
#                     (crawled_df["season"] == s)
#                     & (crawled_df["h_team"] == h_team)
#                     & (crawled_df["a_team"] == a_team)
#                 ].empty:
#                     logger.warning(f"{s} {match}\tMatch odds already crawled.\t{link}")
#                     continue
#                 match_odds_df = d.get_match_odds_df(
#                     s, h_team, a_team, d.absolute_url(link)
#                 )

#                 logger.info(f"Saving Match Odds:\t{s} {match}")
#                 match_odds_df.to_sql(
#                     "raw_match_odds", conn, if_exists="append", index=False
#                 )
#                 crawled_df.loc[len(crawled_df)] = [h_team, a_team, s]

#     conn.close()


# if __name__ == "__main__":
# crawl_team_match_logs()
# crawl_player_match_logs()
# crawl_match_odds()
