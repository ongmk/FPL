# preprocessor.py


def calculate_score_from_odds(df):
    h_score_sum = (df["h_score"] * (1 / (df["odds"] + 1))).sum()
    a_score_sum = (df["a_score"] * (1 / (df["odds"] + 1))).sum()
    inverse_odds_sum = (1 / (df["odds"] + 1)).sum()
    return pd.Series(
        {
            "h_odds_2_score": h_score_sum / inverse_odds_sum,
            "a_odds_2_score": a_score_sum / inverse_odds_sum,
        }
    )


def aggregate_odds_data(odds_data, parameters):
    agg_odds_data = (
        odds_data.groupby(["season", "h_team", "a_team"])
        .apply(lambda group: calculate_score_from_odds(group))
        .reset_index()
    )
    mapping = pd.read_csv("./src/fpl/pipelines/model_pipeline/team_mapping.csv")
    mapping = mapping.set_index("odds_portal_name")["fbref_name"].to_dict()
    agg_odds_data["h_team"] = agg_odds_data["h_team"].map(mapping)
    agg_odds_data["a_team"] = agg_odds_data["a_team"].map(mapping)

    temp_df = agg_odds_data.copy()
    temp_df.columns = [
        "season",
        "team",
        "opponent",
        "team_odds_2_score",
        "opponent_odds_2_score",
    ]

    # Create two copies of the temporary dataframe, one for home matches and one for away matches
    home_df = temp_df.copy()
    away_df = temp_df.copy()

    # Add a 'venue' column to each dataframe
    home_df["venue"] = "Home"
    away_df["venue"] = "Away"

    # Swap the 'team' and 'opponent' columns in the away_df
    away_df["team"], away_df["opponent"] = away_df["opponent"], away_df["team"]
    away_df["team_odds_2_score"], away_df["opponent_odds_2_score"] = (
        away_df["opponent_odds_2_score"],
        away_df["team_odds_2_score"],
    )

    # Concatenate the two dataframes to get the final unpivoted dataframe
    unpivoted_df = pd.concat([home_df, away_df], ignore_index=True)

    return unpivoted_df


#############################################################################
# catalog

# ODDS_DATA:
#   type: pandas.SQLTableDataSet
#   table_name: raw_match_odds
#   credentials: db_credentials

#############################################################################
# create_tables.sql

# CREATE TABLE IF NOT EXISTS raw_match_odds(
#     season TEXT,
#     h_team TEXT,
#     a_team TEXT,
#     h_score INTEGER,
#     a_score INTEGER,
#     odds REAL,
#     link TEXT
# );

# CREATE UNIQUE INDEX IF NOT EXISTS SEASON_MATCH_SCORE_PK ON raw_match_odds(season, h_team, a_team, h_score, a_score);

#############################################################################
# drop_tables.sql

# DROP TABLE IF EXISTS raw_match_odds;

#############################################################################
# OddsPortalDriver.py
import logging
import re

import pandas as pd
from lxml import etree

from fpl.pipelines.scraping_pipeline.BaseDriver import BaseDriver, DelayedRequests

logger = logging.getLogger(__name__)


class OddsPortalDriver(BaseDriver):
    """Custom web driver for OddsPortal.com"""

    def __init__(self):
        BaseDriver.__init__(self)
        self.base_url = "https://www.oddsportal.com/"
        self.requests = DelayedRequests()

    def get_next_page(self):
        pagination = self.get_tree_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[5]/div[5]/div'
        )
        next_page_button = pagination.xpath('./body/a[text()="Next"]')
        if len(next_page_button) == 1:
            self.click_elements_by_xpath(
                '//*[@id="app"]/div/div[1]/div/main/div[2]/div[5]/div[5]/div/a[text()="Next"]'
            )
            return True
        else:
            return False

    def get_match_links(self, season):
        relative_url = f"/football/england/premier-league-{season}/results/"
        url = self.absolute_url(relative_url)
        self.get(url)

        match_links = []
        while True:
            table = self.get_tree_by_xpath(
                '//*[@id="app"]/div/div[1]/div/main/div[2]/div[5]/div[1]'
            )
            rows = table.xpath("./body/div/div/div/a")
            for r in rows:
                url = r.get("href")
                home, away = r.xpath("./div[2]/div/div/a/div[1]")
                name = f"{home.text} - {away.text}"
                print(name)
                match_links.append((name, url))
            if self.get_next_page():
                continue
            else:
                return match_links

    def get_row_data(self, row):
        h_score, a_score = row.xpath("./div/p[1]")[0].text.split(":")
        odds = row.xpath("./div[3]/div/div/div/p")[0].text
        return int(h_score), int(a_score), float(odds)

    def get_match_odds_df(self, season, h_team, a_team, link):
        link = link + "#cs;2"
        logger.info(f"Crawling Match Odds:\t{season} {h_team}-{a_team}\t{link}")
        self.get(link)
        self.get_tree_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]/div/div/div[3]/div/div/div[contains(@class, "font-bold")]'
        )
        table = self.get_tree_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]'
        )
        rows = table.xpath("./body/div/div[./* and text() != '-']")
        rows = [(self.get_row_data(r)) for r in rows]
        match_odds_df = pd.DataFrame(rows, columns=["h_score", "a_score", "odds"])
        match_odds_df = match_odds_df.sort_values("odds").head(10)
        match_odds_df["Season"] = season
        match_odds_df["h_team"] = h_team
        match_odds_df["a_team"] = a_team
        match_odds_df["Link"] = link

        return match_odds_df

    def expand_rows(self):
        self.click_elements_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]/div/div/div[1]'
        )


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
