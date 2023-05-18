import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from logger import logger
from BaseDriver import BaseDriver


class OddsPortalDriver(BaseDriver):
    """Custom web driver for OddsPortal.com"""

    def __init__(self):
        BaseDriver.__init__(self)
        self.base_url = "https://www.oddsportal.com/"

    def get_match_links(self, season):
        i = 1
        links = []
        while True:
            relative_url = (
                f"/soccer/england/premier-league-{season}/results/#/page/{i}/"
            )
            url = self.absolute_url(relative_url)
            logger.info(f"Crawling Season Match Links:\t{season}\t{url}")
            self.get(url)
            tree = self.get_tree_by_id("tournamentTable")
            if tree.xpath("//*[@id='emptyMsg']") != []:
                return [
                    (self.get_table_text(l), self.absolute_url(l.get("href")))
                    for l in links
                ]
            links += tree.xpath("//tbody/tr/td[2]/a")
            i += 1

    def get_row_data(self, row):
        h_score, a_score = row.xpath("./strong/a")[0].text.split(":")
        odds = row.xpath("./span[@class='avg nowrp']/a")
        if len(odds) > 0:
            odds = float(odds[0].text)
        else:
            odds = None
        return int(h_score), int(a_score), odds

    def get_match_odds_df(self, season, h_team, a_team, link):
        logger.info(f"Crawling Match Odds:\t{season} {h_team}-{a_team}\t{link}")
        link = link + "#cs;2"
        self.get(link)
        table = self.get_tree_by_id("odds-data-table")
        rows = table.xpath(
            '//div[@class="table-container"][not (@style = "display: none;")]/div'
        )
        rows = [(self.get_row_data(r)) for r in rows]
        match_odds_df = pd.DataFrame(rows, columns=["h_score", "a_score", "odds"])
        match_odds_df = match_odds_df.sort_values("odds").head(10)
        match_odds_df["Season"] = season
        match_odds_df["h_team"] = h_team
        match_odds_df["a_team"] = a_team
        match_odds_df["Link"] = link

        return match_odds_df


# if __name__ == "__main__":
#     import tqdm
#     from pprint import pprint

#     seasons = [2021 - i for i in range(10)]
#     seasons = [f"{s}-{s+1}" for s in seasons]

#     logger.info(f"Initializing Chrome Webdriver...")
#     with OddsPortalDriver() as d:
#         s = seasons[0]
#         # match_links = d.get_match_links(s)
#         match_links = [
#             (
#                 "Arsenal - Leicester",
#                 "https://www.oddsportal.com/soccer/england/premier-league-2021-2022/arsenal-leicester-nD4wkl9n/",
#             )
#         ]
#         for match, link in match_links:
#             team, opponent = match.split(" - ")
#             match_odds_df = d.get_match_odds_df(s, team, opponent, link)
#             print(match_odds_df)
#             print(match_odds_df.columns)
