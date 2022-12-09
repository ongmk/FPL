import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from logger import logger
from BaseDriver import BaseDriver


class OddsPortalDriver(BaseDriver):
    """Custom web driver for FBRef.com"""

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
            logger.info(f"Crawling Season Result:\t{season}\t{url}")
            self.get(url)
            tree = self.get_tree_by_id("tournamentTable")
            # if tree.xpath("//*[@id='emptyMsg']") != []:
            if i == 2:
                return [
                    (self.get_table_text(l), self.absolute_url(l.get("href")))
                    for l in links
                ]
            links += tree.xpath("//tbody/tr/td[2]/a")
            i += 1


if __name__ == "__main__":
    import tqdm
    from pprint import pprint

    seasons = [2021 - i for i in range(10)]
    seasons = [f"{s}-{s+1}" for s in seasons]

    logger.info(f"Initializing Chrome Webdriver...")
    with OddsPortalDriver() as d:
        s = seasons[0]
        season_match_links = d.get_match_links(s)
        pprint(season_match_links)
