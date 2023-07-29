import logging
import re

import pandas as pd
from lxml import etree
from src.fpl.pipelines.scraping_pipeline.BaseDriver import BaseDriver, DelayedRequests

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
        # pagination_links = container.xpath("//a")
        # hrefs = [element.get(":href") for element in pagination_links]

        # page_1_urls = set()
        # for url in hrefs:
        #     match = re.search(r"/page/(\d+)/", url)
        #     if match:
        #         page_1_url = re.sub(r"/page/\d+/", "/page/1/", url)
        #         page_1_urls.add(page_1_url)
        # if len(page_1_urls) > 1:
        #     raise ValueError("More than one possible page 1 url.")
        # return next(iter(page_1_urls))

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


if __name__ == "__main__":
    with OddsPortalDriver() as d:
        links = d.get_match_links("2022-2023")
        print(links)
